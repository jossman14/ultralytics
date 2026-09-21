"""
Baseline sweep: train the native YOLO family on the judol 5-class logo dataset.
Real numbers only — logs mAP + params + GFLOPs per model to results.csv.
ponytail: one loop, per-model try/except so one failure doesn't kill the sweep.
"""
import csv, gc, time, traceback, sys
sys.path.insert(0, "/home/ftib/ultralytics")  # import local ultralytics source, not paper_judol/
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

ROOT = Path("/home/ftib/ultralytics/paper_judol")
DATA = str(ROOT / "judol_v29.yaml")
RUNS = ROOT / "experiments"
CSV = ROOT / "results.csv"
EPOCHS = 150
IMG = 640
BATCH = 16
SEED = 0

# nano/tiny variants for a fair lightweight comparison (proposed is small)
MODELS = [
    "yolov5n.yaml", "yolov6n.yaml", "yolov8n.yaml", "yolov9t.yaml",
    "yolov10n.yaml", "yolo11n.yaml", "yolo12n.yaml",
]

def flops_params(model):
    """Real params + GFLOPs (thop-based via ultralytics helpers)."""
    try:
        return get_num_params(model.model), round(get_flops(model.model, imgsz=IMG), 4)
    except Exception:
        return None, None

def run(cfg):
    name = cfg.replace(".yaml", "")
    t0 = time.time()
    model = YOLO(cfg)  # build from scratch (no pretrain) for fair architecture comparison
    params, gflops = flops_params(model)
    model.train(data=DATA, epochs=EPOCHS, imgsz=IMG, batch=BATCH, seed=SEED,
                deterministic=True, project=str(RUNS), name=name, exist_ok=True,
                verbose=False, plots=True, val=True)
    m = model.val(data=DATA, split="val", verbose=False)
    row = dict(model=name, params=params, gflops=gflops,
               mAP50=round(float(m.box.map50), 5), mAP=round(float(m.box.map), 5),
               precision=round(float(m.box.mp), 5), recall=round(float(m.box.mr), 5),
               minutes=round((time.time()-t0)/60, 1))
    del model; gc.collect(); torch.cuda.empty_cache()
    return row

def main():
    fields = ["model","params","gflops","mAP50","mAP","precision","recall","minutes"]
    write_header = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        for cfg in MODELS:
            try:
                print(f"\n=== TRAIN {cfg} ===", flush=True)
                row = run(cfg)
                print("RESULT", row, flush=True)
            except Exception as e:
                row = dict(model=cfg.replace('.yaml',''), params="FAIL", gflops="FAIL",
                           mAP50="FAIL", mAP="FAIL", precision="FAIL", recall="FAIL", minutes=0)
                print("FAILED", cfg, e, flush=True)
                traceback.print_exc()
            w.writerow(row); f.flush()

if __name__ == "__main__":
    main()
