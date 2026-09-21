"""
SINGLE-PROCESS training entrypoint (baselines + ablation + proposed, sequential).
Use this to resume after the GPU is freed. One CUDA context only -> no init contention.
Resumable: skips any model already present in results.csv.

  cd /home/ftib/ultralytics
  PYTHONPATH=/home/ftib/ultralytics nohup python3 paper_judol/run_all.py > paper_judol/run_all.log 2>&1 &
"""
import csv, gc, time, traceback, sys
sys.path.insert(0, "/home/ftib/ultralytics")
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

ROOT = Path("/home/ftib/ultralytics/paper_judol")
DATA = str(ROOT / "judol_v29.yaml")
RUNS = ROOT / "experiments"
CSV = ROOT / "results.csv"
EPOCHS, IMG, BATCH, SEED = 150, 640, 16, 0

# label -> config. Baselines (native YOLO family) + ablation ladder + proposed.
JOBS = [
    ("yolov5n", "yolov5n.yaml"), ("yolov6n", "yolov6n.yaml"), ("yolov7t", "yolov7-tiny.yaml"),
    ("yolov8n", "yolov8n.yaml"),
    ("yolov9t", "yolov9t.yaml"), ("yolov10n", "yolov10n.yaml"), ("yolo11n", "yolo11n.yaml"),
    ("yolo12n", "yolo12n.yaml"),
    ("ssga", "yolov8-ssga.yaml"),            # ablation: +SSGA
    ("starghost", "yolov8-starghost.yaml"),  # ablation: +StarGhost
    ("sgsa_proposed", "yolov8-sgsa.yaml"),   # proposed: full
]

def done_models():
    if not CSV.exists(): return set()
    with open(CSV) as f:
        return {r["model"] for r in csv.DictReader(f)}

def run(label, cfg):
    t0 = time.time()
    model = YOLO(cfg)
    params, gflops = get_num_params(model.model), round(get_flops(model.model, imgsz=IMG), 4)
    model.train(data=DATA, epochs=EPOCHS, imgsz=IMG, batch=BATCH, seed=SEED,
                deterministic=True, project=str(RUNS), name=label, exist_ok=True,
                verbose=False, plots=True, val=True)
    m = model.val(data=DATA, split="val", verbose=False)
    row = dict(model=label, params=params, gflops=gflops,
               mAP50=round(float(m.box.map50), 5), mAP=round(float(m.box.map), 5),
               precision=round(float(m.box.mp), 5), recall=round(float(m.box.mr), 5),
               minutes=round((time.time()-t0)/60, 1))
    del model; gc.collect(); torch.cuda.empty_cache()
    return row

def main():
    fields = ["model","params","gflops","mAP50","mAP","precision","recall","minutes"]
    already = done_models()
    write_header = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        for label, cfg in JOBS:
            if label in already:
                print(f"[skip] {label} already in results.csv", flush=True); continue
            try:
                print(f"\n=== TRAIN {label} ({cfg}) ===", flush=True)
                row = run(label, cfg); print("RESULT", row, flush=True)
            except Exception as e:
                row = dict(model=label, params="FAIL", gflops="FAIL", mAP50="FAIL",
                           mAP="FAIL", precision="FAIL", recall="FAIL", minutes=0)
                print("FAILED", label, e, flush=True); traceback.print_exc()
            w.writerow(row); f.flush()
    print("\nALL DONE", flush=True)

if __name__ == "__main__":
    main()
