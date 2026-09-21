"""
Train the ablation ladder + proposed model AFTER the baseline sweep finishes.
Cumulative ablation:
  baseline (yolov8n, from sweep) -> +SSGA -> +StarGhost -> full proposed (SGSA-YOLO).
Waits for the single GPU to be free (baseline sweep done) then trains, logging to results.csv.
ponytail: poll results.csv row count instead of tracking a PID across sessions.
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
N_BASELINES = 7  # yolov5n,v6n,v8n,v9t,v10n,11n,12n

# ablation ladder: label -> cfg  (baseline yolov8n already trained in the sweep)
LADDER = [
    ("ssga", "yolov8-ssga.yaml"),            # baseline + SSGA
    ("starghost", "yolov8-starghost.yaml"),  # lightweight backbone/neck
    ("sgsa_proposed", "yolov8-sgsa.yaml"),   # full proposed
]

def n_done():
    if not CSV.exists(): return 0
    with open(CSV) as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header

def wait_for_gpu_free():
    """Block until the baseline sweep has written all its rows."""
    while n_done() < N_BASELINES:
        print(f"[wait] baseline rows {n_done()}/{N_BASELINES}, sleeping 120s", flush=True)
        time.sleep(120)
    print("[wait] baselines done, starting proposed/ablation", flush=True)

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
    wait_for_gpu_free()
    fields = ["model","params","gflops","mAP50","mAP","precision","recall","minutes"]
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for label, cfg in LADDER:
            try:
                print(f"\n=== TRAIN {label} ({cfg}) ===", flush=True)
                row = run(label, cfg)
                print("RESULT", row, flush=True)
            except Exception as e:
                row = dict(model=label, params="FAIL", gflops="FAIL", mAP50="FAIL",
                           mAP="FAIL", precision="FAIL", recall="FAIL", minutes=0)
                print("FAILED", label, e, flush=True); traceback.print_exc()
            w.writerow(row); f.flush()

if __name__ == "__main__":
    main()
