"""Autonomous resumable driver: yolov7 baseline + 3-scenario ablation study.

Single CUDA context, sequential. Resumable via results_experiments.csv (skips any
label already recorded). Same train/val protocol as run_all.py so numbers are comparable.

  cd /home/ftib/ultralytics
  PYTHONPATH=/home/ftib/ultralytics nohup python3 paper_judol/run_experiments.py \
      > paper_judol/run_experiments.log 2>&1 &

Scenario A (head attention): A0-noattn == starghost (already trained, reused in paper table).
Scenario B (backbone block).  Scenario C (neck fusion).  Proposed SGSA == sgsa_proposed (results.csv).
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
CSV = ROOT / "results_experiments.csv"
EPOCHS, IMG, BATCH, SEED = 150, 640, 16, 0

# label -> config. yolov7 baseline + scenario ablation variants.
JOBS = [
    ("yolov7t", "yolov7-tiny.yaml"),                              # missing baseline
    # Scenario A: head attention (A0-noattn == starghost, reused)
    ("A1_cbam", "yolov8-sgsaA1-cbam.yaml"),
    ("A2_eca", "yolov8-sgsaA2-eca.yaml"),
    ("A3_gam", "yolov8-sgsaA3-gam.yaml"),
    # Scenario B: backbone block
    ("B1_ghost", "yolov8-sgsaB1-ghost.yaml"),
    ("B2_inception", "yolov8-sgsaB2-inception.yaml"),
    ("B3_dense", "yolov8-sgsaB3-dense.yaml"),
    ("B4_c2f", "yolov8-sgsaB4-c2f.yaml"),
    # Scenario C: neck fusion
    ("C1_neckdense", "yolov8-sgsaC1-neckdense.yaml"),
    ("C2_neckinception", "yolov8-sgsaC2-neckinception.yaml"),
    ("C3_neckstar", "yolov8-sgsaC3-neckstar.yaml"),
]


def done_models():
    if not CSV.exists():
        return set()
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
               minutes=round((time.time() - t0) / 60, 1))
    del model; gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    fields = ["model", "params", "gflops", "mAP50", "mAP", "precision", "recall", "minutes"]
    already = done_models()
    write_header = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for label, cfg in JOBS:
            if label in already:
                print(f"[skip] {label} already in results_experiments.csv", flush=True); continue
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
