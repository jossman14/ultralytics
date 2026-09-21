"""5-fold cross-validation on judol/Judol-Detection-v2-1_5fold (methodologically clean:
train+val from the same v2-1 pool per fold, no cross-fold leakage, consistent preprocessing).

4 models x 5 folds = 20 runs. Resumable via results_cv.csv (key = model_fold).
Same train protocol as the main study (150 epochs, imgsz 640, batch 16, seed 0).

  cd /home/ftib/ultralytics
  PYTHONPATH=/home/ftib/ultralytics nohup python3 paper_judol/run_cv.py \
      > paper_judol/run_cv.log 2>&1 &
"""
import csv, gc, time, traceback, sys
sys.path.insert(0, "/home/ftib/ultralytics")
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

ROOT = Path("/home/ftib/ultralytics/paper_judol")
FOLDROOT = Path("/home/ftib/ultralytics/judol/Judol-Detection-v2-1_5fold")
RUNS = ROOT / "experiments_cv"
CSV = ROOT / "results_cv.csv"
EPOCHS, IMG, BATCH, SEED, NFOLD = 150, 640, 16, 0, 5

MODELS = [
    ("proposed_new", "yolov8-sgsaC2-neckinception.yaml"),  # SSGA head + Inception neck
    ("proposed_old", "yolov8-sgsa.yaml"),                  # original SGSA-YOLO
    ("yolov8n", "yolov8n.yaml"),                           # baseline
    ("yolov7t", "yolov7-tiny.yaml"),                       # ceiling
]


def done():
    if not CSV.exists():
        return set()
    with open(CSV) as f:
        return {r["run"] for r in csv.DictReader(f)}


def run(model_label, cfg, fold):
    data = str(FOLDROOT / f"fold_{fold}" / "data_abs.yaml")
    name = f"{model_label}_fold{fold}"
    t0 = time.time()
    model = YOLO(cfg)
    params, gflops = get_num_params(model.model), round(get_flops(model.model, imgsz=IMG), 4)
    model.train(data=data, epochs=EPOCHS, imgsz=IMG, batch=BATCH, seed=SEED,
                deterministic=True, project=str(RUNS), name=name, exist_ok=True,
                verbose=False, plots=False, val=True)
    m = model.val(data=data, split="val", verbose=False)
    row = dict(run=name, model=model_label, fold=fold, params=params, gflops=gflops,
               mAP50=round(float(m.box.map50), 5), mAP=round(float(m.box.map), 5),
               precision=round(float(m.box.mp), 5), recall=round(float(m.box.mr), 5),
               minutes=round((time.time() - t0) / 60, 1))
    del model; gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    fields = ["run", "model", "fold", "params", "gflops", "mAP50", "mAP", "precision", "recall", "minutes"]
    already = done()
    write_header = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for label, cfg in MODELS:
            for fold in range(NFOLD):
                name = f"{label}_fold{fold}"
                if name in already:
                    print(f"[skip] {name}", flush=True); continue
                try:
                    print(f"\n=== TRAIN {name} ({cfg}) ===", flush=True)
                    row = run(label, cfg, fold); print("RESULT", row, flush=True)
                except Exception as e:
                    row = dict(run=name, model=label, fold=fold, params="FAIL", gflops="FAIL",
                               mAP50="FAIL", mAP="FAIL", precision="FAIL", recall="FAIL", minutes=0)
                    print("FAILED", name, e, flush=True); traceback.print_exc()
                w.writerow(row); f.flush()
    print("\nCV ALL DONE", flush=True)


if __name__ == "__main__":
    main()
