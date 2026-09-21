"""5-fold cross-validation for the IJAAS revision: YOLOv8s/v10s/11s/12s on both judol datasets.

One process per worker slot; slots are pinned to a GPU by the launcher.
Hyperparameters copied verbatim from the original YOLOv12s runs (args.yaml).
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/home/ftib/ultralytics")
OUT = ROOT / "revisi/runs"

DATASETS = {
    "binary": ROOT / "judol/dataset_judi_online_yolo_5fold",
    "multiclass": ROOT / "judol/Judol-Detection-v2-1_5fold",
    "binary_videofold": ROOT / "judol/dataset_judi_online_yolo_videofold",
}
MODELS = ["yolov8s", "yolov10s", "yolo11s", "yolo12s"]

TRAIN_ARGS = dict(
    epochs=100, patience=30, batch=16, imgsz=480, optimizer="adamW",
    lr0=0.01, lrf=0.01, cos_lr=True, close_mosaic=10, amp=True,
    seed=0, deterministic=True, pretrained=True, plots=True, val=True, verbose=False,
)


def run_one(model_name, dataset, fold, device, workers):
    tag = f"{model_name}__{dataset}__fold{fold}"
    save_dir = OUT / dataset / model_name
    done = save_dir / f"fold_{fold}" / "metrics.json"
    if done.exists():
        print(f"[skip] {tag}")
        return
    data = DATASETS[dataset] / f"fold_{fold}" / "data.yaml"
    model = YOLO(f"{model_name}.yaml")
    t0 = time.time()
    model.train(data=str(data), project=str(save_dir), name=f"fold_{fold}", exist_ok=True,
                device=device, workers=workers, **TRAIN_ARGS)
    minutes = (time.time() - t0) / 60

    m = model.val(data=str(data), device=device, workers=workers, imgsz=480, batch=16, verbose=False,
                  project=str(save_dir), name=f"fold_{fold}_val", exist_ok=True, plots=False)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.class_result(i)[:4]
        per_class[names[int(c)]] = dict(
            precision=float(p), recall=float(r),
            f1=float(2 * p * r / (p + r)) if (p + r) else 0.0,
            map50=float(ap50), map=float(ap),
        )
    p, r = float(m.box.mp), float(m.box.mr)
    n_params = sum(x.numel() for x in model.model.parameters())
    out = dict(
        model=model_name, dataset=dataset, fold=fold, params=n_params,
        precision=p, recall=r, f1=(2 * p * r / (p + r)) if (p + r) else 0.0,
        map50=float(m.box.map50), map=float(m.box.map),
        speed_ms=dict(m.speed), train_minutes=minutes, per_class=per_class,
    )
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(out, indent=2))
    print(f"[done] {tag} mAP50={out['map50']:.4f} mAP={out['map']:.4f} ({minutes:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--nslots", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--models", default="")
    a = ap.parse_args()

    only = a.datasets.split(",") if a.datasets else list(DATASETS)
    only_m = a.models.split(",") if a.models else MODELS
    jobs = [(m, d, f) for d in only for m in only_m for f in range(5)]
    # YOLOv12s is the primary model and the random-vs-video-disjoint comparison is
    # the paper's headline, so run every yolo12s job first. Baselines follow.
    # Within a tier, sort by fold so the cheap multiclass folds interleave with the
    # expensive binary ones and the slots finish together.
    jobs.sort(key=lambda j: (j[0] != "yolo12s", j[2], j[0], j[1]))
    for i, (m, d, f) in enumerate(jobs):
        if i % a.nslots == a.slot:
            run_one(m, d, f, a.device, a.workers)
