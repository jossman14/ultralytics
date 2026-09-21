"""Autonomous stratified 5-fold sweep over the Judol datasets — parallel-worker edition.

- Queue: 4 native baselines + every experimental config, reordered LIGHT-FIRST so cheap/
  efficient models (the promising ones on small data) finish before heavy flagships.
- Protocol: 100 epochs, imgsz 640, batch 16, patience 30, seed 0, GPU 0, cache='ram'.
- Parallelism: launch N workers sharing one GPU; each atomically CLAIMS the next unclaimed
  job via an fcntl-locked ledger, so no two workers train the same (model,dataset,fold).
- Resumable + fault-tolerant: ledger records ok/error/running; restart skips ok, retries
  error, and reclaims stale 'running' (crashed worker) after RECLAIM_S.
- summary.json (mean/std over folds) rebuilt after every finished job.

Run one worker:   PYTHONPATH=/home/ftib/ultralytics python judol/run_sweep.py --worker 0 --workers-per-job 4
Launch N workers: see judol/launch_parallel.sh
"""
import argparse
import fcntl
import gc
import glob
import json
import time
import traceback
from pathlib import Path
from statistics import mean, pstdev

import torch

from ultralytics import YOLO

ROOT = Path("/home/ftib/ultralytics")
OUT = ROOT / "judol_sweep"
OUT.mkdir(exist_ok=True)
LEDGER = OUT / "ledger.json"
LOCK = OUT / "ledger.lock"
SUMMARY = OUT / "summary.json"

DATASETS = {
    "v2-1": ROOT / "judol/Judol-Detection-v2-1_strat5fold",
    "judi3k": ROOT / "judol/dataset_judi_online_yolo_strat5fold",
}
FOLDS = 5
EPOCHS = 100
PATIENCE = 30
RECLAIM_S = 3 * 3600  # a 'running' job older than this is assumed dead and reclaimable
MAX_ATTEMPTS = 2  # give up on a job after this many errors, so a broken model can't loop forever

# Baselines first (fast, define the target), then experimental models LIGHT-FIRST.
BASELINES = [
    ("yolov8n-base", "yolov8n.yaml"),
    ("yolo26n-base", str(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")),
    ("yolo11n-base", str(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")),
    ("yolo12n-base", str(ROOT / "ultralytics/cfg/models/12/yolo12.yaml")),
]

# Lightness ranking: lower = lighter/cheaper = run earlier. Ranked by dominant block motif.
LIGHT_RANK = [
    "se-baseline", "shuffle", "ghost", "-ema", "simam", "-eca", "-se", "coordatt",
    "mbconv", "star", "res2", "resnext", "sgsa", "ssga", "gam", "capffn", "ledh",
    "nonlocal", "swin", "convnext", "mobilevit", "hematology", "sota-fusion",
]

SKIP = {"yolov8-swin-timm", "yolov8s-swin-timm"}


def rank(stem):
    for i, tok in enumerate(LIGHT_RANK):
        if tok in stem:
            return i
    return len(LIGHT_RANK)  # unknown -> last


def build_queue():
    models = list(BASELINES)
    seen = {n for n, _ in BASELINES}
    exp = []
    for fam in ["v8", "26", "11", "12"]:
        for cfg in glob.glob(str(ROOT / f"ultralytics/cfg/models/{fam}/yolo*[a-z0-9]-*.yaml")):
            stem = Path(cfg).stem
            if any(t in stem for t in ("-seg", "-pose", "-obb", "-cls", "-world", "-p2", "-p6",
                                        "-sem", "rtdetr", "yoloe", "-ghost-p", "oiv7", "grayscale")):
                continue
            if stem in seen or stem in SKIP or stem in {"yolov8-ghost", "yolo26-p2", "yolo26-p6"}:
                continue
            seen.add(stem)
            exp.append((stem, cfg))
    exp.sort(key=lambda mc: (rank(mc[0]), mc[0]))  # light-first, then alpha
    return models + exp


ALL_JOBS = None  # list of (name, cfg, ds_key, fold) in priority order


def job_key(name, ds_key, fold):
    return f"{name}|{ds_key}|f{fold}"


class Locked:
    """Context manager: exclusive fcntl lock on LOCK; yields the loaded ledger dict, saves on exit."""

    def __init__(self, save=True):
        self.save = save

    def __enter__(self):
        self.fh = open(LOCK, "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        try:
            self.d = json.loads(LEDGER.read_text())
        except Exception:
            self.d = {}
        return self.d

    def __exit__(self, *a):
        if self.save:
            LEDGER.write_text(json.dumps(self.d, indent=1))
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def claim_next(worker_id):
    """Atomically claim the highest-priority not-done job. Returns (name,cfg,ds,fold) or None."""
    now = time.time()
    with Locked() as d:
        for name, cfg, ds_key, fold in ALL_JOBS:
            k = job_key(name, ds_key, fold)
            v = d.get(k)
            if v is None or (v.get("status") == "error" and v.get("attempts", 0) < MAX_ATTEMPTS) or (
                v.get("status") == "running" and now - v.get("t", 0) > RECLAIM_S
            ):
                d[k] = {"status": "running", "worker": worker_id, "t": now, "cfg": cfg,
                        "attempts": v.get("attempts", 0) if v else 0}
                return (name, cfg, ds_key, fold)
    return None


def record(name, ds_key, fold, result):
    with Locked() as d:
        k = job_key(name, ds_key, fold)
        if result.get("status") == "error":
            result["attempts"] = d.get(k, {}).get("attempts", 0) + 1
        d[k] = result


def aggregate():
    with Locked(save=False) as d:
        ledger = dict(d)
    agg = {}
    for key, v in ledger.items():
        if v.get("status") != "ok":
            continue
        name, ds_key, _ = key.split("|")
        agg.setdefault(name, {}).setdefault(ds_key, []).append(v)
    out = {}
    for name, per_ds in agg.items():
        out[name] = {}
        for ds_key, runs in per_ds.items():
            m50 = [r["mAP50"] for r in runs]
            m5095 = [r["mAP50-95"] for r in runs]
            out[name][ds_key] = {
                "folds_done": len(runs),
                "mAP50_mean": round(mean(m50), 4),
                "mAP50_std": round(pstdev(m50), 4) if len(m50) > 1 else 0.0,
                "mAP50-95_mean": round(mean(m5095), 4),
                "mAP50-95_std": round(pstdev(m5095), 4) if len(m5095) > 1 else 0.0,
            }
    SUMMARY.write_text(json.dumps(out, indent=1))


def train_job(name, cfg, ds_key, fold, worker_id, wpj):
    data = DATASETS[ds_key] / f"fold_{fold}" / "data.yaml"
    # v2-1 is tiny -> cache in RAM; judi3k is large and, with 2 parallel workers, two RAM
    # caches exhaust the 31GB box -> cache to disk for it. NOTE: in ultralytics cache=True
    # means RAM (not disk!); the disk option is the string "disk".
    cache = "ram" if ds_key == "v2-1" else "disk"
    t0 = time.time()
    try:
        model = YOLO(cfg, task="detect")
        r = model.train(
            data=str(data), epochs=EPOCHS, imgsz=640, batch=16, device=0, workers=wpj,
            patience=PATIENCE, seed=0, verbose=False, plots=False, cache=cache,
            project=str(OUT / "runs"), name=job_key(name, ds_key, fold).replace("|", "_"),
            exist_ok=True,
        )
        rd = dict(r.results_dict)
        res = {
            "status": "ok",
            "mAP50": round(rd.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(rd.get("metrics/mAP50-95(B)", 0), 4),
            "precision": round(rd.get("metrics/precision(B)", 0), 4),
            "recall": round(rd.get("metrics/recall(B)", 0), 4),
            "minutes": round((time.time() - t0) / 60, 1),
            "worker": worker_id,
        }
        print(f"[w{worker_id}] DONE {job_key(name,ds_key,fold)} mAP50={res['mAP50']} ({res['minutes']}m)", flush=True)
    except Exception as e:
        res = {"status": "error", "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-2000:], "worker": worker_id}
        print(f"[w{worker_id}] FAIL {job_key(name,ds_key,fold)} {type(e).__name__}: {e}", flush=True)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    record(name, ds_key, fold, res)
    aggregate()


def main():
    global ALL_JOBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers-per-job", type=int, default=4)  # dataloader workers per training job
    args = ap.parse_args()

    queue = build_queue()
    ALL_JOBS = [(n, c, ds, f) for n, c in queue for ds in DATASETS for f in range(FOLDS)]
    if not LEDGER.exists():
        LEDGER.write_text("{}")
    print(f"[w{args.worker}] queue: {len(queue)} models, {len(ALL_JOBS)} total jobs", flush=True)

    while True:
        job = claim_next(args.worker)
        if job is None:
            print(f"[w{args.worker}] no more jobs — WORKER DONE", flush=True)
            break
        train_job(*job, args.worker, args.workers_per_job)
    # last worker to finish will have written final summary
    print(f"[w{args.worker}] SWEEP COMPLETE (this worker)", flush=True)


if __name__ == "__main__":
    main()
