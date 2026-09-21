"""Held-out test eval: existing v2-9-trained best.pt checkpoints -> v2-1 test split (53 imgs).

No training. Loads each best.pt and runs val on the held-out test split.
Writes paper_judol/results_heldout.csv (resumable by model name).
"""
import csv, sys
sys.path.insert(0, "/home/ftib/ultralytics")
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/ftib/ultralytics/paper_judol")
EXP = ROOT / "experiments"
DATA = str(ROOT / "judol_v21_test.yaml")
CSV = ROOT / "results_heldout.csv"

MODELS = [
    ("C2_neckinception", EXP / "C2_neckinception/weights/best.pt"),  # new proposed
    ("sgsa_proposed", EXP / "sgsa_proposed/weights/best.pt"),        # old proposed
    ("yolov8n", EXP / "yolov8n/weights/best.pt"),                    # baseline
    ("yolov7t", EXP / "yolov7t/weights/best.pt"),                    # ceiling
]


def done():
    if not CSV.exists():
        return set()
    with open(CSV) as f:
        return {r["model"] for r in csv.DictReader(f)}


def main():
    fields = ["model", "mAP50", "mAP", "precision", "recall"]
    already = done()
    write_header = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for label, ckpt in MODELS:
            if label in already:
                print(f"[skip] {label}", flush=True); continue
            print(f"\n=== EVAL {label} on v2-1 test ===", flush=True)
            m = YOLO(str(ckpt))
            r = m.val(data=DATA, split="test", verbose=False)
            row = dict(model=label, mAP50=round(float(r.box.map50), 5), mAP=round(float(r.box.map), 5),
                       precision=round(float(r.box.mp), 5), recall=round(float(r.box.mr), 5))
            print("RESULT", row, flush=True)
            w.writerow(row); f.flush()
    print("\nHELDOUT DONE", flush=True)


if __name__ == "__main__":
    main()
