"""Deterministic computation of paper metrics from the real 5-fold sweep ledger.

Reads judol_sweep/ledger.json (5-fold cross-validation results) and produces:
  - main results table (mean +/- std over folds) for the 4-model shortlist,
  - relative-change table vs baseline,
  - metrics.json consumed by generate_figures.py and the LaTeX tables.

No value is invented: every number derives from ledger entries with status "ok".
Params/GFLOPs are measured once with ultralytics and hard-coded here (see MEASURE note).
"""
import json
import statistics as st
from pathlib import Path

LEDGER = Path("/home/ftib/ultralytics/judol_sweep/ledger.json")
OUT = Path(__file__).resolve().parent.parent / "metrics.json"

# 4-model shortlist (ledger key stem -> display name); YOLOv8-SE is the proposed variant.
MODELS = [
    ("yolov8n-base", "YOLOv8n (baseline)"),
    ("yolov8-se-baseline", "YOLOv8-SE (proposed)"),
    ("yolov8-res2net-spd-ssga", "YOLOv8-Res2Net-SPD-SSGA"),
    ("yolov8-shuffle-se", "YOLOv8-Shuffle-SE"),
]
DATASETS = ["v2-1", "judi3k"]

# Params / GFLOPs measured with ultralytics get_flops at imgsz=640, scale n (see paper scripts).
COMPLEXITY = {
    "yolov8n-base": (3_157_200, 8.86),
    "yolov8-se-baseline": (3_168_428, 8.87),
    "yolov8-res2net-spd-ssga": (4_448_352, 10.98),
    "yolov8-shuffle-se": (2_699_204, 7.50),
}


def fold_values(ledger, stem, ds, metric):
    """Return per-fold metric values (ordered) for a model/dataset."""
    out = []
    for k, v in ledger.items():
        if k.startswith(f"{stem}|{ds}|f") and v.get("status") == "ok":
            out.append((int(k.split("|f")[1]), v[metric]))
    return [val for _, val in sorted(out)]


def agg(vals):
    if not vals:
        return None
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(m, 4), "std": round(s, 4), "n": len(vals)}


def main():
    ledger = json.loads(LEDGER.read_text())
    results = {}
    for stem, name in MODELS:
        entry = {"name": name, "params": COMPLEXITY[stem][0], "gflops": COMPLEXITY[stem][1]}
        for ds in DATASETS:
            entry[ds] = {
                "mAP50": agg(fold_values(ledger, stem, ds, "mAP50")),
                "mAP50-95": agg(fold_values(ledger, stem, ds, "mAP50-95")),
            }
        results[stem] = entry

    base = results["yolov8n-base"]
    rel = {}
    for stem, name in MODELS:
        if stem == "yolov8n-base":
            continue
        r = results[stem]
        d = {
            "name": name,
            "d_v21_map50": round(r["v2-1"]["mAP50"]["mean"] - base["v2-1"]["mAP50"]["mean"], 4),
            "d_v21_map5095": round(r["v2-1"]["mAP50-95"]["mean"] - base["v2-1"]["mAP50-95"]["mean"], 4),
            "d_params_M": round((r["params"] - base["params"]) / 1e6, 3),
            "params_pct": round(100 * (r["params"] - base["params"]) / base["params"], 2),
            "d_gflops": round(r["gflops"] - base["gflops"], 2),
            "gflops_pct": round(100 * (r["gflops"] - base["gflops"]) / base["gflops"], 2),
        }
        rel[stem] = d

    OUT.write_text(json.dumps({"results": results, "relative": rel}, indent=2))

    # human-readable dump
    print("=== MAIN RESULTS (mean +/- std, 5-fold) ===")
    for stem, name in MODELS:
        r = results[stem]
        v = r["v2-1"]; j = r["judi3k"]
        print(f"{name:26s} | v2-1 mAP50 {v['mAP50']['mean']:.4f}+/-{v['mAP50']['std']:.3f} "
              f"mAP50-95 {v['mAP50-95']['mean']:.4f} | judi3k mAP50 {j['mAP50']['mean']:.4f} "
              f"mAP50-95 {j['mAP50-95']['mean']:.4f} | {r['params']:,} p {r['gflops']} G")
    print("\n=== RELATIVE TO BASELINE ===")
    for stem, d in rel.items():
        print(f"{d['name']:26s} | dmAP50 {d['d_v21_map50']:+.4f} dmAP50-95 {d['d_v21_map5095']:+.4f} "
              f"| dParams {d['d_params_M']:+.3f}M ({d['params_pct']:+.2f}%) "
              f"dGFLOPs {d['d_gflops']:+.2f} ({d['gflops_pct']:+.2f}%)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
