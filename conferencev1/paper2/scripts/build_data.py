"""Assemble all real numbers for the applied gambling-detection paper (Topic 3).

- Design-space statistics over the 786-run sweep (complexity trap, winning recipe).
- Comparison table: proposed vs baseline vs standard family detectors + top variants,
  with params/GFLOPs computed from each model's recorded config.
- Per-fold / precision-recall for the proposed model on both datasets.
Writes data.json. Real values only.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import yaml

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
LEDGER = REPO / "judol_sweep/ledger.json"
OUT = Path(__file__).resolve().parent.parent
PROPOSED = "yolov8-spd-shuffle-simam"
BASELINE = "yolov8n-base"

led = json.loads(LEDGER.read_text())
bym = defaultdict(lambda: defaultdict(list))
for k, v in led.items():
    if v.get("status") == "ok":
        n, ds, f = k.split("|")
        bym[n][ds].append(v)


def agg(n, ds, key):
    r = bym[n].get(ds, [])
    if len(r) != 5:
        return None
    vals = [x.get(key, 0) for x in r]
    return {"mean": round(st.mean(vals), 4), "std": round(st.pstdev(vals), 4),
            "vals": [round(x, 4) for x in vals]}


def pg(n):
    a = RUNS / f"{n}_v2-1_f0" / "args.yaml"
    if not a.exists():
        return None, None
    cfg = yaml.safe_load(a.read_text())["model"]
    from ultralytics import YOLO
    m = YOLO(cfg, task="detect")
    np_ = sum(p.numel() for p in m.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops
        g = get_flops(m.model, imgsz=640)
    except Exception:
        g = None
    return round(np_ / 1e6, 3), (round(g, 2) if g else None)


# ---- design-space statistics ----
v21 = {n: agg(n, "v2-1", "mAP50-95") for n in bym}
v21 = {n: v["mean"] for n, v in v21.items() if v}
ju = {n: agg(n, "judi3k", "mAP50-95") for n in bym}
ju = {n: v["mean"] for n, v in ju.items() if v}
both = set(v21) & set(ju)
base = v21[BASELINE]
above = [n for n in v21 if v21[n] > base]
common = sorted(both)


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
    da = [ra[i] for i in range(len(a))]; db = [rb[i] for i in range(len(b))]
    n = len(a); dsq = sum((da[i] - db[i]) ** 2 for i in range(n))
    return 1 - 6 * dsq / (n * (n * n - 1))


design = {
    "n_runs_ok": sum(1 for v in led.values() if v.get("status") == "ok"),
    "n_models_both": len(both),
    "n_models_v21": len(v21),
    "beat_baseline": len(above),
    "beat_baseline_pct": round(100 * len(above) / len(v21), 1),
    "baseline_v21": base,
    "spearman_v21_judi3k": round(spearman([v21[n] for n in common], [ju[n] for n in common]), 3),
    "v21_spread": [round(min(v21.values()), 4), round(max(v21.values()), 4)],
    "collapse_judi3k": sorted([(n, round(ju[n], 4)) for n in ju if ju[n] < 0.9], key=lambda x: x[1]),
}

# scatter payload for the design-space figure
scatter = []
for n in common:
    scatter.append({"name": n, "v21": round(v21[n], 4), "judi3k": round(ju[n], 4),
                    "beats": v21[n] > base,
                    "proposed": n == PROPOSED, "baseline": n == BASELINE})

# ---- comparison table (proposed + baseline + standard families + top recipe winners) ----
COMPARE = [BASELINE, "yolo11n-base", "yolo12n-base", "yolo26n-base",
           "yolov8-se-baseline", "yolov8-spd-ghostv2-sgsa", "yolov8-shuffle-se",
           "yolov8-res2net-spd-ssga", PROPOSED]
comparison = []
for n in COMPARE:
    if n not in bym:
        continue
    p, g = pg(n)
    comparison.append({
        "name": n, "params_M": p, "gflops": g,
        "v21_map50": agg(n, "v2-1", "mAP50"), "v21_map5095": agg(n, "v2-1", "mAP50-95"),
        "ju_map5095": agg(n, "judi3k", "mAP50-95"),
        "v21_P": agg(n, "v2-1", "precision"), "v21_R": agg(n, "v2-1", "recall"),
    })
    print(f"  {n:28s} {p} M {g} G  v2-1 mAP50-95={comparison[-1]['v21_map5095']['mean'] if comparison[-1]['v21_map5095'] else 'NA'}")

# proposed per-fold both datasets
proposed_full = {ds: {m: agg(PROPOSED, ds, m) for m in ["mAP50", "mAP50-95", "precision", "recall"]}
                 for ds in ["v2-1", "judi3k"]}

(OUT / "data.json").write_text(json.dumps(
    {"design": design, "scatter": scatter, "comparison": comparison, "proposed": proposed_full}, indent=1))
print("\nwrote data.json")
print("design:", json.dumps(design, indent=1))
