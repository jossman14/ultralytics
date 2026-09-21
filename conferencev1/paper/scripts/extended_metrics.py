"""Extended real metrics + figures for the expanded (>=10 page) manuscript.

No-GPU parts (ledger + results.csv + model-build for params):
  - per-fold mAP50/mAP50-95/precision/recall for the 4 focus models (both datasets)
  - broader benchmark: top nano models by mean mAP50-95 on v2-1, with params/GFLOPs
  - convergence: fold-averaged mAP50-95 per epoch for the 4 focus models

Figures (fonttype 42 -> embeds TrueType, fixes the Type 3 upload warning):
  - convergence.pdf, broader.pdf
Also re-emits the two existing matplotlib figures with fonttype 42.

Every number is read from the real sweep; nothing is invented.
"""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import yaml

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
LEDGER = REPO / "judol_sweep/ledger.json"
PAPER = REPO / "conferencev1/paper"
FIG = PAPER / "figures"
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

FOCUS = [
    ("yolov8n-base", "YOLOv8n (baseline)"),
    ("yolov8-se-baseline", "YOLOv8-SE (proposed)"),
    ("yolov8-res2net-spd-ssga", "YOLOv8-Res2Net-SPD-SSGA"),
    ("yolov8-shuffle-se", "YOLOv8-Shuffle-SE"),
]
ACC, GREY, INK = "#1a5fb4", "#5a5a5a", "#111111"


def fold_vals(led, stem, ds, metric):
    out = []
    for k, v in led.items():
        if k.startswith(f"{stem}|{ds}|f") and v.get("status") == "ok":
            out.append((int(k.split("|f")[1]), v.get(metric, 0)))
    return [x for _, x in sorted(out)]


def agg(vals):
    if not vals:
        return None
    return {"vals": [round(x, 4) for x in vals], "mean": round(st.mean(vals), 4),
            "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0}


def params_gflops(stem):
    """Build the model from the cfg recorded in args.yaml; return (Mparams, GFLOPs)."""
    args = RUNS / f"{stem}_v2-1_f0" / "args.yaml"
    cfg = yaml.safe_load(args.read_text())["model"]
    from ultralytics import YOLO
    m = YOLO(cfg, task="detect")
    n_p = sum(p.numel() for p in m.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops
        g = get_flops(m.model, imgsz=640)
    except Exception:
        g = None
    return round(n_p / 1e6, 3), (round(g, 2) if g else None)


def convergence(stem, metric="metrics/mAP50-95(B)"):
    """Fold-averaged metric per epoch (truncated to the shortest fold)."""
    series = []
    for f in range(5):
        p = RUNS / f"{stem}_v2-1_f{f}" / "results.csv"
        if not p.exists():
            continue
        col = []
        for row in csv.DictReader(p.open()):
            col.append(float(row[metric]))
        series.append(col)
    if not series:
        return []
    n = min(len(s) for s in series)
    return [round(st.mean(s[i] for s in series), 4) for i in range(n)]


def main():
    led = json.loads(LEDGER.read_text())
    out = {"focus": {}, "convergence": {}, "broader": []}

    for stem, name in FOCUS:
        e = {"name": name}
        for ds in ("v2-1", "judi3k"):
            e[ds] = {m: agg(fold_vals(led, stem, ds, m))
                     for m in ("mAP50", "mAP50-95", "precision", "recall")}
        out["focus"][stem] = e
        out["convergence"][stem] = convergence(stem)

    # broader nano benchmark
    bym = defaultdict(list)
    for k, v in led.items():
        if v.get("status") != "ok":
            continue
        nm, ds, _ = k.split("|")
        if ds == "v2-1":
            bym[nm].append(v.get("mAP50-95", 0))
    nano = [(n, st.mean(x)) for n, x in bym.items()
            if len(x) == 5 and not any(t in n for t in ["yolov8s", "yolo11s", "yolo12s"])]
    nano.sort(key=lambda z: -z[1])
    for nm, m in nano[:12]:
        try:
            p, g = params_gflops(nm)
        except Exception:
            p, g = None, None
        out["broader"].append({"name": nm, "mAP50-95": round(m, 4), "params_M": p, "gflops": g})
        print(f"  {m:.4f}  {nm:32s} {p} M  {g} G")

    (PAPER / "extended_metrics.json").write_text(json.dumps(out, indent=1))
    print("wrote extended_metrics.json")

    # ---- convergence figure ----
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    styles = {"yolov8n-base": (GREY, "-"), "yolov8-se-baseline": (ACC, "-"),
              "yolov8-res2net-spd-ssga": ("#c64600", "--"), "yolov8-shuffle-se": ("#2e7d32", ":")}
    for stem, name in FOCUS:
        y = out["convergence"][stem]
        if not y:
            continue
        c, ls = styles[stem]
        ax.plot(range(1, len(y) + 1), y, color=c, linestyle=ls, linewidth=1.6,
                label=name.replace(" (proposed)", "").replace(" (baseline)", ""))
    ax.set_xlabel("Epoch"); ax.set_ylabel("v2-1 mAP50-95 (fold mean)")
    ax.legend(fontsize=6.6, loc="lower right"); ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout(pad=0.2); fig.savefig(FIG / "convergence.pdf"); plt.close(fig)
    print("wrote convergence.pdf")

    # ---- broader benchmark scatter ----
    pts = [b for b in out["broader"] if b["params_M"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for b in pts:
        prop = b["name"] == "yolov8-se-baseline"
        base = b["name"] == "yolov8n-base"
        ax.scatter(b["params_M"], b["mAP50-95"],
                   s=95 if prop else 55, zorder=3,
                   color=ACC if prop else ("#c64600" if base else GREY),
                   marker="D" if prop else ("s" if base else "o"),
                   edgecolor="white", linewidth=0.7)
    for b in pts:
        if b["name"] in ("yolov8-se-baseline", "yolov8n-base"):
            lbl = "SE (proposed)" if b["name"] == "yolov8-se-baseline" else "baseline"
            ax.annotate(lbl, (b["params_M"], b["mAP50-95"]), (b["params_M"], b["mAP50-95"] + 0.0012),
                        fontsize=7.2, ha="center",
                        color=ACC if "SE" in lbl else "#c64600", fontweight="bold")
    ax.set_xlabel("Parameters (M)"); ax.set_ylabel("v2-1 mAP50-95 (fold mean)")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout(pad=0.2); fig.savefig(FIG / "broader.pdf"); plt.close(fig)
    print("wrote broader.pdf")


if __name__ == "__main__":
    main()
