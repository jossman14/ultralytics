"""Regenerate convergence / broader / perclass figures with larger fonts and the
new dataset name, reading cached values from extended_metrics.json (no recompute)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

PAPER = Path("/home/ftib/ultralytics/conferencev1/paper")
FIG = PAPER / "figures"
ext = json.loads((PAPER / "extended_metrics.json").read_text())
plt.rcParams.update({"font.size": 12, "font.family": "serif"})  # bigger base font
ACC, GREY = "#1a5fb4", "#5a5a5a"
FOCUS = [("yolov8n-base", "YOLOv8n"), ("yolov8-se-baseline", "YOLOv8-SE"),
         ("yolov8-res2net-spd-ssga", "YOLOv8-Res2Net-SPD-SSGA"), ("yolov8-shuffle-se", "YOLOv8-Shuffle-SE")]
YLAB = "Judol Detection v2 mAP50-95"

# convergence
sty = {"yolov8n-base": (GREY, "-"), "yolov8-se-baseline": (ACC, "-"),
       "yolov8-res2net-spd-ssga": ("#c64600", "--"), "yolov8-shuffle-se": ("#2e7d32", ":")}
fig, ax = plt.subplots(figsize=(4.0, 3.1))
for stem, name in FOCUS:
    y = ext["convergence"][stem]
    c, ls = sty[stem]
    ax.plot(range(1, len(y) + 1), y, color=c, linestyle=ls, linewidth=2.0, label=name)
ax.set_xlabel("Epoch"); ax.set_ylabel(YLAB)
ax.legend(fontsize=9, loc="lower right"); ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout(pad=0.2); fig.savefig(FIG / "convergence.pdf"); plt.close(fig)

# broader (only models not beating SE, same rule as table)
se = ext["focus"]["yolov8-se-baseline"]["v2-1"]["mAP50-95"]["mean"]
pts = [b for b in ext["broader"] if b["params_M"] and (b["mAP50-95"] <= se or b["name"] == "yolov8-se-baseline")]
fig, ax = plt.subplots(figsize=(4.0, 3.2))
for b in pts:
    prop = b["name"] == "yolov8-se-baseline"; base = b["name"] == "yolov8n-base"
    ax.scatter(b["params_M"], b["mAP50-95"], s=140 if prop else 70, zorder=3,
               color=ACC if prop else ("#c64600" if base else GREY),
               marker="D" if prop else ("s" if base else "o"), edgecolor="white", linewidth=0.8)
for b in pts:
    if b["name"] in ("yolov8-se-baseline", "yolov8n-base"):
        lbl = "SE (ours)" if b["name"] == "yolov8-se-baseline" else "baseline"
        ax.annotate(lbl, (b["params_M"], b["mAP50-95"]), (b["params_M"], b["mAP50-95"] + 0.0011),
                    fontsize=11, ha="center", color=ACC if "SE" in lbl else "#c64600", fontweight="bold")
ax.set_xlabel("Parameters (M)"); ax.set_ylabel(YLAB); ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout(pad=0.2); fig.savefig(FIG / "broader.pdf"); plt.close(fig)

# perclass
CLASSES = ["BK8", "Gate-of-olympus", "Princess", "Starlight-Princess", "Zeus"]
COL = {"yolov8n-base": "#c64600", "yolov8-se-baseline": "#1a5fb4",
       "yolov8-res2net-spd-ssga": "#5a5a5a", "yolov8-shuffle-se": "#2e7d32"}
NAME = {"yolov8n-base": "Baseline", "yolov8-se-baseline": "SE (ours)",
        "yolov8-res2net-spd-ssga": "Res2Net-SPD", "yolov8-shuffle-se": "Shuffle-SE"}
x = np.arange(len(CLASSES)); w = 0.2
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for j, (stem, _) in enumerate(FOCUS):
    vals = [ext["perclass"][stem]["ap50"][c] for c in CLASSES]
    ax.bar(x + (j - 1.5) * w, vals, w, label=NAME[stem], color=COL[stem], edgecolor="white", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel("mAP50 (fold 0)", fontsize=12); ax.set_ylim(0, 1.0)
ax.legend(fontsize=10, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
fig.tight_layout(pad=0.2); fig.savefig(FIG / "perclass.pdf"); plt.close(fig)
print("refreshed convergence, broader, perclass with larger fonts")
