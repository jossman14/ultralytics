"""Per-class AP for the 4 focus models on v2-1 (fold 0), computed on CPU so the
running GPU sweep is undisturbed. Writes per-class numbers into extended_metrics.json
and a grouped-bar figure figures/perclass.pdf. Real validation, no invented values.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
PAPER = REPO / "conferencev1/paper"
DATA = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/data.yaml"
CLASSES = ["BK8", "Gate-of-olympus", "Princess", "Starlight-Princess", "Zeus"]
FOCUS = [
    ("yolov8n-base", "Baseline"),
    ("yolov8-se-baseline", "SE (proposed)"),
    ("yolov8-res2net-spd-ssga", "Res2Net-SPD-SSGA"),
    ("yolov8-shuffle-se", "Shuffle-SE"),
]
COLORS = {"Baseline": "#c64600", "SE (proposed)": "#1a5fb4",
          "Res2Net-SPD-SSGA": "#5a5a5a", "Shuffle-SE": "#2e7d32"}
plt.rcParams.update({"font.size": 9, "font.family": "serif"})


def per_class(stem):
    m = YOLO(str(RUNS / f"{stem}_v2-1_f0" / "weights" / "best.pt"))
    r = m.val(data=str(DATA), split="val", device="cpu", plots=False, verbose=False)
    ap50 = {}  # class name -> mAP50
    for i, ci in enumerate(r.box.ap_class_index):
        ap50[CLASSES[ci]] = round(float(r.box.ap50[i]), 4)
    return {c: ap50.get(c, 0.0) for c in CLASSES}


def main():
    ext = json.loads((PAPER / "extended_metrics.json").read_text())
    ext["perclass"] = {}
    for stem, name in FOCUS:
        ext["perclass"][stem] = {"name": name, "ap50": per_class(stem)}
        print(name, ext["perclass"][stem]["ap50"])
    (PAPER / "extended_metrics.json").write_text(json.dumps(ext, indent=1))

    x = np.arange(len(CLASSES))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for j, (stem, name) in enumerate(FOCUS):
        vals = [ext["perclass"][stem]["ap50"][c] for c in CLASSES]
        ax.bar(x + (j - 1.5) * w, vals, w, label=name, color=COLORS[name],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_ylabel("mAP50 (v2-1, fold 0)"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(PAPER / "figures" / "perclass.pdf")
    print("wrote perclass.pdf")


if __name__ == "__main__":
    main()
