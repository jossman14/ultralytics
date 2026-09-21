"""Regenerate the two confusion matrices with large fonts.
plots=True is required so ultralytics actually populates the matrix."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from ultralytics import YOLO

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
FIG = REPO / "conferencev1/paper/figures"
TMP = FIG / "_cmtmp"


def plot_cm(matrix, names, out, big=17):
    m = matrix.astype(float)
    m = m / (m.sum(0, keepdims=True) + 1e-9)
    labels = names + ["background"]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.15), max(5, n * 1.0)))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if m[i, j] >= 0.005:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="white" if m[i, j] > 0.5 else "black", fontsize=big)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=big)
    ax.set_yticklabels(labels, fontsize=big)
    ax.set_xlabel("True", fontsize=big + 2); ax.set_ylabel("Predicted", fontsize=big + 2)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=big - 3)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out, "sum", int(matrix.sum()))


def run(weights, data, names, out):
    m = YOLO(str(weights))
    r = m.val(data=str(data), split="val", device="cpu", plots=True,
              project=str(TMP), name="v", exist_ok=True, verbose=False)
    plot_cm(r.confusion_matrix.matrix, names, out)


run(RUNS / "yolov8-se-baseline_v2-1_f0/weights/best.pt",
    REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/data.yaml",
    ["BK8", "Gate-of-olympus", "Princess", "Starlight-Princess", "Zeus"],
    FIG / "confusion_matrix.png")
run(RUNS / "yolov8-se-baseline_judi3k_f0/weights/best.pt",
    REPO / "judol/dataset_judi_online_yolo_strat5fold/fold_0/data.yaml",
    ["gambling", "nongambling"],
    FIG / "confusion_matrix_judi3k.png")
