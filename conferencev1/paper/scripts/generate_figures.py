"""Generate the three paper figures as vector PDFs.

Fig 1: comparative evaluation workflow (flowchart).
Fig 2: conceptual Squeeze-and-Excitation operation (flowchart).
Fig 3: accuracy-complexity trade-off scatter, from the real metrics.json.

All figures use only measured values; no fabricated points or error bars.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3 (fixes upload font warning)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "svg.fonttype": "none"})

INK = "#111111"
ACC = "#1a5fb4"      # proposed / accent
GREY = "#5a5a5a"
BOX = "#f2f4f8"
BOXP = "#dbe7ff"     # proposed box fill


def flow(boxes, fname, w=3.3, h=None, accent_idx=None):
    """Vertical flowchart: boxes is a list of text strings, top to bottom."""
    n = len(boxes)
    h = h or 0.62 * n + 0.2
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, n)
    ax.axis("off")
    yc = [n - 0.5 - i for i in range(n)]
    for i, (txt, y) in enumerate(zip(boxes, yc)):
        fill = BOXP if (accent_idx is not None and i in accent_idx) else BOX
        edge = ACC if (accent_idx is not None and i in accent_idx) else GREY
        b = FancyBboxPatch((0.08, y - 0.30), 0.84, 0.60,
                           boxstyle="round,pad=0.02,rounding_size=0.04",
                           linewidth=1.1, edgecolor=edge, facecolor=fill)
        ax.add_patch(b)
        ax.text(0.5, y, txt, ha="center", va="center", color=INK, fontsize=8.4, wrap=True)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.31), (0.5, yc[i + 1] + 0.31),
                         arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color=GREY))
    fig.tight_layout(pad=0.15)
    fig.savefig(FIG / f"{fname}.pdf"); fig.savefig(FIG / f"{fname}.svg")
    plt.close(fig)


def fig1():
    flow([
        "Two datasets: v2-1 (443 img, 5 cls)\nand judi3k (3000 img, 2 cls)",
        "Data preparation\n(5-fold cross-validation)",
        "Four YOLOv8 variants:\nbaseline, SE, Res2Net-SPD-SSGA, Shuffle-SE",
        "Model training (per fold)",
        "Evaluation: mAP50, mAP50-95,\nparameters, GFLOPs",
        "Accuracy-efficiency analysis",
    ], "fig1_workflow", w=3.35, accent_idx=[2])


def fig2():
    flow([
        "Input feature map  U (H x W x C)",
        "Squeeze: global average pooling\nz (1 x 1 x C)",
        "Excitation: FC reduction (C -> C/r)",
        "Non-linear activation (ReLU)",
        "FC expansion (C/r -> C)",
        "Sigmoid channel weights  s",
        "Channel-wise recalibration  U x s",
        "Reweighted feature map",
    ], "fig2_se_module", w=3.15, accent_idx=[1, 6])


def fig3():
    m = json.loads((ROOT / "metrics.json").read_text())["results"]
    pts = []
    for stem, r in m.items():
        pts.append((r["gflops"], r["v2-1"]["mAP50-95"]["mean"], r["name"], "SE (proposed)" in r["name"]))
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for g, acc, name, prop in pts:
        ax.scatter(g, acc, s=95 if prop else 62,
                   color=ACC if prop else GREY, zorder=3,
                   marker="D" if prop else "o", edgecolor="white", linewidth=0.8)
    # labels with manual offsets to avoid overlap
    off = {"YOLOv8n (baseline)": (0.0, -0.0015, "center"),
           "YOLOv8-SE (proposed)": (0.14, -0.0011, "left"),
           "YOLOv8-Res2Net-SPD-SSGA": (0.0, 0.0016, "center"),
           "YOLOv8-Shuffle-SE": (0.14, 0.0, "left")}
    short = {"YOLOv8n (baseline)": "Baseline",
             "YOLOv8-SE (proposed)": "SE (proposed)",
             "YOLOv8-Res2Net-SPD-SSGA": "Res2Net-SPD-SSGA",
             "YOLOv8-Shuffle-SE": "Shuffle-SE"}
    for g, acc, name, prop in pts:
        dx, dy, ha = off[name]
        ax.annotate(short[name], (g, acc), (g + dx, acc + dy), ha=ha, va="center",
                    fontsize=7.6, color=ACC if prop else INK,
                    fontweight="bold" if prop else "normal")
    ax.annotate("preferred:\nhigher accuracy,\nlower computation",
                (7.55, 0.5952), fontsize=6.8, color=GREY, ha="left", va="top", style="italic")
    ax.add_patch(FancyArrowPatch((9.4, 0.5760), (7.7, 0.5905), arrowstyle="-|>",
                 mutation_scale=10, linewidth=1.0, color=GREY, linestyle=":"))
    ax.set_xlabel("Computational complexity (GFLOPs)")
    ax.set_ylabel("Judol Detection v2 mAP50-95")
    ax.set_xlim(7.0, 11.6); ax.set_ylim(0.573, 0.596)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "fig3_accuracy_complexity.pdf")
    fig.savefig(FIG / "fig3_accuracy_complexity.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("figures written to", FIG)
