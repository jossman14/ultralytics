"""Analytical figures from data.json / experiments.json (no GPU)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = Path(__file__).resolve().parent.parent
FIG = P / "figures"; FIG.mkdir(exist_ok=True)
data = json.loads((P / "data.json").read_text())
exp = json.loads((P / "experiments.json").read_text())
plt.rcParams.update({"font.size": 12, "font.family": "serif"})
ACC, GREY, RED, GREEN = "#1a5fb4", "#8a8a8a", "#c64600", "#2e7d32"

# ---- 1. design-space scatter: v2-1 vs judi3k, complexity trap ----
sc = data["scatter"]
fig, ax = plt.subplots(figsize=(5.2, 4.2))
base = data["design"]["baseline_v21"]
for s in sc:
    if s["proposed"] or s["baseline"]:
        continue
    ax.scatter(s["v21"], s["judi3k"], s=34,
               color=GREEN if s["beats"] else GREY, alpha=0.75,
               edgecolor="white", linewidth=0.4, zorder=2)
for s in sc:
    if s["baseline"]:
        ax.scatter(s["v21"], s["judi3k"], s=150, marker="s", color=RED, zorder=4,
                   edgecolor="white", linewidth=1, label="YOLOv8n baseline")
    if s["proposed"]:
        ax.scatter(s["v21"], s["judi3k"], s=180, marker="D", color=ACC, zorder=5,
                   edgecolor="white", linewidth=1, label="Proposed")
ax.axvline(base, color=RED, linestyle="--", linewidth=1, alpha=0.7)
ax.text(base + 0.002, 0.55, "baseline\nv2-1 level", color=RED, fontsize=9, va="center")
ax.scatter([], [], s=34, color=GREEN, label="beats baseline")
ax.scatter([], [], s=34, color=GREY, label="below baseline")
ax.set_xlabel("Judol Detection v2 mAP50-95 (hard)")
ax.set_ylabel("Instagram Reels mAP50-95 (easy)")
ax.legend(fontsize=9, loc="lower left"); ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "designspace.pdf"); plt.close(fig)
print("wrote designspace.pdf")

# ---- 2. efficiency frontier: accuracy vs params ----
comp = data["comparison"]
fig, ax = plt.subplots(figsize=(5.2, 4.0))
NM = {"yolov8n-base": "YOLOv8n", "yolo11n-base": "YOLO11n", "yolo12n-base": "YOLO12n",
      "yolo26n-base": "YOLO26n", "yolov8-se-baseline": "YOLOv8-SE",
      "yolov8-spd-ghostv2-sgsa": "SPD-GhostV2", "yolov8-shuffle-se": "Shuffle-SE",
      "yolov8-res2net-spd-ssga": "Res2Net-SPD", "yolov8-spd-shuffle-simam": "Proposed"}
for c in comp:
    if not c["v21_map5095"]:
        continue
    prop = c["name"] == "yolov8-spd-shuffle-simam"
    ax.scatter(c["params_M"], c["v21_map5095"]["mean"], s=180 if prop else 70,
               marker="D" if prop else "o", color=ACC if prop else GREY,
               edgecolor="white", linewidth=0.8, zorder=4 if prop else 3)
    ax.annotate(NM.get(c["name"], c["name"]), (c["params_M"], c["v21_map5095"]["mean"]),
                (c["params_M"], c["v21_map5095"]["mean"] + 0.0016), fontsize=8.5, ha="center",
                color=ACC if prop else "black", fontweight="bold" if prop else "normal")
ax.set_xlabel("Parameters (M)"); ax.set_ylabel("Judol Detection v2 mAP50-95")
ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "efficiency.pdf"); plt.close(fig)
print("wrote efficiency.pdf")

# ---- 3. latency bars (GPU FPS) ----
lat = exp["latency"]
order = ["yolo26n-base", "yolo11n-base", "yolov8n-base", "yolov8-se-baseline",
         "yolov8-spd-shuffle-simam", "yolov8-res2net-spd-ssga"]
order = [n for n in order if n in lat]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
labels = [NM.get(n, n) for n in order]
fps = [lat[n].get("cuda", {}).get("fps", 0) for n in order]
cols = [ACC if n == "yolov8-spd-shuffle-simam" else (RED if "res2net" in n else GREY) for n in order]
bars = ax.bar(labels, fps, color=cols, edgecolor="white")
for b, f in zip(bars, fps):
    ax.text(b.get_x() + b.get_width() / 2, f + 2, f"{f:.0f}", ha="center", fontsize=9)
ax.axhline(30, color="black", linestyle=":", linewidth=1)
ax.text(0, 33, "30 FPS real-time", fontsize=8.5)
ax.set_ylabel("Inference speed (FPS, GPU)")
plt.xticks(rotation=20, ha="right", fontsize=9)
ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "latency.pdf"); plt.close(fig)
print("wrote latency.pdf")

# ---- 4. proposed architecture schematic ----
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
blocks = [("Input\n640x640", GREY), ("Stem\nConv", GREY), ("SPD-Conv\ndownsample", ACC),
          ("Shuffle\nBlock", ACC), ("SPD-Conv\ndownsample", ACC), ("Shuffle\nBlock", ACC),
          ("SimAM\n(param-free)", GREEN), ("SPPF", GREY), ("PAN\nNeck", GREY), ("Detect\nP3/P4/P5", GREY)]
w = 1.05
for i, (txt, col) in enumerate(blocks):
    x = 0.3 + i * 1.15
    ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.0, boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.2, edgecolor=col, facecolor="#eef2fb" if col == ACC else ("#eaf6ec" if col == GREEN else "#f4f4f4")))
    ax.text(x + w / 2, 1.5, txt, ha="center", va="center", fontsize=8.2)
    if i < len(blocks) - 1:
        ax.add_patch(FancyArrowPatch((x + w, 1.5), (x + 1.15, 1.5), arrowstyle="-|>",
                     mutation_scale=10, linewidth=1, color="#555"))
ax.text(6, 2.6, "SPD-Conv + ShuffleNet + parameter-free SimAM attention",
        ha="center", fontsize=9.5, style="italic")
fig.tight_layout(); fig.savefig(FIG / "architecture.pdf"); plt.close(fig)
print("wrote architecture.pdf")
