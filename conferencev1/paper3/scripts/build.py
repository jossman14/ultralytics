"""Gather all data for the YOLO-generation benchmark paper and build figures.
Four standard nano detectors x two datasets x {mAP50,mAP50-95,P,R,per-fold},
plus params/GFLOPs and measured latency (GPU+CPU). Real values only."""
import json, statistics as st, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, yaml
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from ultralytics import YOLO

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
OUT = Path(__file__).resolve().parent.parent
FIG = OUT / "figures"
MODELS = [("yolov8n-base", "YOLOv8n", 2023), ("yolo11n-base", "YOLO11n", 2024),
          ("yolo12n-base", "YOLO12n", 2025), ("yolo26n-base", "YOLO26n", 2025)]
DS = [("v2-1", "Judol Detection v2"), ("judi3k", "Instagram Reels")]
plt.rcParams.update({"font.size": 12, "font.family": "serif"})
BLUE, GREY, RED = "#1a5fb4", "#8a8a8a", "#c64600"

led = json.loads((REPO / "judol_sweep/ledger.json").read_text())
bym = defaultdict(lambda: defaultdict(list))
for k, v in led.items():
    if v.get("status") == "ok":
        n, ds, f = k.split("|"); bym[n][ds].append((int(f[1:]), v))


def agg(n, ds, key):
    r = [x for _, x in sorted(bym[n][ds])]
    if len(r) != 5:
        return None
    vals = [x.get(key, 0) for x in r]
    return {"mean": round(st.mean(vals), 4), "std": round(st.pstdev(vals), 4),
            "vals": [round(x, 4) for x in vals]}


def pg(n):
    cfg = yaml.safe_load((RUNS / f"{n}_v2-1_f0" / "args.yaml").read_text())["model"]
    m = YOLO(cfg, task="detect")
    np_ = sum(p.numel() for p in m.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops
        g = get_flops(m.model, imgsz=640)
    except Exception:
        g = None
    return round(np_ / 1e6, 3), (round(g, 2) if g else None)


def latency(n, device, reps=60, warm=10):
    cfg = yaml.safe_load((RUNS / f"{n}_v2-1_f0" / "args.yaml").read_text())["model"]
    m = YOLO(cfg, task="detect").model.to(device).eval()
    x = torch.rand(1, 3, 640, 640, device=device)
    with torch.no_grad():
        for _ in range(warm):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
    return round(dt * 1000, 2), round(1 / dt, 1)


data = {"models": []}
devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for stem, name, year in MODELS:
    p, g = pg(stem)
    e = {"stem": stem, "name": name, "year": year, "params_M": p, "gflops": g, "metrics": {}, "lat": {}}
    for ds, _ in DS:
        e["metrics"][ds] = {m: agg(stem, ds, m) for m in ["mAP50", "mAP50-95", "precision", "recall"]}
    for dev in devs:
        ms, fps = latency(stem, dev)
        e["lat"][dev] = {"ms": ms, "fps": fps}
    data["models"].append(e)
    print(f"{name}: {p}M {g}G  v2-1 mAP50-95={e['metrics']['v2-1']['mAP50-95']['mean']:.4f}  "
          f"GPU {e['lat'].get('cuda',{}).get('fps','-')}FPS")

# statistics: Friedman across the 4 versions on hard dataset
from scipy import stats
mat = [[x.get("mAP50-95", 0) for _, x in sorted(bym[s][ ' v2-1'.strip()])] for s, _, _ in MODELS]
fr = stats.friedmanchisquare(*mat)
data["friedman_v21"] = {"stat": round(float(fr.statistic), 4), "p": round(float(fr.pvalue), 4)}
# Wilcoxon: v8 vs each newer
v8 = [x.get("mAP50-95", 0) for _, x in sorted(bym["yolov8n-base"]["v2-1"])]
data["wilcoxon_vs_v8"] = {}
for s, name, _ in MODELS[1:]:
    other = [x.get("mAP50-95", 0) for _, x in sorted(bym[s]["v2-1"])]
    w = stats.wilcoxon(v8, other)
    data["wilcoxon_vs_v8"][name] = round(float(w.pvalue), 4)
print("Friedman p =", data["friedman_v21"]["p"], "| Wilcoxon v8 vs newer:", data["wilcoxon_vs_v8"])
(OUT / "data.json").write_text(json.dumps(data, indent=1))

# ---- FIG 1: accuracy trend by release year ----
fig, ax = plt.subplots(figsize=(5.0, 3.6))
names = [m["name"] for m in data["models"]]
acc = [m["metrics"]["v2-1"]["mAP50-95"]["mean"] for m in data["models"]]
err = [m["metrics"]["v2-1"]["mAP50-95"]["std"] for m in data["models"]]
x = range(len(names))
ax.errorbar(x, acc, yerr=err, marker="o", markersize=9, linewidth=2, color=BLUE, capsize=4)
for i, a in enumerate(acc):
    ax.annotate(f"{a:.4f}", (i, a), (i, a + err[i] + 0.002), ha="center", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{m['name']}\n({m['year']})" for m in data["models"]], fontsize=10)
ax.set_ylabel("Judol Detection v2 mAP50-95")
ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "trend.pdf"); plt.close(fig); print("wrote trend.pdf")

# ---- FIG 2: accuracy vs params (efficiency) ----
fig, ax = plt.subplots(figsize=(5.0, 3.6))
for m in data["models"]:
    best = m["stem"] == "yolov8n-base"
    ax.scatter(m["params_M"], m["metrics"]["v2-1"]["mAP50-95"]["mean"], s=150,
               color=BLUE if best else GREY, edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(m["name"], (m["params_M"], m["metrics"]["v2-1"]["mAP50-95"]["mean"]),
                (m["params_M"], m["metrics"]["v2-1"]["mAP50-95"]["mean"] + 0.0015), ha="center", fontsize=10)
ax.set_xlabel("Parameters (M)"); ax.set_ylabel("Judol Detection v2 mAP50-95")
ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "efficiency.pdf"); plt.close(fig); print("wrote efficiency.pdf")

# ---- FIG 3: per-fold spread (box) on hard dataset ----
fig, ax = plt.subplots(figsize=(5.0, 3.4))
data_box = [m["metrics"]["v2-1"]["mAP50-95"]["vals"] for m in data["models"]]
bp = ax.boxplot(data_box, labels=names, patch_artist=True, widths=0.5)
for patch in bp["boxes"]:
    patch.set_facecolor("#dbe7ff")
ax.set_ylabel("Judol Detection v2 mAP50-95")
ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
plt.xticks(fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "spread.pdf"); plt.close(fig); print("wrote spread.pdf")

# ---- FIG 4: speed (GPU FPS) ----
fig, ax = plt.subplots(figsize=(5.0, 3.4))
fps = [m["lat"].get("cuda", {}).get("fps", 0) for m in data["models"]]
bars = ax.bar(names, fps, color=[BLUE if m["stem"] == "yolov8n-base" else GREY for m in data["models"]],
              edgecolor="white")
for b, f in zip(bars, fps):
    ax.text(b.get_x() + b.get_width() / 2, f + 2, f"{f:.0f}", ha="center", fontsize=10)
ax.set_ylabel("Inference speed (FPS, GPU)")
ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "speed.pdf"); plt.close(fig); print("wrote speed.pdf")
print("done")
