"""Generate LaTeX tables for the applied paper from data.json + experiments.json."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent.parent
T = P / "tables"; T.mkdir(exist_ok=True)
d = json.loads((P / "data.json").read_text())
e = json.loads((P / "experiments.json").read_text())
NM = {"yolov8n-base": "YOLOv8n", "yolo11n-base": "YOLO11n", "yolo12n-base": "YOLO12n",
      "yolo26n-base": "YOLO26n", "yolov8-se-baseline": "YOLOv8-SE",
      "yolov8-spd-ghostv2-sgsa": "SPD-GhostV2-SGSA", "yolov8-shuffle-se": "Shuffle-SE",
      "yolov8-res2net-spd-ssga": "Res2Net-SPD-SSGA", "yolov8-spd-shuffle-simam": "\\textbf{Proposed}"}
comp = d["comparison"]
PROP = "yolov8-spd-shuffle-simam"

# best value per metric for bolding
best = {}
for key, sub in [("v21_map50", None), ("v21_map5095", None), ("v21_R", None), ("ju_map5095", None)]:
    vals = [c[key]["mean"] for c in comp if c.get(key)]
    best[key] = max(vals)


def cell(c, key, fmt="{:.4f}"):
    if not c.get(key):
        return "--"
    v = c[key]["mean"]
    s = fmt.format(v)
    return "\\textbf{" + s + "}" if abs(v - best[key]) < 1e-9 else s


# ---- Table: main comparison ----
rows = []
for c in comp:
    fps = e["latency"].get(c["name"], {}).get("cuda", {}).get("fps")
    fps_s = f"{fps:.0f}" if fps else "--"
    rows.append(f"{NM[c['name']]} & {c['params_M']:.2f} & {c['gflops']:.1f} & {fps_s} & "
                f"{cell(c,'v21_map50')} & {cell(c,'v21_map5095')} & {cell(c,'v21_R','{:.3f}')} & "
                f"{cell(c,'ju_map5095')} \\\\")
(T / "comparison.tex").write_text(
    "\\begin{table*}[t]\n\\centering\n"
    "\\caption{Detection accuracy, complexity, and speed on both datasets (five-fold means). "
    "Standard detectors, design-space variants, and the proposed model. Best per column in bold. "
    "FPS measured on an NVIDIA RTX A4000 at $640\\times640$.}\n\\label{tab:comparison}\n"
    "\\begin{tabular}{@{}lrrrcccc@{}}\n\\toprule\n"
    "& & & & \\multicolumn{3}{c}{\\textbf{Judol Detection v2 (hard)}} & \\textbf{IG Reels} \\\\\n"
    "\\cmidrule(lr){5-7}\\cmidrule(lr){8-8}\n"
    "\\textbf{Model} & \\textbf{Par.(M)} & \\textbf{GFLOPs} & \\textbf{FPS} & "
    "\\textbf{mAP50} & \\textbf{mAP50-95} & \\textbf{Recall} & \\textbf{mAP50-95} \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---- Table: proposed per-fold on hard dataset ----
pf = d["proposed"]["v2-1"]["mAP50-95"]["vals"]
mean = d["proposed"]["v2-1"]["mAP50-95"]["mean"]; std = d["proposed"]["v2-1"]["mAP50-95"]["std"]
(T / "perfold.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Per-fold mAP50-95 of the proposed model on Judol Detection v2.}\n"
    "\\label{tab:perfold}\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
    "& F1 & F2 & F3 & F4 & F5 & Mean$\\pm$SD \\\\\n\\midrule\n"
    "Proposed & " + " & ".join(f"{v:.4f}" for v in pf) + f" & {mean:.4f}$\\pm${std:.3f} \\\\\n"
    "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- Table: statistical tests ----
w = e["stats"]["wilcoxon_proposed_vs_baseline_v21"]; fr = e["stats"]["friedman_v21"]
(T / "stats.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Statistical tests on per-fold mAP50-95 (Judol Detection v2, "
    "five folds). Improvements are consistent but not significant at $\\alpha=0.05$ given the small fold count.}\n"
    "\\label{tab:stats}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
    "\\textbf{Test} & \\textbf{Comparison} & \\textbf{Statistic} & \\textbf{$p$-value} \\\\\n\\midrule\n"
    f"Wilcoxon signed-rank & Proposed vs YOLOv8n & {w['statistic']:.1f} & {w['pvalue']:.3f} \\\\\n"
    f"Friedman & {fr['k']} models & {fr['statistic']:.2f} & {fr['pvalue']:.3f} \\\\\n"
    "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("wrote comparison.tex, perfold.tex, stats.tex")
