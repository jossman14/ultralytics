import json
from pathlib import Path
P = Path(__file__).resolve().parent.parent
d = json.loads((P / "data.json").read_text())
M = d["models"]

# best per metric (higher better) for bolding on hard dataset
def bestval(ds, key): return max(m["metrics"][ds][key]["mean"] for m in M)

def c(m, ds, key, fmt="{:.4f}"):
    v = m["metrics"][ds][key]["mean"]; s = fmt.format(v)
    return "\\textbf{" + s + "}" if abs(v - bestval(ds, key)) < 1e-9 else s

# ---- main results table* ----
rows = []
for m in M:
    fps = m["lat"].get("cuda", {}).get("fps", "--")
    rows.append(f"{m['name']} ({m['year']}) & {m['params_M']:.2f} & {m['gflops']:.1f} & {fps:.0f} & "
                f"{c(m,'v2-1','mAP50')} & {c(m,'v2-1','mAP50-95')} & {c(m,'v2-1','recall','{:.3f}')} & "
                f"{c(m,'judi3k','mAP50-95')} \\\\")
(P / "tables/main.tex").write_text(
    "\\begin{table*}[t]\n\\caption{Accuracy, complexity, and speed of four standard nano detectors "
    "on both datasets (five-fold means). Best per column in bold. FPS on an NVIDIA RTX A4000 at "
    "$640\\times640$. Newer generations are smaller but less accurate on the hard dataset.}\n"
    "\\label{tab:main}\n\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.3}\n"
    "\\begin{tabular}{@{}lrrrcccc@{}}\n\\toprule\n"
    "& & & & \\multicolumn{3}{c}{\\textbf{Judol Detection v2 (hard)}} & \\textbf{IG Reels} \\\\\n"
    "\\cmidrule(lr){5-7}\\cmidrule(lr){8-8}\n"
    "\\textbf{Model} & \\textbf{Par.(M)} & \\textbf{GFLOPs} & \\textbf{FPS} & "
    "\\textbf{mAP50} & \\textbf{mAP50-95} & \\textbf{Recall} & \\textbf{mAP50-95} \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---- per-fold + relative-to-v8 table ----
v8 = next(m for m in M if m["stem"] == "yolov8n-base")
base = v8["metrics"]["v2-1"]["mAP50-95"]["mean"]
rows = []
for m in M:
    mm = m["metrics"]["v2-1"]["mAP50-95"]
    dv = mm["mean"] - base
    dp = 100 * (m["params_M"] - v8["params_M"]) / v8["params_M"]
    rows.append(f"{m['name']} & {mm['mean']:.4f}$\\pm${mm['std']:.3f} & "
                f"{'--' if m['stem']=='yolov8n-base' else f'{dv:+.4f}'} & "
                f"{'--' if m['stem']=='yolov8n-base' else f'{dp:+.1f}\\%'} \\\\")
(P / "tables/relative.tex").write_text(
    "\\begin{table}[t]\n\\caption{Change relative to YOLOv8n on the hard dataset. "
    "Newer models save parameters but lose accuracy.}\n\\label{tab:relative}\n"
    "\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.3}\n"
    "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
    "\\textbf{Model} & \\textbf{mAP50-95} & \\textbf{$\\Delta$Acc} & \\textbf{$\\Delta$Params} \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- stats table ----
fr = d["friedman_v21"]; w = d["wilcoxon_vs_v8"]
wl = " ".join(f"{k}: $p={v:.3f}$;" for k, v in w.items())
(P / "tables/stats.tex").write_text(
    "\\begin{table}[t]\n\\caption{Statistical tests on per-fold mAP50-95 (hard dataset). The Friedman "
    "test confirms the differences across the four generations are significant.}\n\\label{tab:stats}\n"
    "\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.3}\n"
    "\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
    "\\textbf{Test} & \\textbf{Statistic} & \\textbf{$p$-value} \\\\\n\\midrule\n"
    f"Friedman (4 models) & {fr['stat']:.2f} & \\textbf{{{fr['p']:.3f}}} \\\\\n"
    f"Wilcoxon YOLOv8n vs YOLO11n & -- & {w['YOLO11n']:.3f} \\\\\n"
    f"Wilcoxon YOLOv8n vs YOLO12n & -- & {w['YOLO12n']:.3f} \\\\\n"
    f"Wilcoxon YOLOv8n vs YOLO26n & -- & {w['YOLO26n']:.3f} \\\\\n"
    "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("wrote main.tex, relative.tex, stats.tex")
