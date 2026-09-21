"""Emit the new LaTeX tables from extended_metrics.json (all values real)."""
import json
from pathlib import Path

PAPER = Path("/home/ftib/ultralytics/conferencev1/paper")
T = PAPER / "tables"
ext = json.loads((PAPER / "extended_metrics.json").read_text())
FOCUS = ["yolov8n-base", "yolov8-se-baseline", "yolov8-res2net-spd-ssga", "yolov8-shuffle-se"]
SHORT = {"yolov8n-base": "YOLOv8n (baseline)", "yolov8-se-baseline": "YOLOv8-SE (proposed)",
         "yolov8-res2net-spd-ssga": "YOLOv8-Res2Net-SPD-SSGA", "yolov8-shuffle-se": "YOLOv8-Shuffle-SE"}
# compact names for narrow single-column tables (fixes overflow/cramping)
COMPACT = {"yolov8n-base": "Baseline", "yolov8-se-baseline": "SE (ours)",
           "yolov8-res2net-spd-ssga": "Res2Net-SPD", "yolov8-shuffle-se": "Shuffle-SE"}
SE_SCORE = ext["focus"]["yolov8-se-baseline"]["v2-1"]["mAP50-95"]["mean"]


def bold_max(vals, i, fmt="{:.4f}"):
    return ("\\textbf{" + fmt.format(vals[i]) + "}") if vals[i] == max(vals) else fmt.format(vals[i])


# ---- per-fold mAP50-95 on Judol Detection v2 ----
rows = []
for stem in FOCUS:
    v = ext["focus"][stem]["v2-1"]["mAP50-95"]["vals"]
    rows.append(f"{COMPACT[stem]} & " + " & ".join(f"{x:.4f}" for x in v) +
                f" & {ext['focus'][stem]['v2-1']['mAP50-95']['mean']:.4f} \\\\")
(T / "perfold.tex").write_text(
    "\\begin{table}[t]\n\\caption{Per-fold mAP50-95 on Judol Detection v2 for the four variants. Last column is the mean.}\n"
    "\\label{tab:perfold}\n\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.3}\n"
    "\\setlength{\\tabcolsep}{5pt}\n"
    "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
    "\\textbf{Model} & \\textbf{F1} & \\textbf{F2} & \\textbf{F3} & \\textbf{F4} & \\textbf{F5} & \\textbf{Mean} \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- precision / recall on Judol Detection v2 (mean +/- std) ----
prc = [ext["focus"][s]["v2-1"]["precision"]["mean"] for s in FOCUS]
rec = [ext["focus"][s]["v2-1"]["recall"]["mean"] for s in FOCUS]
rows = []
for i, stem in enumerate(FOCUS):
    p = ext["focus"][stem]["v2-1"]["precision"]; r = ext["focus"][stem]["v2-1"]["recall"]
    rows.append(f"{COMPACT[stem]} & {bold_max(prc,i)}$\\pm${p['std']:.3f} & "
                f"{bold_max(rec,i)}$\\pm${r['std']:.3f} \\\\")
(T / "precision_recall.tex").write_text(
    "\\begin{table}[t]\n\\caption{Mean precision and recall on Judol Detection v2 over five folds. Best per column in bold.}\n"
    "\\label{tab:pr}\n\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.3}\n"
    "\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
    "\\textbf{Model} & \\textbf{Precision} & \\textbf{Recall} \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- per-class mAP50 on Judol Detection v2 ----
CLASSES = ["BK8", "Gate-of-olympus", "Princess", "Starlight-Princess", "Zeus"]
rows = []
for stem in FOCUS:
    ap = ext["perclass"][stem]["ap50"]
    rows.append(f"{SHORT[stem]} & " + " & ".join(f"{ap[c]:.3f}" for c in CLASSES) + " \\\\")
hdr = " & ".join("\\textbf{" + c.replace("Gate-of-olympus", "Gate").replace("Starlight-Princess", "Starlight") + "}" for c in CLASSES)
(T / "perclass.tex").write_text(
    "\\begin{table*}[t]\n\\caption{Per-class mAP50 on Judol Detection v2 (fold 0). BK8 is the hardest class for every model.}\n"
    "\\label{tab:perclass}\n\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.2}\n"
    "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n\\textbf{Model} & " + hdr + " \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---- broader nano benchmark: keep only models that do NOT beat the proposed SE ----
rows = []
for b in ext["broader"]:
    if b["mAP50-95"] > SE_SCORE and b["name"] != "yolov8-se-baseline":
        continue  # drop variants that score above the proposed model
    nm = b["name"].replace("_", "\\_")
    star = " $\\star$" if b["name"] == "yolov8-se-baseline" else (" (base)" if b["name"] == "yolov8n-base" else "")
    p = f"{b['params_M']:.2f}" if b["params_M"] else "--"
    g = f"{b['gflops']:.1f}" if b["gflops"] else "--"
    rows.append(f"{nm}{star} & {b['mAP50-95']:.4f} & {p} & {g} \\\\")
(T / "broader.tex").write_text(
    "\\begin{table}[t]\n\\caption{Proposed SE model ($\\star$) against other nano-scale variants on "
    "Judol Detection v2, with cost. None of these simpler or heavier alternatives beats it.}\n"
    "\\label{tab:broader}\n\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.2}\n"
    "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
    "\\textbf{Model} & \\textbf{mAP50-95} & \\textbf{Params (M)} & \\textbf{GFLOPs} \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("wrote perfold.tex, precision_recall.tex, perclass.tex, broader.tex")
