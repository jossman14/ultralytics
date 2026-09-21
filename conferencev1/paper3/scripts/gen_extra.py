"""Generate the extra figures/tables so paper3 mirrors the SE paper's flow:
convergence, per-class AP, confusion matrices (best vs worst), detections.
CPU. Real data."""
import csv, json, statistics as st
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from ultralytics import YOLO

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
P = Path(__file__).resolve().parent.parent
FIG = P / "figures"; T = P / "tables"
DATA = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/data.yaml"
IMGDIR = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/valid/images"
CLS = ["BK8", "Gate-of-olympus", "Princess", "Starlight-Princess", "Zeus"]
MODELS = [("yolov8n-base", "YOLOv8n"), ("yolo11n-base", "YOLO11n"),
          ("yolo12n-base", "YOLO12n"), ("yolo26n-base", "YOLO26n")]
plt.rcParams.update({"font.size": 12, "font.family": "serif"})
COL = {"yolov8n-base": "#1a5fb4", "yolo11n-base": "#2e7d32",
       "yolo12n-base": "#8a6d00", "yolo26n-base": "#c64600"}


def conv(stem, metric="metrics/mAP50-95(B)"):
    series = []
    for f in range(5):
        p = RUNS / f"{stem}_v2-1_f{f}" / "results.csv"
        if not p.exists():
            continue
        series.append([float(r[metric]) for r in csv.DictReader(p.open())])
    if not series:
        return []
    n = min(len(s) for s in series)
    return [round(st.mean(s[i] for s in series), 4) for i in range(n)]


def plot_cm(matrix, out, big=17):
    m = matrix.astype(float); m = m / (m.sum(0, keepdims=True) + 1e-9)
    labels = CLS + ["background"]; n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.15), max(5, n)))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if m[i, j] >= 0.005:
                ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                        color="white" if m[i, j] > 0.5 else "black", fontsize=big)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=big); ax.set_yticklabels(labels, fontsize=big)
    ax.set_xlabel("True", fontsize=big + 2); ax.set_ylabel("Predicted", fontsize=big + 2)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=big - 3)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def montage(tiles, out, cols=3, pad=6):
    rows = (len(tiles) + cols - 1) // cols; ch, cw = tiles[0].shape[:2]
    canvas = np.full((rows * ch + (rows + 1) * pad, cols * cw + (cols + 1) * pad, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); y, x = pad + r * (ch + pad), pad + c * (cw + pad)
        canvas[y:y + ch, x:x + cw] = cv2.resize(t, (cw, ch))
    cv2.imwrite(str(out), canvas); print("wrote", out)


def per_class(stem):
    m = YOLO(str(RUNS / f"{stem}_v2-1_f0" / "weights" / "best.pt"))
    r = m.val(data=str(DATA), split="val", device="cpu", plots=False, verbose=False)
    ap = {}
    for i, ci in enumerate(r.box.ap_class_index):
        ap[CLS[ci]] = round(float(r.box.ap50[i]), 4)
    return {c: ap.get(c, 0.0) for c in CLS}


def main():
    # convergence
    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    for stem, name in MODELS:
        y = conv(stem)
        ax.plot(range(1, len(y) + 1), y, color=COL[stem], linewidth=2.0, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Judol Detection v2 mAP50-95")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "convergence.pdf"); plt.close(fig); print("wrote convergence.pdf")

    # per-class (4 models) + table + figure
    pc = {stem: per_class(stem) for stem, _ in MODELS}
    x = np.arange(len(CLS)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for j, (stem, name) in enumerate(MODELS):
        ax.bar(x + (j - 1.5) * w, [pc[stem][c] for c in CLS], w, label=name, color=COL[stem], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CLS, fontsize=10); ax.set_ylabel("mAP50 (fold 0)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "perclass.pdf"); plt.close(fig); print("wrote perclass.pdf")
    rows = []
    for stem, name in MODELS:
        rows.append(f"{name} & " + " & ".join(f"{pc[stem][c]:.3f}" for c in CLS) + " \\\\")
    hdr = " & ".join("\\textbf{" + c.replace("Gate-of-olympus", "Gate").replace("Starlight-Princess", "Starlight") + "}" for c in CLS)
    (T / "perclass.tex").write_text(
        "\\begin{table*}[t]\n\\caption{Per-class mAP50 on Judol Detection v2 (fold 0). "
        "BK8 is the hardest class, and the newest models lose the most on it.}\n\\label{tab:perclass}\n"
        "\\centering\\footnotesize\\renewcommand{\\arraystretch}{1.3}\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n\\textbf{Model} & " + hdr + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print("wrote perclass.tex")

    # confusion matrices: best (v8) vs worst (v26)
    for stem, tag in [("yolov8n-base", "v8"), ("yolo26n-base", "v26")]:
        m = YOLO(str(RUNS / f"{stem}_v2-1_f0" / "weights" / "best.pt"))
        r = m.val(data=str(DATA), split="val", device="cpu", plots=True,
                  project=str(FIG / "_v"), name=tag, exist_ok=True, verbose=False)
        plot_cm(r.confusion_matrix.matrix, FIG / f"cm_{tag}.png")

    # detections (v8)
    m = YOLO(str(RUNS / "yolov8n-base_v2-1_f0" / "weights" / "best.pt"))
    imgs = sorted(IMGDIR.glob("*.jpg"))[:6]
    pr = m.predict(source=[str(p) for p in imgs], conf=0.25, device="cpu", save=True,
                   project=str(FIG / "_pred"), name="v8", exist_ok=True, line_width=3)
    saved = sorted(Path(pr[0].save_dir).glob("*.jpg"))[:6]
    montage([cv2.imread(str(s)) for s in saved], FIG / "detections.png", cols=3)
    print("done")


if __name__ == "__main__":
    main()
