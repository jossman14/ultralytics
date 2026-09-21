"""Regenerate the results figures from the revision's runs.

Every figure is drawn from revisi/runs/**/metrics.json or the per-run results.csv,
so nothing here is hand-drawn or carried over from the submitted version. Figures
whose source data is not yet present are skipped with a message rather than drawn
from partial folds.
"""
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/ftib/ultralytics")
RUNS = ROOT / "revisi/runs"
FIGS = ROOT / "revisi/figures"
METRICS = ["precision", "recall", "f1", "map50", "map"]
LABELS = ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95"]
PRIMARY = "yolo12s"

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight"})


def load():
    d = defaultdict(list)
    for f in glob.glob(str(RUNS / "*/*/fold_*/metrics.json")):
        m = json.loads(Path(f).read_text())
        d[(m["dataset"], m["model"])].append(m)
    return d


def ms(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def need(d, keys, name):
    for k in keys:
        if len(d.get(k, [])) < 5:
            print(f"[skip] {name}: {k} has {len(d.get(k, []))}/5 folds")
            return False
    return True


def fig_protocol(d):
    keys = [("binary", PRIMARY), ("binary_videofold", PRIMARY)]
    if not need(d, keys, "fig_protocol"):
        return
    x = np.arange(len(METRICS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, (k, lab, c) in enumerate(zip(keys, ["Random 5-fold", "Video-disjoint 5-fold"],
                                        ["#7a9cc6", "#c6785a"])):
        stats = [ms([r[m] for r in d[k]]) for m in METRICS]
        ax.bar(x + (i - 0.5) * w, [s[0] for s in stats], w, yerr=[s[1] for s in stats],
               capsize=3, label=lab, color=c, edgecolor="black", linewidth=0.4)
        for xi, s in zip(x + (i - 0.5) * w, stats):
            ax.text(xi, s[0] + s[1] + 0.02, f"{s[0]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGS / "fig_protocol.png")
    plt.close(fig)
    print("[ok] fig_protocol.png")


def fig_models(d):
    models = ["yolov8s", "yolov10s", "yolo11s", "yolo12s"]
    for ds, title in [("binary_videofold", "Binary, video-disjoint"),
                      ("multiclass", "Multi-class")]:
        keys = [(ds, m) for m in models]
        if not need(d, keys, f"fig_models[{ds}]"):
            continue
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        x = np.arange(len(models))
        for i, (m, lab) in enumerate(zip(["map50", "map"], ["mAP@50", "mAP@50-95"])):
            stats = [ms([r[m] for r in d[(ds, mo)]]) for mo in models]
            ax.bar(x + (i - 0.5) * 0.36, [s[0] for s in stats], 0.36,
                   yerr=[s[1] for s in stats], capsize=3, label=lab,
                   color=["#7a9cc6", "#c6785a"][i], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, ["YOLOv8s", "YOLOv10s", "YOLO11s", "YOLOv12s"])
        ax.set_ylabel("Score")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(FIGS / f"fig_models_{ds}.png")
        plt.close(fig)
        print(f"[ok] fig_models_{ds}.png")


def fig_perclass(d):
    k = ("multiclass", PRIMARY)
    if not need(d, [k], "fig_perclass"):
        return
    cls = defaultdict(lambda: defaultdict(list))
    for r in d[k]:
        for name, v in r["per_class"].items():
            for m in ("recall", "map50", "map"):
                cls[name][m].append(v[m])
    names = sorted(cls, key=lambda n: st.mean(cls[n]["map50"]))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, (m, lab, c) in enumerate(zip(["recall", "map50", "map"],
                                        ["Recall", "mAP@50", "mAP@50-95"],
                                        ["#8fae8b", "#7a9cc6", "#c6785a"])):
        stats = [ms(cls[n][m]) for n in names]
        ax.bar(x + (i - 1) * 0.27, [s[0] for s in stats], 0.27, yerr=[s[1] for s in stats],
               capsize=2, label=lab, color=c, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGS / "fig_perclass.png")
    plt.close(fig)
    print("[ok] fig_perclass.png")


def fig_loss():
    for ds in ("binary_videofold", "multiclass"):
        csvs = sorted(glob.glob(str(RUNS / ds / PRIMARY / "fold_*/results.csv")))
        if len(csvs) < 5:
            print(f"[skip] fig_loss[{ds}]: {len(csvs)}/5 results.csv")
            continue
        cols = [("train/box_loss", "val/box_loss", "Box"),
                ("train/cls_loss", "val/cls_loss", "Class"),
                ("train/dfl_loss", "val/dfl_loss", "DFL")]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
        dfs = [pd.read_csv(c) for c in csvs]
        n = min(len(x) for x in dfs)
        for ax, (tr, va, name) in zip(axes, cols):
            for key, lab, c in ((tr, "Train", "#7a9cc6"), (va, "Validation", "#c6785a")):
                arr = np.stack([x[key].values[:n] for x in dfs])
                mu, sd = arr.mean(0), arr.std(0)
                ep = np.arange(1, n + 1)
                ax.plot(ep, mu, color=c, lw=1.2, label=lab)
                ax.fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.22, lw=0)
            ax.set_title(f"{name} loss", fontsize=9)
            ax.set_xlabel("Epoch")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Loss")
        axes[0].legend(frameon=False, fontsize=8)
        fig.savefig(FIGS / f"fig_loss_{ds}.png")
        plt.close(fig)
        print(f"[ok] fig_loss_{ds}.png")


def fig_xai():
    f = ROOT / "revisi/out/xai_eval.json"
    if not f.exists():
        print("[skip] fig_xai: no xai_eval.json")
        return
    data = json.loads(f.read_text())
    rows = [(ds, r) for ds, rs in data.items() for r in rs]
    if not rows:
        print("[skip] fig_xai: empty")
        return
    groups = [("pg_glob", "CAM\n(global)"), ("pg_box", "CAM\n(box-conditioned)"),
              ("pg_rand", "Random-box\ncontrol"), ("pg_sanity", "Randomized\nweights")]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    vals, errs = [], []
    for k, _ in groups:
        v = [r[k] for _, r in rows if r.get(k) is not None]
        m, s = ms(v) if v else (0.0, 0.0)
        vals.append(m)
        errs.append(s)
    ax.bar(range(len(groups)), vals, 0.6, yerr=errs, capsize=3,
           color=["#7a9cc6", "#9db8d4", "#b9b9b9", "#d6a0a0"],
           edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylabel("Pointing game accuracy")
    ax.set_ylim(0, max(vals) * 1.35 + 0.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGS / "fig_xai.png")
    plt.close(fig)
    print("[ok] fig_xai.png")




def fig_workflow():
    """Workflow diagram. Drawn here rather than reused from the submitted version,
    whose single-protocol flow no longer describes the experiment."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(-0.05, 10.05); ax.set_ylim(1.85, 5.15); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec="black", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.7, color="black"))

    blue, orange, green, grey = "#cfdcea", "#f0d5c8", "#d5e3d3", "#e4e4e4"

    box(0.0, 3.9, 2.45, 0.9, "Instagram Reels\n3,000 frames\n300 videos", blue)
    box(0.0, 2.5, 2.45, 0.9, "Multi-class set\n443 images\n932 marks", blue)

    box(2.75, 4.3, 2.15, 0.75, "Random 5-fold\n(frame-level)", orange)
    box(2.75, 3.3, 2.15, 0.75, "Video-disjoint 5-fold\n(StratifiedGroupKFold)", orange)
    box(2.75, 2.55, 2.15, 0.6, "Published 5-fold", orange)

    box(5.3, 3.2, 2.0, 1.6, "Train under identical\nhyperparameters\n\nYOLOv8s / v10s\nYOLO11s / v12s", green)

    box(7.8, 4.1, 2.0, 0.85, "Detection metrics\nP, R, F1, mAP@50,\nmAP@50-95", grey)
    box(7.8, 3.05, 2.0, 0.85, "Latency\ndetection vs\nexplanation", grey)
    box(7.8, 1.95, 2.0, 0.9, "Explanation quality\npointing game, EBPG,\nrandom-box + sanity", grey)

    box(5.3, 1.95, 2.0, 0.95, "Multi-scale\nactivation CAM\nP3 / P4 / P5", green)

    arrow(2.45, 4.45, 2.75, 4.65); arrow(2.45, 4.15, 2.75, 3.7)
    arrow(2.45, 2.95, 2.75, 2.85)
    for y in (4.65, 3.65, 2.85):
        arrow(4.9, y, 5.3, 4.0 if y > 3.5 else 3.5)
    arrow(6.3, 3.2, 6.3, 2.9)
    arrow(7.3, 4.3, 7.8, 4.5); arrow(7.3, 3.9, 7.8, 3.5)
    arrow(7.3, 2.4, 7.8, 2.4)

    fig.savefig(FIGS / "fig_workflow.png")
    plt.close(fig)
    print("[ok] fig_workflow.png")


def fig_cam_examples():
    """Qualitative CAM panels, drawn with the same ActivationCAM used for the
    quantitative scores in Table 12 so the figure and the numbers describe one method.

    Every panel is drawn in the CAM's own 640x640 input space. The sources are
    portrait Reels and landscape screenshots, so plotting them at native aspect
    ratio left the grid ragged and the titles on different baselines.
    """
    import sys
    sys.path.insert(0, str(ROOT / "revisi/scripts"))
    import cv2
    import torch
    from xai_eval import ActivationCAM, IMG_SIZE, load_gt

    for ds, folder, n_show in (("binary_videofold", "judol/dataset_judi_online_yolo_videofold", 4),
                               ("multiclass", "judol/Judol-Detection-v2-1_5fold", 5)):
        w = RUNS / ds / PRIMARY / "fold_0/weights/best.pt"
        if not w.exists():
            print(f"[skip] fig_cam[{ds}]: no weights"); continue
        vdir = ROOT / folder / "fold_0/valid/images"
        agent = ActivationCAM(str(w), "cuda:0" if torch_ok() else "cpu")
        picked, seen, groups = [], set(), set()
        # Two passes: the first takes one example per class so the panel spans the
        # label set, the second fills any remaining slots. Binary frames are named
        # <reel>_<frame>, and consecutive frames of one Reel are near-duplicates,
        # so one panel per Reel keeps the row showing four different posts.
        for require_new in (True, False):
            for ip in sorted(vdir.glob("*.jpg")):
                if len(picked) >= n_show:
                    break
                img = cv2.imread(str(ip))
                if img is None:
                    continue
                h, wd = img.shape[:2]
                lpath = vdir.parent / "labels" / f"{ip.stem}.txt"
                gts = load_gt(lpath, h, wd)
                if not gts:
                    continue
                res = agent.model.predict(str(ip), imgsz=IMG_SIZE, verbose=False)[0]
                if len(res.boxes) == 0:
                    continue
                b = int(res.boxes.conf.argmax())
                label = res.names[int(res.boxes.cls[b])]
                group = ip.stem.rsplit("_", 1)[0] if ds.startswith("binary") else ip.stem
                if group in groups or (require_new and label in seen):
                    continue
                seen.add(label); groups.add(group)
                box = res.boxes.xyxy[b].cpu().numpy()
                sx, sy = IMG_SIZE / wd, IMG_SIZE / h
                cam_box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
                sq = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
                t = torch.from_numpy(sq).permute(2, 0, 1).float().div(255).unsqueeze(0).to(agent.device)
                agent.forward(t)
                cam = agent.cam(cam_box, (IMG_SIZE, IMG_SIZE))
                # Name the ground truth when the panel is a miss, so a wrong
                # prediction cannot be read as a correct one.
                k = max(range(len(gts)), key=lambda n: (gts[n][2] - gts[n][0]) * (gts[n][3] - gts[n][1]))
                truth = res.names[int(lpath.read_text().strip().splitlines()[k].split()[0])]
                title = f"{label} {float(res.boxes.conf[b]):.2f}"
                picked.append((sq, cam, cam_box, title if truth == label else f"{title}\n(true: {truth})"))

        if not picked:
            print(f"[skip] fig_cam[{ds}]: no detections"); continue
        fig, axes = plt.subplots(2, len(picked), figsize=(1.85 * len(picked), 4.0))
        axes = np.atleast_2d(axes)
        for j, (sq, cam, box, lab) in enumerate(picked):
            for row in (0, 1):
                a = axes[row, j]
                a.imshow(sq)
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_edgecolor("#999999"); sp.set_linewidth(0.6)
            x1, y1, x2, y2 = box
            axes[0, j].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, ec="#e04a2f", lw=1.4))
            axes[0, j].set_title(lab, fontsize=7, pad=3)
            im = axes[1, j].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[0, 0].set_ylabel("Detection", fontsize=8)
        axes[1, 0].set_ylabel("Activation CAM", fontsize=8)
        fig.tight_layout(pad=0.4, w_pad=0.25, h_pad=0.25)
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.012)
        cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
        cb.ax.tick_params(labelsize=7)
        fig.savefig(FIGS / f"fig_cam_{ds}.png")
        plt.close(fig)
        print(f"[ok] fig_cam_{ds}.png ({len(picked)} panels)")


def torch_ok():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    d = load()
    fig_protocol(d)
    fig_models(d)
    fig_perclass(d)
    fig_loss()
    fig_xai()
    fig_workflow()
    fig_cam_examples()
