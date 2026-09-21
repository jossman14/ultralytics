"""Regenerate the v2-1 confusion matrix with large fonts, and produce judi3k
figures (confusion matrix, detections, Grad-CAM) for the second-dataset analysis.
All on CPU so the running GPU sweep is undisturbed. Real data only.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
FIG = REPO / "conferencev1/paper/figures"
IMGSZ = 640
lb = LetterBox((IMGSZ, IMGSZ))

V21_W = RUNS / "yolov8-se-baseline_v2-1_f0/weights/best.pt"
V21_D = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/data.yaml"
V21_CLS = ["BK8", "Gate-of-olympus", "Princess", "Starlight-Princess", "Zeus"]
JU_W = RUNS / "yolov8-se-baseline_judi3k_f0/weights/best.pt"
JU_D = REPO / "judol/dataset_judi_online_yolo_strat5fold/fold_0/data.yaml"
JU_IMG = REPO / "judol/dataset_judi_online_yolo_strat5fold/fold_0/valid/images"
JU_LBL = REPO / "judol/dataset_judi_online_yolo_strat5fold/fold_0/valid/labels"
JU_CLS = ["gambling", "nongambling"]


def plot_cm(matrix, names, out, big=17):
    """Column-normalized confusion matrix with large fonts."""
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
    ax.set_xlabel("True", fontsize=big + 2)
    ax.set_ylabel("Predicted", fontsize=big + 2)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=big - 3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def montage(tiles, out, cols=3, pad=6):
    rows = (len(tiles) + cols - 1) // cols
    ch, cw = tiles[0].shape[:2]
    canvas = np.full((rows * ch + (rows + 1) * pad, cols * cw + (cols + 1) * pad, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (ch + pad), pad + c * (cw + pad)
        canvas[y:y + ch, x:x + cw] = cv2.resize(t, (cw, ch))
    cv2.imwrite(str(out), canvas)
    print("wrote", out)


def gradcam_tiles(weights, images, idx=(22, 23, 24)):
    net = YOLO(str(weights)).model.eval()
    for p in net.parameters():
        p.requires_grad_(True)
    targets = [net.model[i] for i in idx]
    tiles = []
    for ip in images:
        bgr = lb(image=cv2.imread(str(ip)))
        acts, grads, hooks = {}, {}, []
        for k, t in enumerate(targets):
            hooks.append(t.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(k, o)))
            hooks.append(t.register_full_backward_hook(lambda m, gi, go, k=k: grads.__setitem__(k, go[0])))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().requires_grad_(True)
        out = net(x)
        pred = out[0] if isinstance(out, (list, tuple)) else out
        net.zero_grad()
        pred[:, 4:, :].max().backward()
        agg = np.zeros((IMGSZ, IMGSZ), np.float32)
        for k in range(len(targets)):
            A, G = acts[k], grads[k]
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * A).sum(1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, (IMGSZ, IMGSZ), mode="bilinear", align_corners=False)
            c = cam[0, 0].detach().numpy()
            agg += (c - c.min()) / (c.max() - c.min() + 1e-8)
        for h in hooks:
            h.remove()
        agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)
        hm = cv2.applyColorMap(np.uint8(255 * agg), cv2.COLORMAP_JET)
        tiles.append(cv2.addWeighted(bgr, 0.55, hm, 0.45, 0))
    return tiles


def gambling_images(n=6):
    out = []
    for p in sorted(JU_IMG.glob("*.jpg")):
        lf = JU_LBL / (p.stem + ".txt")
        if lf.exists() and lf.read_text().strip() and lf.read_text().split()[0] == "0":
            out.append(p)
        if len(out) >= n:
            break
    return out


def main():
    # 1) v2-1 confusion matrix, big fonts
    m = YOLO(str(V21_W))
    r = m.val(data=str(V21_D), split="val", device="cpu", plots=False, verbose=False)
    plot_cm(r.confusion_matrix.matrix, V21_CLS, FIG / "confusion_matrix.png")

    # 2) judi3k confusion matrix, big fonts
    mj = YOLO(str(JU_W))
    rj = mj.val(data=str(JU_D), split="val", device="cpu", plots=False, verbose=False)
    plot_cm(rj.confusion_matrix.matrix, JU_CLS, FIG / "confusion_matrix_judi3k.png")

    # 3) judi3k detections
    imgs = gambling_images(6)
    pr = mj.predict(source=[str(p) for p in imgs], conf=0.25, device="cpu", save=True,
                    project=str(FIG / "_pred_judi"), name="se", exist_ok=True, line_width=4)
    saved = sorted(Path(pr[0].save_dir).glob("*.jpg"))[:6]
    montage([cv2.imread(str(s)) for s in saved], FIG / "detections_judi3k.png", cols=3)

    # 4) judi3k Grad-CAM
    montage(gradcam_tiles(JU_W, imgs), FIG / "gradcam_judi3k.png", cols=3)
    print("done")


if __name__ == "__main__":
    main()
