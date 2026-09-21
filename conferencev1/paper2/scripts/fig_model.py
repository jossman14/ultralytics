"""Proposed-model figures: confusion matrix (big font), detections, Grad-CAM.
Grad-CAM targets the layers that feed the Detect head, so it is architecture-agnostic.
CPU. Real data."""
import shutil
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
FIG = REPO / "conferencev1/paper2/figures"
PROP = "yolov8-spd-shuffle-simam"
W = RUNS / f"{PROP}_v2-1_f0/weights/best.pt"
DATA = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/data.yaml"
IMGDIR = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/valid/images"
CLS = ["BK8", "Gate-of-olympus", "Princess", "Starlight-Princess", "Zeus"]
IMGSZ = 640
lb = LetterBox((IMGSZ, IMGSZ))


def plot_cm(matrix, names, out, big=17):
    m = matrix.astype(float); m = m / (m.sum(0, keepdims=True) + 1e-9)
    labels = names + ["background"]; n = len(labels)
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


def gradcam(weights, images):
    net = YOLO(str(weights)).model.eval()
    for p in net.parameters():
        p.requires_grad_(True)
    det = net.model[-1]
    idx = det.f if isinstance(det.f, (list, tuple)) else [det.f]  # layers feeding Detect
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
        out = net(x); pred = out[0] if isinstance(out, (list, tuple)) else out
        net.zero_grad(); pred[:, 4:, :].max().backward()
        agg = np.zeros((IMGSZ, IMGSZ), np.float32)
        for k in range(len(targets)):
            A, G = acts[k], grads[k]
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * A).sum(1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, (IMGSZ, IMGSZ), mode="bilinear", align_corners=False)
            c = cam[0, 0].detach().numpy(); agg += (c - c.min()) / (c.max() - c.min() + 1e-8)
        for h in hooks:
            h.remove()
        agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)
        hm = cv2.applyColorMap(np.uint8(255 * agg), cv2.COLORMAP_JET)
        tiles.append(cv2.addWeighted(bgr, 0.55, hm, 0.45, 0))
    return tiles


def main():
    m = YOLO(str(W))
    r = m.val(data=str(DATA), split="val", device="cpu", plots=True,
              project=str(FIG / "_v"), name="p", exist_ok=True, verbose=False)
    plot_cm(r.confusion_matrix.matrix, CLS, FIG / "cm_proposed.png")
    imgs = sorted(IMGDIR.glob("*.jpg"))[:6]
    pr = m.predict(source=[str(p) for p in imgs], conf=0.25, device="cpu", save=True,
                   project=str(FIG / "_pred"), name="p", exist_ok=True, line_width=3)
    saved = sorted(Path(pr[0].save_dir).glob("*.jpg"))[:6]
    montage([cv2.imread(str(s)) for s in saved], FIG / "detections_proposed.png", cols=3)
    montage(gradcam(W, imgs), FIG / "gradcam_proposed.png", cols=3)
    print("done")


if __name__ == "__main__":
    main()
