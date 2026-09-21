"""Grad-CAM heatmaps for the trained YOLOv8-SE model on v2-1 images.

Gradient-based CAM: hook activations and gradients at a deep backbone layer,
backpropagate the maximum class score, and weight activations by mean gradient.
Overlays show which regions drive each gambling-logo detection.

Output: figures/gradcam.png (montage of overlays). Uses the real fold-0 SE weights.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

REPO = Path("/home/ftib/ultralytics")
FIG = REPO / "conferencev1/paper/figures"
SE_V21 = REPO / "judol_sweep/runs/yolov8-se-baseline_v2-1_f0/weights/best.pt"
V21 = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0/valid/images"
IMGSZ = 640
TARGET_IDX = [22, 23, 24]  # SE-recalibrated P3/P4/P5 outputs: multi-scale coverage

lb = LetterBox((IMGSZ, IMGSZ))


def cam_for(net, targets, bgr):
    """Aggregated Grad-CAM over several target layers, returned in [0,1]."""
    acts, grads, hooks = {}, {}, []
    for i, t in enumerate(targets):
        hooks.append(t.register_forward_hook(lambda m, inp, o, i=i: acts.__setitem__(i, o)))
        hooks.append(t.register_full_backward_hook(lambda m, gi, go, i=i: grads.__setitem__(i, go[0])))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    x.requires_grad_(True)
    out = net(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out  # [1, 4+nc, anchors]
    score = pred[:, 4:, :].max()  # strongest class response anywhere
    net.zero_grad()
    score.backward()
    agg = np.zeros((IMGSZ, IMGSZ), np.float32)
    for i in range(len(targets)):
        A, G = acts[i], grads[i]                     # [1,C,h,w]
        w = G.mean(dim=(2, 3), keepdim=True)         # gradient-weighted importance
        cam = torch.relu((w * A).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, (IMGSZ, IMGSZ), mode="bilinear", align_corners=False)
        c = cam[0, 0].detach().numpy()
        agg += (c - c.min()) / (c.max() - c.min() + 1e-8)  # normalize per scale, then sum
    for h in hooks:
        h.remove()
    return (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)


def overlay(bgr, cam):
    hm = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.55, hm, 0.45, 0)


def montage(tiles, out, cols=3, pad=6):
    rows = (len(tiles) + cols - 1) // cols
    ch, cw = tiles[0].shape[:2]
    canvas = np.full((rows * ch + (rows + 1) * pad, cols * cw + (cols + 1) * pad, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (ch + pad), pad + c * (cw + pad)
        canvas[y:y + ch, x:x + cw] = t
    cv2.imwrite(str(out), canvas)
    print("wrote", out)


def main():
    model = YOLO(str(SE_V21))
    net = model.model.eval()
    for p in net.parameters():
        p.requires_grad_(True)
    targets = [net.model[i] for i in TARGET_IDX]
    imgs = sorted(V21.glob("*.jpg"))[:6]
    tiles = []
    for ip in imgs:
        bgr = lb(image=cv2.imread(str(ip)))
        cam = cam_for(net, targets, bgr)
        tiles.append(overlay(bgr, cam))
    montage(tiles, FIG / "gradcam.png", cols=3)


if __name__ == "__main__":
    main()
