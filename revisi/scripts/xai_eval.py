"""Quantitative validation of the activation-CAM explanations, plus CAM pipeline cost.

Three things the reviewers asked for:
  1. Localization scores (pointing game, energy pointing game) instead of eyeballed heatmaps.
  2. Controls that make the scores meaningful: a random box of matched size, and a
     model-randomization sanity check. The paper's CAM weights channels inside the
     predicted box, so it can agree with the ground truth for trivial reasons; the
     box-free ("global") variant is prediction-independent and is scored alongside it.
  3. Detection latency separated from detection + CAM latency.

Only the multi-class dataset is scored. Every annotation in the binary dataset is
the full frame (0.5 0.5 1.0 1.0), so a localization metric against it is degenerate:
the box covers the image, every peak falls inside it, and the random-box control and
the randomized-model check score 1.0 alongside the real map. Those runs measure
nothing, so the binary sets are not a valid target for this evaluation.
"""
import argparse, json, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO

ROOT = Path("/home/ftib/ultralytics")
IMG_SIZE = 640
LAYERS = [14, 17, 20]
LAYER_W = [0.25, 0.35, 0.40]


class ActivationCAM:
    def __init__(self, weights, device):
        self.device = f"cuda:{device}" if str(device).isdigit() else device
        device = self.device
        self.model = YOLO(weights)
        self.model.model.to(device).eval()
        self.act = {}
        for idx in LAYERS:
            self.model.model.model[idx].register_forward_hook(self._hook(idx))

    def _hook(self, idx):
        def fn(_m, _i, out):
            self.act[idx] = out.detach()
        return fn

    def forward(self, tensor):
        self.act.clear()
        with torch.no_grad():
            self.model.model(tensor)

    def cam(self, box_xyxy, out_hw):
        """Fuse the three FPN scales. box_xyxy=None gives the prediction-independent variant."""
        cams = []
        for w, idx in zip(LAYER_W, LAYERS):
            a = self.act.get(idx)
            if a is None:
                continue
            _, C, H, W = a.shape
            if box_xyxy is None:
                region = a[0]
            else:
                sx, sy = W / IMG_SIZE, H / IMG_SIZE
                x1, y1 = max(0, int(box_xyxy[0] * sx)), max(0, int(box_xyxy[1] * sy))
                x2, y2 = min(W, int(box_xyxy[2] * sx) + 1), min(H, int(box_xyxy[3] * sy) + 1)
                region = a[0] if (x2 <= x1 or y2 <= y1) else a[0, :, y1:y2, x1:x2]
            wts = F.relu(region.mean(dim=[1, 2]))
            if wts.sum() > 0:
                wts = wts / wts.sum()
            c = F.relu((wts.view(C, 1, 1) * a[0]).sum(0)).cpu().numpy()
            if c.max() > 0:
                c = c / c.max()
            cams.append(cv2.resize(c, (out_hw[1], out_hw[0])) * w)
        if not cams:
            return np.zeros(out_hw, np.float32)
        c = sum(cams)
        return c / c.max() if c.max() > 0 else c


def load_gt(label_path, h, w):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def score(cam, box):
    """Pointing game hit and the fraction of CAM energy inside the box."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = cam.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    hit = int(y1 <= peak[0] < y2 and x1 <= peak[1] < x2)
    total = float(cam.sum())
    energy = float(cam[y1:y2, x1:x2].sum()) / total if total > 0 else 0.0
    area = (x2 - x1) * (y2 - y1) / float(h * w)
    return hit, energy, area


def random_box_like(box, h, w, rng):
    bw, bh = box[2] - box[0], box[3] - box[1]
    x = rng.uniform(0, max(1e-6, w - bw))
    y = rng.uniform(0, max(1e-6, h - bh))
    return [x, y, x + bw, y + bh]


def run_fold(weights, data_yaml, device, limit, rng, sanity=True):
    import yaml as _y
    device = f"cuda:{device}" if str(device).isdigit() else device
    cfg = _y.safe_load(Path(data_yaml).read_text())
    vdir = Path(data_yaml).parent / cfg["val"]
    imgs = sorted(vdir.glob("*.jpg"))[:limit]

    cam_agent = ActivationCAM(weights, device)
    rand_agent = None
    if sanity:
        rand_agent = ActivationCAM(weights, device)
        # Adebayo et al. randomization: re-initialize as an untrained network of the
        # same architecture. Normalization scales stay at 1 and biases at 0, the
        # standard init. Zeroing them instead would zero every BatchNorm output and
        # make the check vacuously pass.
        for name, p in rand_agent.model.model.model.named_parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")
            elif name.endswith("bias"):
                torch.nn.init.zeros_(p)
            else:
                torch.nn.init.ones_(p)

    acc = {k: [] for k in ("pg_box", "eb_box", "pg_glob", "eb_glob",
                           "pg_rand", "eb_rand", "pg_sanity", "eb_sanity", "area")}
    t_det, t_cam, n = 0.0, 0.0, 0

    # Warm up CUDA kernels and autocast caches so the first image does not carry
    # initialization cost into the reported latency.
    if imgs:
        warm = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        for _ in range(3):
            cam_agent.model.predict(str(imgs[0]), imgsz=IMG_SIZE, verbose=False, device=device)
            cam_agent.forward(warm)
        torch.cuda.synchronize()

    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(vdir.parent / "labels" / f"{ip.stem}.txt", h, w)
        if not gts:
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = cam_agent.model.predict(str(ip), imgsz=IMG_SIZE, verbose=False, device=device)[0]
        torch.cuda.synchronize()
        t_det += time.perf_counter() - t0

        if len(res.boxes) == 0:
            continue
        best = int(res.boxes.conf.argmax())
        pred = res.boxes.xyxy[best].cpu().numpy()
        # CAM input space is the letterbox-free square resize used by the original script
        pred_cam = [pred[0] * IMG_SIZE / w, pred[1] * IMG_SIZE / h,
                    pred[2] * IMG_SIZE / w, pred[3] * IMG_SIZE / h]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cam_agent.forward(t)
        c_box = cam_agent.cam(pred_cam, (h, w))
        torch.cuda.synchronize()
        t_cam += time.perf_counter() - t1

        c_glob = cam_agent.cam(None, (h, w))
        gt = max(gts, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        for key, cam in (("box", c_box), ("glob", c_glob)):
            s = score(cam, gt)
            if s:
                acc[f"pg_{key}"].append(s[0]); acc[f"eb_{key}"].append(s[1]); 
        s = score(c_glob, random_box_like(gt, h, w, rng))
        if s:
            acc["pg_rand"].append(s[0]); acc["eb_rand"].append(s[1])
        if rand_agent is not None:
            rand_agent.forward(t)
            s = score(rand_agent.cam(None, (h, w)), gt)
            if s:
                acc["pg_sanity"].append(s[0]); acc["eb_sanity"].append(s[1])
        s = score(c_glob, gt)
        if s:
            acc["area"].append(s[2])
        n += 1

    return dict(n=n,
                det_ms=1000 * t_det / max(n, 1),
                cam_ms=1000 * t_cam / max(n, 1),
                **{k: (float(np.mean(v)) if v else None) for k, v in acc.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "revisi/runs"))
    ap.add_argument("--model", default="yolo12s")
    ap.add_argument("--datasets", default="multiclass")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    folders = {"binary": "judol/dataset_judi_online_yolo_5fold",
               "multiclass": "judol/Judol-Detection-v2-1_5fold",
               "binary_videofold": "judol/dataset_judi_online_yolo_videofold"}
    rng = np.random.default_rng(0)
    out = {}
    for ds in a.datasets.split(","):
        rows = []
        for fold in range(5):
            w = Path(a.runs) / ds / a.model / f"fold_{fold}" / "weights" / "best.pt"
            y = ROOT / folders[ds] / f"fold_{fold}" / "data.yaml"
            if not w.exists():
                print(f"[miss] {w}"); continue
            r = run_fold(str(w), str(y), a.device, a.limit, rng)
            r["fold"] = fold
            rows.append(r)
            print(ds, fold, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
        out[ds] = rows
    (ROOT / "revisi/out/xai_eval.json").write_text(json.dumps(out, indent=1))
    print("saved revisi/out/xai_eval.json")
