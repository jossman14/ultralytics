"""
Grad-CAM Comparison Report (Refined Visualization)

Generates 3-column composite images for each class:
1. Original Image + Ground Truth Boxes
2. Heatmap Image (Activation-based)
3. Overlay Image + Predicted Boxes + Confidence
"""

import os
import glob
import random
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO

# reuse YOLOGradCAM if possible, but let's just implement the necessary parts here 
# to keep this script self-contained and flexible for the new layout.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 640
ALPHA = 0.6
THRESHOLD = 0.15
PADDING = 0.25

class YOLOGradCAM:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.det_model = YOLO(model_path)
        self.cam_model = YOLO(model_path)
        self.cam_model.model.to(self.device)
        self.cam_model.model.eval()
        self.target_layer_indices = [14, 17, 20] # FPN neck layers
        self.activations = {}
        
        for idx in self.target_layer_indices:
            layer = self.cam_model.model.model[idx]
            layer.register_forward_hook(self._make_hook(idx))
            
    def _make_hook(self, idx):
        def hook(module, input, output):
            self.activations[idx] = output.detach()
        return hook

    def generate_heatmap(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        h_orig, w_orig = image.shape[:2]
        self.activations.clear()
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _ = self.cam_model.model(img_tensor)
            
        cams = []
        layer_weights = [0.25, 0.35, 0.40]
        
        for i, idx in enumerate(self.target_layer_indices):
            if idx not in self.activations: continue
            act = self.activations[idx]
            _, C, H, W = act.shape
            
            # Map bbox to feature map resolution
            sx, sy = W / IMG_SIZE, H / IMG_SIZE
            fx1, fy1 = max(0, int(bbox[0]*sx)), max(0, int(bbox[1]*sy))
            fx2, fy2 = min(W, int(bbox[2]*sx)+1), min(H, int(bbox[3]*sy)+1)
            
            if fx2 <= fx1 or fy2 <= fy1:
                region = act[0]
            else:
                region = act[0, :, fy1:fy2, fx1:fx2]
                
            weights = region.mean(dim=[1, 2])
            weights = F.relu(weights)
            if weights.sum() > 0: weights /= weights.sum()
            
            cam = (weights.view(C, 1, 1) * act[0]).sum(dim=0)
            cam = F.relu(cam).cpu().numpy()
            if cam.max() > 0: cam /= cam.max()
            
            cams.append(cv2.resize(cam, (w_orig, h_orig)) * layer_weights[i])
            
        if not cams: return np.zeros((h_orig, w_orig), dtype=np.float32)
        combined = sum(cams)
        if combined.max() > 0: combined /= combined.max()
        
        # Apply soft mask
        mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        # Scaled bbox to original image size
        scx, scy = w_orig / IMG_SIZE, h_orig / IMG_SIZE
        x1, y1, x2, y2 = bbox[0]*scx, bbox[1]*scy, bbox[2]*scx, bbox[3]*scy
        
        bw, bh = x2-x1, y2-y1
        px, py = bw * PADDING, bh * PADDING
        mx1, my1 = max(0, int(x1-px)), max(0, int(y1-py))
        mx2, my2 = min(w_orig, int(x2+px)), min(h_orig, int(y2+py))
        mask[my1:my2, mx1:mx2] = 1.0
        
        k = max(int(min(bw, bh) * 0.4), 5)
        if k % 2 == 0: k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        if mask.max() > 0: mask /= mask.max()
        
        final = combined * mask
        if final.max() > 0: final /= final.max()
        return final

def draw_info(img, text, bbox, color, is_top=True):
    h = img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Adaptive scale & thickness based on image height
    scale = max(0.6, h / 800.0)
    thick = max(2, int(h / 300.0))
    
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw Red Box (BGR)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thick)
    
    if is_top:
        ty = y1 - 10 if y1 - 10 > th else y1 + th + 10
    else:
        ty = min(img.shape[0], y2 + th + 10)
        
    # Semi-transparent yellowish-white background for text
    tx1, ty1 = max(0, x1), max(0, ty - th - 5)
    tx2, ty2 = min(img.shape[1], x1 + tw + 5), min(img.shape[0], ty + 5)
    
    if tx2 > tx1 and ty2 > ty1:
        overlay = img.copy()
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (180, 255, 255), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
    # Draw black text for contrast
    cv2.putText(img, text, (x1, int(ty)), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img

def find_latest_results(dataset_name):
    root = Path("/home/ftib/ultralytics/judol/result")
    pattern = str(root / f"results_{dataset_name}_*")
    matches = glob.glob(pattern)
    if not matches: return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return Path(matches[0])

def process():
    datasets = ["dataset_judi_online_yolo_5fold", "Judol-Detection-v2-1_5fold"]
    
    for ds_name in datasets:
        logger.info(f"Processing {ds_name}...")
        res_dir = find_latest_results(ds_name)
        if not res_dir: continue
        
        rekap_gradcam = res_dir / "rekap" / "gradcam"
        rekap_gradcam.mkdir(parents=True, exist_ok=True)
        
        # Use Fold 0 as representative
        fold_dir = res_dir / "fold_0"
        weights = fold_dir / "weights" / "best.pt"
        data_yaml_path = Path("/home/ftib/ultralytics/judol") / ds_name / "fold_0" / "data.yaml"
        
        if not weights.exists() or not data_yaml_path.exists():
            logger.warning(f"  Missing files in {fold_dir}")
            continue
            
        with open(data_yaml_path) as f:
            cfg = yaml.safe_load(f)
        names = cfg['names']
        
        cam_agent = YOLOGradCAM(str(weights))
        
        # Get samples from validation set
        val_img_dir = Path("/home/ftib/ultralytics/judol") / ds_name / "fold_0" / "valid" / "images"
        val_lbl_dir = Path("/home/ftib/ultralytics/judol") / ds_name / "fold_0" / "valid" / "labels"
        
        class_samples = {n: [] for n in names}
        for img_p in val_img_dir.glob("*"):
            if img_p.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
            lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists(): continue
            with open(lbl_p) as f:
                content = f.read().strip()
                if not content: continue
                # Pick first class in image for sampling
                cid = int(content.split('\n')[0].split()[0])
                if len(class_samples[names[cid]]) < 1: # One good sample per class
                    class_samples[names[cid]].append((img_p, lbl_p))
                    
        # Randomize selection
        for n in names:
            random.shuffle(class_samples[n])
            
        # Determine target size based on first valid image or dataset characteristics
        # We find max width/height among selected samples to avoid distortion
        selected_samples = []
        for cls_name in names:
            samples = class_samples.get(cls_name, [])
            if not samples: continue
            img_p, lbl_p = samples[0]
            selected_samples.append((cls_name, img_p, lbl_p))
            
        if not selected_samples: continue

        # Determine maximum required image width across all samples to ensure grid alignment
        max_img_w = 0
        target_h = 640
        for cls_name, img_p, lbl_p in selected_samples:
            img_temp = cv2.imread(str(img_p))
            if img_temp is not None:
                h_t, w_t = img_temp.shape[:2]
                calc_w = int(w_t * (target_h / h_t))
                max_img_w = max(max_img_w, calc_w)

        rows_list = []
        for cls_name, img_p, lbl_p in selected_samples:
            logger.info(f"  Generating for {cls_name}: {img_p.name}")
            
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2]
            
            # Prediction for CAM
            results = cam_agent.det_model.predict(str(img_p), imgsz=IMG_SIZE, verbose=False)[0]
            
            best_det = None
            for box in results.boxes:
                if int(box.cls[0]) == names.index(cls_name):
                    if best_det is None or box.conf[0] > best_det.conf[0]:
                        best_det = box
            
            if best_det is None:
                if len(results.boxes) > 0:
                    best_det = results.boxes[0]
                else:
                    logger.warning(f"    No detections found in {img_p.name}")
                    continue
            
            pred_bbox = best_det.xyxy[0].cpu().numpy()
            det_bbox_640 = best_det.xyxyn[0].cpu().numpy() * IMG_SIZE
            heatmap = cam_agent.generate_heatmap(img, det_bbox_640)
            
            # Col 1: Original + GT (Green)
            col1 = img.copy()
            with open(lbl_p) as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    gx1, gy1, gx2, gy2 = (xc - bw/2) * w, (yc - bh/2) * h, (xc + bw/2) * w, (yc + bh/2) * h
                    col1 = draw_info(col1, f"GT: {names[cid]}", [gx1, gy1, gx2, gy2], (0, 0, 255))
            
            # Col 2: Heatmap
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            col2 = heatmap_color
            
            # Col 3: Overlay + Pred (Orange)
            alpha = 0.6
            cam_3ch = np.stack([heatmap, heatmap, heatmap], axis=-1)
            blend_mask = (cam_3ch > THRESHOLD).astype(np.float32)
            overlay = (img.astype(np.float32) * (1 - alpha * blend_mask) + 
                       heatmap_color.astype(np.float32) * alpha * blend_mask)
            col3 = np.clip(overlay, 0, 255).astype(np.uint8)
            
            px1, py1, px2, py2 = pred_bbox
            conf, p_name = float(best_det.conf[0]), names[int(best_det.cls[0])]
            col3 = draw_info(col3, f"Pred: {p_name} {conf:.2f}", [px1, py1, px2, py2], (0, 0, 255), is_top=False)
            
            # Create standardized columns by centering images in fixed boxes
            def get_standard_col(img_data):
                ih, iw = img_data.shape[:2]
                sc_w = int(iw * (target_h / ih))
                img_res = cv2.resize(img_data, (sc_w, target_h))
                
                # Canvas for one column
                canvas = np.ones((target_h, max_img_w, 3), dtype=np.uint8) * 255
                # Center horizontally
                start_x = (max_img_w - sc_w) // 2
                canvas[:, start_x:start_x+sc_w] = img_res
                return canvas

            c1_std = get_standard_col(col1)
            c2_std = get_standard_col(col2)
            c3_std = get_standard_col(col3)
            
            row_stack = np.hstack([c1_std, c2_std, c3_std])
            
            # Label area on the left (White)
            label_w = 300
            row_canvas = np.ones((target_h, max_img_w * 3 + label_w, 3), dtype=np.uint8) * 255
            row_canvas[:, label_w:] = row_stack
            
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
            (tw, th), _ = cv2.getTextSize(cls_name, font, scale, thick)
            cv2.putText(row_canvas, cls_name, (20, target_h // 2 + th // 2), font, scale, (0, 0, 0), thick)
            
            rows_list.append(row_canvas)
            
        if not rows_list: continue

        # Final Assembly (all rows are now naturally the same width)
        max_row_w = rows_list[0].shape[1]
        row_h = rows_list[0].shape[0]
        header_h = 120
        total_h = header_h + row_h * len(rows_list)
        final_canvas = np.ones((total_h, max_row_w, 3), dtype=np.uint8) * 255
        
        # Headers (aligned with centers of col boxes)
        headers = ["Original Image", "Heatmap", "Results"]
        for i, txt in enumerate(headers):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
            # Center of i-th image box: 300 + i*max_img_w + max_img_w/2
            tx = 300 + i * max_img_w + (max_img_w - tw) // 2
            cv2.putText(final_canvas, txt, (tx, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
            
        # Paste rows
        for i, row in enumerate(rows_list):
            final_canvas[header_h + i*row_h : header_h + (i+1)*row_h, :] = row
            # Horizontal separator
            if i < len(rows_list) - 1:
                cv2.line(final_canvas, (0, header_h + (i+1)*row_h), (max_row_w, header_h + (i+1)*row_h), (200, 200, 200), 2)
            
        out_p = rekap_gradcam / "rekap_gradcam_comparison.jpg"
        cv2.imwrite(str(out_p), final_canvas)
        logger.info(f"  ✅ Saved master composite: {out_p}")

if __name__ == "__main__":
    process()
