"""
Grad-CAM Visualization for YOLOv12 5-Fold Cross Validation

Generates class-aware activation heatmaps for each detected object across 
multiple datasets and their 5 folds.
"""

import os
import glob
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ultralytics import YOLO

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
JUDOL_DIR = SCRIPT_DIR / "judol"
RESULT_DIR = JUDOL_DIR / "result"

DATASETS = [
    "dataset_judi_online_yolo_5fold",
    "Judol-Detection-v2-1_5fold"
]
NUM_FOLDS = 5

SAMPLES_PER_CLASS = 3
IMG_SIZE = 640


# ==============================================================================
# YOLO GRAD-CAM (Activation-based, Multi-scale)
# ==============================================================================

class YOLOGradCAM:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load two separate model instances:
        # 1) For detection
        self.det_model = YOLO(model_path)
        
        # 2) For CAM
        self.cam_model = YOLO(model_path)
        self.cam_model.model.to(self.device)
        self.cam_model.model.eval()
        
        # YOLO12s neck layers
        self.target_layer_indices = [14, 17, 20]
        self.activations: Dict[int, torch.Tensor] = {}
        
        for idx in self.target_layer_indices:
            layer = self.cam_model.model.model[idx]
            layer.register_forward_hook(self._make_hook(idx))
            
    def _make_hook(self, idx):
        def hook(module, input, output):
            self.activations[idx] = output.detach()
        return hook

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def run_detection(self, image_path: str) -> list:
        results = self.det_model.predict(
            image_path, imgsz=IMG_SIZE, verbose=False, device=self.device
        )
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'cls_id': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                })
        return detections

    def generate_cam_for_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        h_orig, w_orig = image.shape[:2]
        
        self.activations.clear()
        img_tensor = self._preprocess(image)
        with torch.no_grad():
            _ = self.cam_model.model(img_tensor)
        
        cams = []
        layer_weights = [0.25, 0.35, 0.40]
        
        for i, idx in enumerate(self.target_layer_indices):
            if idx not in self.activations:
                continue
            
            act = self.activations[idx]
            _, C, H_feat, W_feat = act.shape
            
            scale_x = W_feat / IMG_SIZE
            scale_y = H_feat / IMG_SIZE
            
            fx1 = max(0, int(bbox[0] * scale_x))
            fy1 = max(0, int(bbox[1] * scale_y))
            fx2 = min(W_feat, int(bbox[2] * scale_x) + 1)
            fy2 = min(H_feat, int(bbox[3] * scale_y) + 1)
            
            if fx2 <= fx1 or fy2 <= fy1:
                fx1, fy1 = 0, 0
                fx2, fy2 = W_feat, H_feat
            
            bbox_region = act[0, :, fy1:fy2, fx1:fx2]
            channel_weights = bbox_region.mean(dim=[1, 2])
            channel_weights = F.relu(channel_weights)
            
            if channel_weights.sum() > 0:
                channel_weights = channel_weights / channel_weights.sum()
            
            cam = (channel_weights.view(C, 1, 1) * act[0]).sum(dim=0)
            cam = F.relu(cam)
            
            cam = cam.cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            cam_resized = cv2.resize(cam, (w_orig, h_orig))
            cams.append(cam_resized * layer_weights[i])
        
        if not cams:
            return np.zeros((h_orig, w_orig), dtype=np.float32)
        
        combined = sum(cams)
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined.astype(np.float32)

    def process_image(self, image: np.ndarray, image_path: str, class_names: list) -> Tuple[np.ndarray, np.ndarray, list]:
        h, w = image.shape[:2]
        detections = self.run_detection(image_path)
        det_image = self._draw_boxes(image.copy(), detections, class_names, h, w)
        
        if not detections:
            return image.copy(), det_image, detections
        
        combined_cam = np.zeros((h, w), dtype=np.float32)
        
        for det in detections:
            bbox = det['bbox']
            cam = self.generate_cam_for_bbox(image, bbox)
            
            sx, sy = w / IMG_SIZE, h / IMG_SIZE
            bbox_orig = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
            mask = self._soft_bbox_mask(h, w, bbox_orig, padding=0.25)
            
            masked_cam = cam * mask
            combined_cam = np.maximum(combined_cam, masked_cam)
        
        if combined_cam.max() > 0:
            combined_cam = combined_cam / combined_cam.max()
        
        overlay = self._create_overlay(image, combined_cam, detections, class_names, h, w)
        return overlay, det_image, detections

    def _create_overlay(self, image: np.ndarray, cam: np.ndarray, detections: list, class_names: list, h: int, w: int) -> np.ndarray:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        alpha = 0.6
        cam_3ch = np.stack([cam, cam, cam], axis=-1)
        threshold = 0.15
        blend_mask = (cam_3ch > threshold).astype(np.float32)
        
        overlay = (
            image.astype(np.float32) * (1 - alpha * blend_mask) +
            heatmap.astype(np.float32) * alpha * blend_mask
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        overlay = self._draw_boxes(overlay, detections, class_names, h, w)
        return overlay

    def _soft_bbox_mask(self, h: int, w: int, bbox: List[float], padding: float = 0.25) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * padding, bh * padding
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        
        mask = np.zeros((h, w), dtype=np.float32)
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return mask
        mask[iy1:iy2, ix1:ix2] = 1.0
        
        ksize = max(int(min(bw, bh) * 0.4), 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        if mask.max() > 0:
            mask = mask / mask.max()
        return mask

    def _draw_boxes(self, image: np.ndarray, detections: list, class_names: list, h: int, w: int, thickness: int = 2) -> np.ndarray:
        sx, sy = w / IMG_SIZE, h / IMG_SIZE
        colors = [(0, 140, 255), (0, 255, 100), (255, 80, 80), (220, 40, 255), (40, 220, 255)]
        
        for det in detections:
            bbox = det['bbox']
            cls_id = det['cls_id']
            conf = det['conf']
            
            x1, y1, x2, y2 = int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy)
            color = colors[cls_id % len(colors)]
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            label = f"{cls_name} {conf:.2f}"
            (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th_text - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(image, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

# ==============================================================================
# HELPERS
# ==============================================================================

def find_latest_experiment_dir(dataset_name: str) -> Optional[Path]:
    """Find the most recent results_{dataset_name}_{timestamp} directory."""
    pattern = str(RESULT_DIR / f"results_{dataset_name}_*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(matches[0])


def get_images_by_class(data_yaml: Path, split: str = "valid") -> dict:
    if not data_yaml.exists():
        return {}
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get('names', [])
    nc = data_cfg.get('nc', len(names))
    dataset_dir = data_yaml.parent
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"
    if not img_dir.exists():
        return {}
        
    class_images = {name: [] for name in names}
    for img_file in sorted(img_dir.iterdir()):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            continue
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        if not lbl_file.exists():
            continue
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < nc:
                        class_name = names[cls_id]
                        if str(img_file) not in class_images[class_name]:
                            class_images[class_name].append(str(img_file))
    return class_images


# ==============================================================================
# COMPOSITE VISUALIZATION
# ==============================================================================

def generate_composite(results: list, class_names: list, output_dir: Path, title_prefix: str) -> None:
    by_class = {}
    for item in results:
        cls = item['class']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(item)

    # Per-class summaries
    for cls_name in class_names:
        if cls_name not in by_class:
            continue
        items = by_class[cls_name]
        n = len(items)
        
        fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n), squeeze=False)
        for i, item in enumerate(items):
            axes[i, 0].imshow(cv2.cvtColor(item['original'], cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(cv2.cvtColor(item['overlay'], cv2.COLOR_BGR2RGB))
            dets = item.get('detections', [])
            det_info = ", ".join(f"{class_names[d['cls_id']]}({d['conf']:.2f})" for d in dets if d['cls_id'] < len(class_names)) or "No detections"
            axes[i, 1].set_title(f'Grad-CAM: {det_info}', fontsize=11, fontweight='bold')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(cv2.cvtColor(item['detection'], cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title('Detection Result', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'{title_prefix} — {cls_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        cls_path = output_dir / f"gradcam_{cls_name.lower().replace('-', '_')}.png"
        plt.savefig(str(cls_path), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"    ✓ {cls_path.name}")

    # Full composite
    n_classes = len(by_class)
    max_samples = max(len(v) for v in by_class.values()) if by_class else 1
    
    fig, axes = plt.subplots(n_classes, max_samples * 3, figsize=(6 * max_samples * 3, 6 * n_classes), squeeze=False)
    row = 0
    for cls_name in class_names:
        if cls_name not in by_class:
            continue
        items = by_class[cls_name]
        for col, item in enumerate(items):
            bc = col * 3
            axes[row, bc].imshow(cv2.cvtColor(item['original'], cv2.COLOR_BGR2RGB))
            if col == 0:
                axes[row, bc].set_ylabel(cls_name, fontsize=13, fontweight='bold')
            axes[row, bc].set_title('Original', fontsize=10)
            axes[row, bc].axis('off')
            
            axes[row, bc+1].imshow(cv2.cvtColor(item['overlay'], cv2.COLOR_BGR2RGB))
            axes[row, bc+1].set_title('Grad-CAM', fontsize=10)
            axes[row, bc+1].axis('off')
            
            axes[row, bc+2].imshow(cv2.cvtColor(item['detection'], cv2.COLOR_BGR2RGB))
            axes[row, bc+2].set_title('Detection', fontsize=10)
            axes[row, bc+2].axis('off')
            
        for col in range(len(items) * 3, max_samples * 3):
            axes[row, col].axis('off')
        row += 1

    plt.suptitle(f'{title_prefix} Analysis', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    comp_path = output_dir / "gradcam_composite.png"
    plt.savefig(str(comp_path), dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    logger.info(f"    ✓ {comp_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def process_dataset(dataset_name: str):
    logger.info("=" * 60)
    logger.info(f"🔬 GRAD-CAM ANALYSIS: {dataset_name}")
    logger.info("=" * 60)

    experiment_dir = find_latest_experiment_dir(dataset_name)
    if not experiment_dir:
        logger.warning(f"⚠️ No results found for dataset '{dataset_name}'. Skipping.")
        return

    logger.info(f"📁 Target Experiment: {experiment_dir}")

    total_images = 0

    for fold in range(NUM_FOLDS):
        logger.info(f"\n{'-' * 40}")
        fold_dir = experiment_dir / f"fold_{fold}"
        if not fold_dir.exists():
            logger.warning(f"⚠️ Fold {fold} directory not found.")
            continue
            
        weights_path = fold_dir / "weights" / "best.pt"
        if not weights_path.exists():
            logger.warning(f"⚠️ best.pt not found in fold {fold}.")
            continue

        gradcam_dir = fold_dir / "gradcam"
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        
        # Source dataset for this fold
        data_yaml = JUDOL_DIR / dataset_name / f"fold_{fold}" / "data.yaml"
        if not data_yaml.exists():
            logger.warning(f"⚠️ data.yaml not found: {data_yaml}")
            continue

        with open(data_yaml) as f:
            data_cfg = yaml.safe_load(f)
        class_names = data_cfg.get('names', [])
        
        logger.info(f"🎯 Fold {fold} | Weights: {weights_path.name} | Classes: {class_names}")

        gradcam = YOLOGradCAM(str(weights_path))

        class_images = get_images_by_class(data_yaml, split="valid")
        all_results = []

        for cls_name in class_names:
            imgs = class_images.get(cls_name, [])
            if not imgs:
                continue

            n = min(SAMPLES_PER_CLASS, len(imgs))
            sampled = random.sample(imgs, n)
            logger.info(f"  • {cls_name}: Processing {n} samples")

            cls_dir = gradcam_dir / cls_name
            cls_dir.mkdir(exist_ok=True)

            for img_path in sampled:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                basename = Path(img_path).stem
                overlay, det_img, detections = gradcam.process_image(img, img_path, class_names)

                cv2.imwrite(str(cls_dir / f"{basename}_original.jpg"), img)
                cv2.imwrite(str(cls_dir / f"{basename}_gradcam.jpg"), overlay)
                cv2.imwrite(str(cls_dir / f"{basename}_detection.jpg"), det_img)

                all_results.append({
                    'class': cls_name,
                    'original': img,
                    'overlay': overlay,
                    'detection': det_img,
                    'detections': detections,
                    'path': img_path,
                })

        if MATPLOTLIB_AVAILABLE and all_results:
            generate_composite(all_results, class_names, gradcam_dir, f"YOLOv12s {dataset_name} Fold_{fold}")
            
        total_images += len(all_results)
        logger.info(f"✅ Fold {fold} complete: {len(all_results)} images processed.")
        
    logger.info(f"\n🏁 Analysis Complete for {dataset_name}: {total_images} total images.")


def main():
    logger.info("=" * 60)
    logger.info("YOLOv12s MULTI-DATASET 5-FOLD GRAD-CAM BATCH RUNNER")
    logger.info("=" * 60)
    
    for dataset in DATASETS:
        process_dataset(dataset)

if __name__ == "__main__":
    main()
