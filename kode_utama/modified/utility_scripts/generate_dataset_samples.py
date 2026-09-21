"""
Generate Dataset Samples Visualization

Creates a 4-column composite image for each dataset:
- Column 1: Class Name (White Background, Black Text)
- Columns 2-4: 3 random samples from the dataset for that class
"""

import os
import glob
import random
import logging
import yaml
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_H = 480
LABEL_W = 400

def find_latest_results(dataset_name):
    root = Path("/home/ftib/ultralytics/judol/result")
    pattern = str(root / f"results_{dataset_name}_*")
    matches = glob.glob(pattern)
    if not matches: return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return Path(matches[0])

def generate_samples():
    datasets = ["dataset_judi_online_yolo_5fold", "Judol-Detection-v2-1_5fold"]
    
    for ds_name in datasets:
        logger.info(f"Processing {ds_name}...")
        res_dir = find_latest_results(ds_name)
        if not res_dir:
            logger.warning(f"  No results found for {ds_name}")
            continue
            
        rekap_dir = res_dir / "rekap"
        rekap_dir.mkdir(parents=True, exist_ok=True)
        
        # We can use fold_0 as the source of images
        data_yaml_path = Path("/home/ftib/ultralytics/judol") / ds_name / "fold_0" / "data.yaml"
        if not data_yaml_path.exists():
            logger.warning(f"  data.yaml not found: {data_yaml_path}")
            continue
            
        with open(data_yaml_path) as f:
            cfg = yaml.safe_load(f)
        names = cfg['names']
        
        # Gather images per class from training/valid split
        # We'll check both train and valid to have a wider pool
        img_dirs = [
            Path("/home/ftib/ultralytics/judol") / ds_name / "fold_0" / "train" / "images",
            Path("/home/ftib/ultralytics/judol") / ds_name / "fold_0" / "valid" / "images"
        ]
        
        class_samples = {n: [] for n in names}
        
        for img_dir in img_dirs:
            if not img_dir.exists(): continue
            lbl_dir = img_dir.parent / "labels"
            
            for img_p in img_dir.glob("*"):
                if img_p.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                if not lbl_p.exists(): continue
                
                with open(lbl_p) as f:
                    lines = f.readlines()
                    if not lines: continue
                    # Take the first class mentioned in the label file for categorization
                    cid = int(lines[0].split()[0])
                    if cid < len(names):
                        class_samples[names[cid]].append(img_p)
        
        # Build Rows
        rows_list = []
        max_row_w = 0
        
        for cls_name in names:
            samples = class_samples[cls_name]
            if not samples:
                logger.warning(f"  No samples found for class: {cls_name}")
                continue
                
            # Randomly pick 3
            sample_count = min(3, len(samples))
            pick = random.sample(samples, sample_count)
            
            # Load and resize images
            resized_imgs = []
            row_img_w = 0
            for s_p in pick:
                img = cv2.imread(str(s_p))
                if img is None: continue
                ih, iw = img.shape[:2]
                sw = int(iw * (TARGET_H / ih))
                resized = cv2.resize(img, (sw, TARGET_H))
                resized_imgs.append(resized)
                row_img_w += sw
            
            if not resized_imgs: continue
            
            # Create row canvas
            # Pad with white if less than 3 samples
            # We'll align columns properly in a later step, for now just stack
            row_stack = np.hstack(resized_imgs)
            
            # Add label area
            actual_row_w = LABEL_W + row_stack.shape[1]
            row_canvas = np.ones((TARGET_H, actual_row_w, 3), dtype=np.uint8) * 255
            row_canvas[:, LABEL_W:] = row_stack
            
            # Draw Class Name
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
            (tw, th), _ = cv2.getTextSize(cls_name, font, scale, thick)
            cv2.putText(row_canvas, cls_name, (20, TARGET_H // 2 + th // 2), font, scale, (0, 0, 0), thick)
            
            rows_list.append(row_canvas)
            max_row_w = max(max_row_w, actual_row_w)
            
        if not rows_list: continue
        
        # Standardize row widths
        standardized_rows = []
        for row in rows_list:
            if row.shape[1] < max_row_w:
                padded = np.ones((TARGET_H, max_row_w, 3), dtype=np.uint8) * 255
                padded[:, :row.shape[1]] = row
                standardized_rows.append(padded)
            else:
                standardized_rows.append(row)
                
        # Final Stacking
        final_h = TARGET_H * len(standardized_rows)
        final_img = np.vstack(standardized_rows)
        
        # Add white lines between rows
        for i in range(1, len(standardized_rows)):
            cv2.line(final_img, (0, i * TARGET_H), (max_row_w, i * TARGET_H), (200, 200, 200), 2)
            
        output_path = rekap_dir / "dataset_samples.jpg"
        cv2.imwrite(str(output_path), final_img)
        logger.info(f"  ✅ Saved dataset samples: {output_path}")

if __name__ == "__main__":
    generate_samples()
