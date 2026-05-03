import os
import sys
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from ultralytics import YOLO
from sklearn.manifold import TSNE
import torchvision.transforms as T
from PIL import Image
from analytics_core import AdvancedMetrics, XAIAnalyzer, ErrorAnalysis
from adapters.yolo_adapter import YOLOv8Adapter
import seaborn as sns
from scipy import stats

def normalized_to_pixel(bbox_norm, img_w, img_h):
    """Convert YOLO [cls, cx, cy, w, h] normalized to [cls, x1, y1, x2, y2] pixel coordinates."""
    cls, cx, cy, w, h = bbox_norm
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [int(cls), x1, y1, x2, y2]

def draw_text_with_bg(img, text, pos, font_scale=1.0, color=(255, 255, 255), thickness=2, bg_color=(0, 0, 0)):
    """Draw text with a solid background for better readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - h - 5), (x + w + 5, y + baseline + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, img_tensor):
        # Forward pass with gradients enabled
        # Cloning the tensor to avoid "Inference tensors cannot be saved for backward"
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                img_t = img_tensor.clone()
                img_t.requires_grad = True
                preds = self.model(img_t)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                    
                # Extract scores (YOLOv8: [batch, 4 + num_classes, anchors])
                scores = preds[0, 4:, :]
                max_score, _ = torch.max(scores.flatten(), dim=0)
                
                # Backward pass
                self.model.zero_grad()
                max_score.backward(retain_graph=True)
        
        # Grad-CAM Calculation: weights = mean(gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and Normalization
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam.detach().cpu().numpy()
        if np.max(grad_cam) > 0:
            grad_cam /= np.max(grad_cam)
        return grad_cam
        
    def release(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def run_gradcam(weights_path, val_images_dir, output_dir, num_samples=5):
    model = YOLO(weights_path)
    try:
        target_layer = model.model.model[-2]
    except Exception as e:
        print("Could not find target layer:", e)
        return

    cam = GradCAM(model.model, target_layer)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(val_images_dir):
        print(f"Validation dir not found: {val_images_dir}")
        return
        
    img_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    for img_name in img_files:
        img_path = os.path.join(val_images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = F.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.to(model.device)
        
        # Enable gradients for the pass
        heatmap = cam(img_tensor)
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_img = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            cam_img = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
            cv2.imwrite(os.path.join(output_dir, f"gradcam_{img_name}"), cam_img)
    cam.release()

def run_tsne(weights_path, val_images_dir, output_dir, num_samples=50):
    model = YOLO(weights_path)
    try:
        target_layer = model.model.model[-2]
    except Exception as e:
        print("Could not find target layer:", e)
        return
        
    features_list = []
    def hook_fn(module, input, output):
        feat = torch.mean(output, dim=[2, 3]).cpu().detach().numpy()
        features_list.append(feat.flatten())
        
    hook = target_layer.register_forward_hook(hook_fn)
    if not os.path.exists(val_images_dir): return
    img_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    for img_name in img_files:
        model.predict(os.path.join(val_images_dir, img_name), imgsz=640, verbose=False)
    hook.remove()
    
    if len(features_list) < 2: return
    X = np.stack(features_list, axis=0)
    perplexity = min(30, max(2, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, c='red', edgecolors='k')
    plt.title(f"t-SNE Projection ({os.path.basename(weights_path)})")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tsne_plot.png"), dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {output_dir}")

def plot_error_composition(error_data, output_path):
    """Plot Stacked Bar Chart of Error Composition per Fold."""
    df = pd.DataFrame(error_data).T
    # Normalize to 100%
    df_percent = df.div(df.sum(axis=1), axis=0) * 100
    
    colors = ['#4CAF50', '#FFC107', '#FF5722', '#9C27B0', '#795548'] # Correct, Loc, Cls, Bkg, Missed
    ax = df_percent.plot(kind='bar', stacked=True, figsize=(12, 7), color=colors)
    
    plt.title("Error Composition per Fold (%)", fontsize=15)
    plt.xlabel("Fold", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_iou_histogram(iou_list, output_path):
    """Plot IoU distribution for all detections."""
    if not iou_list: return
    plt.figure(figsize=(10, 6))
    sns.histplot(iou_list, bins=20, kde=True, color='skyblue')
    plt.axvline(np.mean(iou_list), color='red', linestyle='--', label=f'Mean: {np.mean(iou_list):.4f}')
    plt.title("IoU Distribution for True Positives", fontsize=15)
    plt.xlabel("IoU Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_error_gallery(error_samples, output_dir):
    """Generate individual error images and a 3-row montage."""
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    for error_type, img in error_samples.items():
        if img is not None:
            # Resize to a fixed size for the gallery
            img_resized = cv2.resize(img, (640, 640))
            # Save individual
            cv2.imwrite(os.path.join(output_dir, f"{error_type}.jpg"), img)
            
            # Add label for montage with background
            labeled_img = img_resized.copy()
            draw_text_with_bg(labeled_img, error_type, (20, 50), font_scale=1.4, color=(255, 255, 255), thickness=3, bg_color=(0, 0, 200))
            # Add Legend
            draw_text_with_bg(labeled_img, "GT: GREEN | PRED: RED", (20, 620), font_scale=0.8, color=(255, 255, 255), thickness=2, bg_color=(50, 50, 50))
            images.append(labeled_img)
            
    if len(images) >= 3:
        # Create montage (3 rows)
        row1 = np.hstack(images[:len(images)//3]) if len(images) >= 3 else images[0]
        # For simplicity, if we have enough samples, we stack them. 
        # User asked for 3 rows.
        rows = []
        n = len(images)
        per_row = (n + 2) // 3
        for i in range(0, n, per_row):
            row = np.hstack(images[i:i+per_row])
            rows.append(row)
        
        # Ensure all rows have same width
        max_w = max(r.shape[1] for r in rows)
        padded_rows = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded_rows.append(r)
            
        montage = np.vstack(padded_rows)
        cv2.imwrite(os.path.join(output_dir, "error_gallery_montage.jpg"), montage)

def statistical_analysis(model_output_dir):
    """
    Jalankan analisis statistik lengkap: mean, std, T-Test, Wilcoxon untuk semua metrik, 
    termasuk F1, Waktu Komputasi, GFLOPs, AOPC, dan Error Breakdown.
    """
    metrics_data = {
        'mAP50': [], 'mAP50-95': [], 'Precision': [], 'Recall': [],
        'F1': [], 'TrainingTime': [], 'GFLOPs': []
    }
    folds = []
    error_composition_data = {}
    
    col_map = {
        'metrics/mAP50(B)': 'mAP50',
        'metrics/mAP50-95(B)': 'mAP50-95',
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall'
    }

    for fold_name in sorted(os.listdir(model_output_dir)):
        if not fold_name.startswith("fold_"): continue
        fold_dir = os.path.join(model_output_dir, fold_name)
        results_csv = os.path.join(fold_dir, "results.csv")
        weights_path = os.path.join(fold_dir, "weights", "best.pt")
        
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            valid_fold = False
            for csv_col, clean_name in col_map.items():
                if csv_col in df.columns:
                    metrics_data[clean_name].append(df[csv_col].iloc[-1])
                    valid_fold = True
            
            if valid_fold:
                folds.append(fold_name)
                # F1
                p, r = metrics_data['Precision'][-1], metrics_data['Recall'][-1]
                metrics_data['F1'].append(2 * (p * r) / (p + r + 1e-8))
                # Training Time
                metrics_data['TrainingTime'].append(df['time'].sum() if 'time' in df.columns else 0)
                # GFLOPs
                if os.path.exists(weights_path):
                    try:
                        m_tmp = YOLO(weights_path)
                        _, _, _, flops = m_tmp.info()
                        metrics_data['GFLOPs'].append(flops)
                    except: metrics_data['GFLOPs'].append(0)
                else: metrics_data['GFLOPs'].append(0)

                # Extended Metrics
                ext_csv = os.path.join(fold_dir, "analysis", "extended_metrics.csv")
                if os.path.exists(ext_csv):
                    ext_df = pd.read_csv(ext_csv)
                    for col in ext_df.columns:
                        if col not in metrics_data: metrics_data[col] = [0] * (len(folds)-1)
                        metrics_data[col].append(ext_df[col].iloc[0])
                    
                    error_cols = ['Correct', 'LocError', 'ClsError', 'BkgError', 'MissedObj']
                    if all(c in ext_df.columns for c in error_cols):
                        error_composition_data[fold_name] = {c: ext_df[c].iloc[0] for c in error_cols}
                
                # Align lengths
                for k in metrics_data:
                    if len(metrics_data[k]) < len(folds): metrics_data[k].append(0)

    if not folds: return

    # Aggregating Per-Class Metrics
    per_class_data = []
    for fold_name in folds:
        pc_csv = os.path.join(model_output_dir, fold_name, "analysis", "per_class_metrics.csv")
        if os.path.exists(pc_csv):
            df_pc = pd.read_csv(pc_csv)
            if 'Unnamed: 0' in df_pc.columns:
                df_pc = df_pc.rename(columns={'Unnamed: 0': 'Class'})
            df_pc['Fold'] = fold_name
            per_class_data.append(df_pc)
            
    df_per_class_all = pd.concat(per_class_data) if per_class_data else pd.DataFrame()
    df_per_class_summary = df_per_class_all.groupby('Class').mean(numeric_only=True).reset_index() if not df_per_class_all.empty else pd.DataFrame()

    # Visualizations
    if error_composition_data:
        plot_error_composition(error_composition_data, os.path.join(model_output_dir, "error_composition_overall.png"))

    # Stats Helper
    def calculate_cohen_d(x, mu0=0.5):
        std = np.std(x, ddof=1)
        return (np.mean(x) - mu0) / std if std > 0 else 0

    # Stats Calculation
    df_detailed = pd.DataFrame({'Fold': folds, **{k: v for k, v in metrics_data.items() if len(v) == len(folds)}})
    summary_rows = []
    for m in df_detailed.columns:
        if m == 'Fold': continue
        vals = pd.to_numeric(df_detailed[m], errors='coerce').dropna().values
        if len(vals) > 0:
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            sem = std / np.sqrt(len(vals))
            ci95 = 1.96 * sem
            cv = (std / mean) * 100 if mean > 0 else 0
            skew = stats.skew(vals)
            kurt = stats.kurtosis(vals)
            
            summary_rows.append({
                'Metric': m, 
                'Mean': mean, 
                'Std Dev': std,
                'SEM': sem,
                'CI 95% Low': mean - ci95,
                'CI 95% High': mean + ci95,
                'CV (%)': cv,
                'Skewness': skew,
                'Kurtosis': kurt,
                'Cohen d': calculate_cohen_d(vals)
            })
    df_summary = pd.DataFrame(summary_rows)
    
    # Tests
    test_rows = []
    if len(folds) >= 3:
        for m in df_detailed.columns:
            if m in ['Fold', 'GFLOPs'] or np.std(df_detailed[m]) == 0: continue
            try:
                vals = pd.to_numeric(df_detailed[m], errors='coerce').dropna().values
                s_stat, s_p = stats.shapiro(vals)
                test_rows.append({'Metric': m, 'Test': 'Shapiro-Wilk', 'p-value': s_p})
                if m in ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']:
                    t_stat, t_p = stats.ttest_1samp(vals, 0.5)
                    test_rows.append({'Metric': m, 'Test': 'T-Test (vs 0.5)', 'p-value': t_p})
            except: pass
    df_tests = pd.DataFrame(test_rows)

    # Export
    report_path = os.path.join(model_output_dir, "statistical_analysis.xlsx")
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
        df_detailed.to_excel(writer, sheet_name='Metrics_per_Fold', index=False)
        if error_composition_data:
            pd.DataFrame(error_composition_data).T.to_excel(writer, sheet_name='Error_Composition')
        if not df_per_class_summary.empty:
            df_per_class_summary.to_excel(writer, sheet_name='Per_Class_Summary', index=False)
            df_per_class_all.to_excel(writer, sheet_name='Per_Class_Detailed', index=False)
        if not df_tests.empty:
            df_tests.to_excel(writer, sheet_name='Statistical_Tests', index=False)
            
    # Generate Analytics Summary Image
    kpi_plot_path = os.path.join(model_output_dir, "analytics_summary_kpi.png")
    try:
        plt.figure(figsize=(12, 6))
        kpi_metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1', 'AOPC']
        main_metrics = df_summary[df_summary['Metric'].isin(kpi_metrics)]
        if not main_metrics.empty:
            sns.barplot(x='Metric', y='Mean', data=main_metrics, palette='viridis')
            plt.title("Key Performance Indicators (Mean across folds)")
            plt.ylim(0, 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(kpi_plot_path, dpi=300)
        plt.close()
    except: pass

    # --- COLLECT TO analytics_summary FOLDER ---
    analytics_dir = os.path.join(model_output_dir, "analytics_summary")
    os.makedirs(analytics_dir, exist_ok=True)
    
    # 1. Copy Excel
    shutil.copy(report_path, os.path.join(analytics_dir, "statistical_analysis.xlsx"))
    
    # 2. Move KPI Plot
    if os.path.exists(kpi_plot_path):
        shutil.move(kpi_plot_path, os.path.join(analytics_dir, "kpi_summary.png"))
        
    # 3. Move Error Composition Plot
    err_comp_path = os.path.join(model_output_dir, "error_composition_overall.png")
    if os.path.exists(err_comp_path):
        shutil.move(err_comp_path, os.path.join(analytics_dir, "error_composition_overall.png"))
        
    # 4. Copy Best Fold Gallery
    try:
        if not df_detailed.empty and 'mAP50' in df_detailed.columns:
            best_fold = df_detailed.sort_values('mAP50', ascending=False)['Fold'].iloc[0]
            best_gallery = os.path.join(model_output_dir, best_fold, "analysis", "error_gallery", "error_gallery_montage.jpg")
            if os.path.exists(best_gallery):
                shutil.copy(best_gallery, os.path.join(analytics_dir, "representative_error_gallery.jpg"))
    except: pass
    
    # 5. Create Executive Summary TXT
    try:
        with open(os.path.join(analytics_dir, "executive_summary.txt"), "w") as f:
            f.write("=== EXECUTIVE SUMMARY (ADVANCED ANALYTICS) ===\n")
            f.write(f"Model: {os.path.basename(model_output_dir)}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Metrics Summary (Mean ± Std [95% CI]):\n")
            f.write("-" * 50 + "\n")
            
            # List of metrics to look for (common YOLO and custom names)
            target_metrics = [
                'mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1', 'AOPC', 
                'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)'
            ]
            
            for _, row in df_summary.iterrows():
                m_name = str(row['Metric'])
                # Check for exact match or fuzzy match
                if any(tm.lower() in m_name.lower() for tm in target_metrics):
                    f.write(f"{m_name:25}: {row['Mean']:.4f} ± {row['Std Dev']:.4f} [{row['CI 95% Low']:.4f}, {row['CI 95% High']:.4f}]\n")
            
            if not df_tests.empty:
                f.write("\nStatistical Significance (vs 0.5):\n")
                f.write("-" * 50 + "\n")
                for _, row in df_tests.iterrows():
                    m_name = str(row['Metric'])
                    if 'T-Test' in str(row['Test']):
                        sig = " (Significant)" if row['p-value'] < 0.05 else " (Not Significant)"
                        f.write(f"{m_name:25}: p={row['p-value']:.4f}{sig}\n")
    except Exception as e:
        print(f"Error creating executive summary: {e}")
    
    print(f"Comprehensive Analytics Report saved to {analytics_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Existing YOLO Training Results")
    parser.add_argument("--dir", type=str, required=True, help="Path to model output directory (e.g., output/dataset/yolov8s)")
    parser.add_argument("--dataset_root", type=str, default="Judol-Detection-v2-1_5fold", help="Root of the dataset folds")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Directory not found: {args.dir}")
        return

    # dataset_root is used to find validation images
    dataset_path = os.path.abspath(args.dataset_root)

    for fold_name in os.listdir(args.dir):
        if not fold_name.startswith("fold_"): continue
        fold_dir = os.path.join(args.dir, fold_name)
        weights_path = os.path.join(fold_dir, "weights", "best.pt")
        
        if os.path.exists(weights_path):
            print(f"\n--- Analyzing {fold_name} ---")
            analysis_dir = os.path.join(fold_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Find validation images for this fold
            val_images_dir = os.path.join(dataset_path, fold_name, "valid", "images")
            data_yaml = os.path.join(dataset_path, fold_name, "data.yaml")
            
            if not os.path.exists(val_images_dir):
                print(f"Warning: Validation images not found at {val_images_dir}")
                continue

            # 1. GradCAM & t-SNE
            print("Generating GradCAM and t-SNE...")
            run_gradcam(weights_path, val_images_dir, analysis_dir)
            run_tsne(weights_path, val_images_dir, analysis_dir)

            # 2. Metrics (Sens/Spec)
            print("Calculating Sensitivity/Specificity...")
            adapter = YOLOv8Adapter(weights_path)
            if os.path.exists(data_yaml):
                metrics, cm, per_class = adapter.get_validation_metrics(data_yaml)
                sens, spec = AdvancedMetrics.calculate_sens_spec(cm)
                
                # Save per-class metrics
                df_pc = pd.DataFrame(per_class).T
                df_pc.to_csv(os.path.join(analysis_dir, "per_class_metrics.csv"))
                
                # --- NEW: Error Breakdown Analysis ---
                print(f"Running Robust Error Breakdown for {fold_name}...")
                val_labels_dir = os.path.join(dataset_path, fold_name, "valid", "labels")
                
                fold_errors = {'Correct': 0, 'LocError': 0, 'ClsError': 0, 'BkgError': 0, 'MissedObj': 0}
                fold_ious = []
                error_samples = {k: None for k in fold_errors.keys()}
                
                if os.path.exists(val_labels_dir):
                    label_files = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
                    for lbl_file in label_files:
                        img_file = lbl_file.replace('.txt', '.jpg')
                        if not os.path.exists(os.path.join(val_images_dir, img_file)):
                            img_file = lbl_file.replace('.txt', '.png')
                        
                        full_img_path = os.path.join(val_images_dir, img_file)
                        if os.path.exists(full_img_path):
                            # Load Image to get dimensions
                            img_cv = cv2.imread(full_img_path)
                            if img_cv is None: continue
                            h, w, _ = img_cv.shape
                            
                            # Load GT
                            gt = []
                            with open(os.path.join(val_labels_dir, lbl_file), 'r') as f:
                                for line in f:
                                    parts = list(map(float, line.split()))
                                    gt.append(normalized_to_pixel(parts, w, h))
                            
                            # Get Preds
                            confs, bboxes, clss = adapter.get_predictions(full_img_path)
                            class_names = adapter.model.names
                            
                            # Combine for ErrorAnalysis: [cls, conf, x1, y1, x2, y2]
                            preds = []
                            for i in range(len(confs)):
                                preds.append([clss[i], confs[i], bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]])
                            
                            errs, ious, img_cls_errs = ErrorAnalysis.analyze_errors(gt, preds)
                            fold_ious.extend(ious)
                            for k in fold_errors: fold_errors[k] += errs[k]
                            
                            # Aggregate Per-Class Correct/Incorrect
                            for cls_id, counts in img_cls_errs.items():
                                cls_name = class_names[cls_id]
                                c_key = f"Class_{cls_name}_Correct"
                                i_key = f"Class_{cls_name}_Incorrect"
                                metrics[c_key] = metrics.get(c_key, 0) + counts['Correct']
                                metrics[i_key] = metrics.get(i_key, 0) + counts['Incorrect']
                            
                            # --- SAVE ALL IMAGES FOR DETAILED ANALYSIS ---
                            det_dir = os.path.join(analysis_dir, "all_detections")
                            os.makedirs(os.path.join(det_dir, "correct"), exist_ok=True)
                            os.makedirs(os.path.join(det_dir, "errors"), exist_ok=True)
                            
                            # Use the labeled image (drawing on a fresh copy to avoid gallery side-effects if needed)
                            labeled_full = img_cv.copy()
                            for p in preds:
                                cls_id, conf = int(p[0]), p[1]
                                x1, y1, x2, y2 = map(int, p[2:])
                                cv2.rectangle(labeled_full, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                label = f"P:{class_names[cls_id]} {conf:.2f}"
                                draw_text_with_bg(labeled_full, label, (x1, y1 - 5), font_scale=0.8, color=(0, 0, 0), thickness=2, bg_color=(255, 255, 255))
                            for g in gt:
                                cls_id = int(g[0])
                                x1, y1, x2, y2 = map(int, g[1:])
                                cv2.rectangle(labeled_full, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                label = f"GT:{class_names[cls_id]}"
                                draw_text_with_bg(labeled_full, label, (x1, y1 - 40 if y1 > 40 else y1 + 40), font_scale=0.8, color=(0, 0, 0), thickness=2, bg_color=(255, 255, 255))

                            has_error = (errs['LocError'] > 0 or errs['ClsError'] > 0 or errs['BkgError'] > 0 or errs['MissedObj'] > 0)
                            subfolder = "errors" if has_error else "correct"
                            cv2.imwrite(os.path.join(det_dir, subfolder, img_file), labeled_full)

                            # Collect sample for gallery if we don't have one for this error type
                            if any(errs.values()):
                                for k, v in errs.items():
                                    if v > 0 and error_samples[k] is None:
                                        error_samples[k] = labeled_full.copy()

                # --- NEW: AOPC XAI Metric ---
                print(f"Calculating AOPC (XAI Trustworthiness)...")
                aopc_scores = []
                img_files_xai = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png'))][:10]
                
                def model_predict_fn(img_tensor):
                    with torch.no_grad():
                        output = adapter.model.model(img_tensor)
                        if isinstance(output, (list, tuple)): output = output[0]
                        scores = output[0, 4:, :]
                        max_score, _ = torch.max(scores.flatten(), dim=0)
                        return float(max_score.cpu().numpy())

                for img_name in img_files_xai:
                    img_path = os.path.join(val_images_dir, img_name)
                    heatmap = adapter.generate_gradcam(img_path)
                    if heatmap is not None:
                        img_pil = Image.open(img_path).convert('RGB').resize((640, 640))
                        img_t = T.ToTensor()(img_pil).unsqueeze(0).to(next(adapter.model.model.parameters()).device)
                        aopc = XAIAnalyzer.calculate_aopc(model_predict_fn, img_t, heatmap, steps=10)
                        aopc_scores.append(aopc)

                # Save all to metrics
                metrics['AOPC'] = np.mean(aopc_scores) if aopc_scores else 0
                metrics.update(fold_errors)
                metrics['MeanIoU'] = np.mean(fold_ious) if fold_ious else 0

                # 3. Visualizations
                print(f"Generating Visualizations for {fold_name}...")
                plot_iou_histogram(fold_ious, os.path.join(analysis_dir, "iou_distribution.png"))
                generate_error_gallery(error_samples, os.path.join(analysis_dir, "error_gallery"))

                # Save to text file
                with open(os.path.join(analysis_dir, "metrics_report.txt"), "w") as f:
                    f.write(f"Fold: {fold_name}\n")
                    f.write("-" * 20 + "\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")
                    f.write(f"Sensitivity: {np.mean(sens):.4f}\n")
                    f.write(f"Specificity: {np.mean(spec):.4f}\n")
                
                # Save extended metrics for statistical summary
                pd.DataFrame([metrics]).to_csv(os.path.join(analysis_dir, "extended_metrics.csv"), index=False)

                # Save Heatmap IoU
                iou_scores = []
                for img_name in img_files_xai:
                    img_path = os.path.join(val_images_dir, img_name)
                    heatmap = adapter.generate_gradcam(img_path)
                    if heatmap is not None:
                        _, bboxes, _ = adapter.get_predictions(img_path)
                        iou = XAIAnalyzer.calculate_heatmap_iou(heatmap, bboxes)
                        iou_scores.append(iou)
                
                if iou_scores:
                    with open(os.path.join(analysis_dir, "metrics_report.txt"), "a") as f:
                        f.write(f"Mean Heatmap IoU: {np.mean(iou_scores):.4f}\n")
                    print(f"Mean Heatmap IoU: {np.mean(iou_scores):.4f}")

    print("\nAnalysis complete. Running final statistical summary...")
    statistical_analysis(args.dir)
    print(f"Results saved in {args.dir}")

if __name__ == "__main__":
    main()
