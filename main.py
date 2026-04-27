import argparse
import os
import shutil
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from sklearn.manifold import TSNE
from scipy import stats
from scopus_analytics_core import AdvancedMetrics, StatisticalSuite, XAIAnalyzer
from adapters.yolo_adapter import YOLOv8Adapter

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 K-Fold Training and Analysis")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base YOLO model to use (e.g., yolov8n.pt, yolov8s.pt, yolov26s.pt)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size for training")
    return parser.parse_args()

def save_to_excel(results_csv_path, output_excel_path):
    """
    Simpan hasil training dari results.csv ke format Excel,
    dibagi menjadi sheet Training_Loss dan Testing_Metrics.
    """
    if not os.path.exists(results_csv_path):
        print(f"File {results_csv_path} not found.")
        return
    
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip() # Strip whitespace dari nama kolom
    
    # Kolom training dan val sesuai standar ultralytics/yolov8
    train_cols = [c for c in df.columns if 'train' in c or 'lr' in c or 'epoch' in c]
    val_cols = [c for c in df.columns if 'val' in c or 'metrics' in c or 'epoch' in c]
    
    df_train = df[train_cols] if train_cols else df
    df_val = df[val_cols] if val_cols else df
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Results', index=False)
        df_train.to_excel(writer, sheet_name='Training_Loss', index=False)
        df_val.to_excel(writer, sheet_name='Testing_Metrics', index=False)
    print(f"Saved results to {output_excel_path}")

class EigenCAM:
    """Custom hook untuk generate pseudo-GradCAM/EigenCAM map."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.hook = self.target_layer.register_forward_hook(self.forward_hook)
        
    def forward_hook(self, module, input, output):
        self.activations.append(output.cpu().detach())
        
    def __call__(self, img_tensor):
        self.activations = []
        with torch.no_grad():
            self.model(img_tensor)
        
        if not self.activations:
            return None
            
        activation = self.activations[0] # Bentuk: [1, C, H, W]
        # Menggunakan mean antar channel (pseudo EigenCAM)
        heatmap = torch.mean(activation, dim=1).squeeze().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap
        
    def release(self):
        self.hook.remove()

def run_gradcam(weights_path, val_images_dir, output_dir, num_samples=5):
    """Jalankan analisis GradCAM pada layer sebelum detection head."""
    model = YOLO(weights_path)
    
    # Target layer: biasanya layer[-2] dalam YOLOv8 architecture (sebelum Detect head)
    try:
        target_layer = model.model.model[-2]
    except Exception as e:
        print("Could not find target layer for GradCAM:", e)
        return

    cam = EigenCAM(model.model, target_layer)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(val_images_dir):
        print(f"Validation dir not found: {val_images_dir}")
        return
        
    img_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    for img_name in img_files:
        img_path = os.path.join(val_images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = F.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.to(model.device)
        
        heatmap = cam(img_tensor)
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_img = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            cam_img = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
            
            cv2.imwrite(os.path.join(output_dir, f"gradcam_{img_name}"), cam_img)
            
    cam.release()
    print(f"GradCAM analysis saved to {output_dir}")

def run_tsne(weights_path, val_images_dir, output_dir, num_samples=100):
    """Extract features and plot t-SNE."""
    model = YOLO(weights_path)
    
    try:
        target_layer = model.model.model[-2]
    except Exception as e:
        print("Could not find target layer for t-SNE:", e)
        return
        
    features_list = []
    def hook_fn(module, input, output):
        # Global Average Pooling [1, C]
        feat = torch.mean(output, dim=[2, 3]).cpu().detach().numpy()
        features_list.append(feat.flatten())
        
    hook = target_layer.register_forward_hook(hook_fn)
    
    if not os.path.exists(val_images_dir):
        return
        
    img_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    if len(img_files) < 2:
        return
        
    for img_name in img_files:
        img_path = os.path.join(val_images_dir, img_name)
        model.predict(img_path, imgsz=640, verbose=False)
        
    hook.remove()
    
    if len(features_list) == 0:
        return
        
    X = np.stack(features_list, axis=0) # [N, C]
    
    # Run t-SNE
    perplexity = min(30, max(2, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, c='blue', edgecolors='k')
    plt.title(f"t-SNE Projection (Layer Before Detect Head, N={len(X)})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tsne_plot.png"), dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {output_dir}")

def statistical_analysis(dataset_outputs_dir):
    """
    Jalankan analisis statistik: mean, std, T-Test, Wilcoxon.
    """
    mAP50 = []
    mAP50_95 = []
    folds = []
    
    for fold in range(5):
        fold_dir = os.path.join(dataset_outputs_dir, f"fold_{fold}")
        results_csv = os.path.join(fold_dir, "results.csv")
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            if 'metrics/mAP50(B)' in df.columns:
                mAP50.append(df['metrics/mAP50(B)'].iloc[-1])
                mAP50_95.append(df['metrics/mAP50-95(B)'].iloc[-1])
                folds.append(fold)
                
    if len(mAP50) == 0:
        print(f"No results found for statistical analysis in {dataset_outputs_dir}")
        return
        
    stats_dict = {
        'Fold': folds + ['Mean', 'Std Dev'],
        'mAP50': mAP50 + [np.mean(mAP50), np.std(mAP50)],
        'mAP50-95': mAP50_95 + [np.mean(mAP50_95), np.std(mAP50_95)]
    }
    
    df_stats = pd.DataFrame(stats_dict)
    
    test_results = []
    if len(mAP50) >= 3:
        # Shapiro-Wilk Test
        stat_s, p_s = stats.shapiro(mAP50)
        test_results.append({'Test': 'Shapiro-Wilk Normality', 'Metric': 'mAP50', 'Statistic': stat_s, 'p-value': p_s})
        
        # 1-Sample T-Test (baseline 0.5)
        stat_t, p_t = stats.ttest_1samp(mAP50, popmean=0.5)
        test_results.append({'Test': '1-Sample T-Test (vs 0.5 baseline)', 'Metric': 'mAP50', 'Statistic': stat_t, 'p-value': p_t})
        
        # Wilcoxon Signed-Rank
        try:
            stat_w, p_w = stats.wilcoxon([m - 0.5 for m in mAP50])
            test_results.append({'Test': 'Wilcoxon Signed-Rank (vs 0.5 baseline)', 'Metric': 'mAP50', 'Statistic': stat_w, 'p-value': p_w})
        except:
            pass
            
    df_tests = pd.DataFrame(test_results)
    
    report_path = os.path.join(dataset_outputs_dir, "statistical_analysis.xlsx")
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df_stats.to_excel(writer, sheet_name='Metrics_per_Fold', index=False)
        if not df_tests.empty:
            df_tests.to_excel(writer, sheet_name='Statistical_Tests', index=False)
            
    print(f"Statistical analysis saved to {report_path}")

def main():
    args = parse_args()
    
    datasets = [
        "Judol-Detection-v2-1_5fold",
        "dataset_judi_online_yolo_5fold"
    ]
    
    models_to_train = ["yolov8n.pt", "yolov8s.pt"] # Can add baseline/modifications here
    
    base_dir = os.path.abspath(".")
    
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset} not found! Skipping...")
            continue
            
        dataset_output_dir = os.path.join(base_dir, "output", dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        for model_weights in models_to_train:
            model_name = model_weights.replace(".pt", "")
            model_output_dir = os.path.join(dataset_output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            
            for fold in range(5):
                fold_name = f"fold_{fold}"
                fold_dir = os.path.join(dataset_path, fold_name)
                data_yaml = os.path.join(fold_dir, "data.yaml")
                
                if not os.path.exists(data_yaml):
                    continue
                    
                print(f"\nTraining {dataset} - {model_name} - {fold_name}")
                
                # Training Logic (simplified for brevity, reuse current)
                os.chdir(fold_dir)
                try:
                    model = YOLO(model_weights)
                    results = model.train(
                        data="data.yaml",
                        epochs=args.epochs,
                        imgsz=args.imgsz,
                        project=model_output_dir,
                        name=fold_name,
                        exist_ok=True
                    )
                finally:
                    os.chdir(base_dir)
                
                # Advanced Analysis per fold
                weights_path = os.path.join(model_output_dir, fold_name, "weights", "best.pt")
                if os.path.exists(weights_path):
                    adapter = YOLOv8Adapter(weights_path)
                    # Run Tier 1: Bootstrapping & Sens/Spec
                    metrics, cm = adapter.get_validation_metrics(data_yaml)
                    sens, spec = AdvancedMetrics.calculate_sens_spec(cm)
                    
                    # Run Tier 2: Heatmap IoU
                    val_images_dir = os.path.join(fold_dir, "valid", "images")
                    if os.path.exists(val_images_dir):
                        img_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png'))][:5]
                        iou_scores = []
                        for img_name in img_files:
                            img_path = os.path.join(val_images_dir, img_name)
                            heatmap = adapter.generate_eigencam(img_path)
                            if heatmap is not None:
                                _, bboxes = adapter.get_predictions(img_path)
                                iou = XAIAnalyzer.calculate_heatmap_iou(heatmap, bboxes)
                                iou_scores.append(iou)
                        
                        if iou_scores:
                            print(f"Mean Heatmap IoU for {fold_name}: {np.mean(iou_scores):.4f}")

        # Final Statistical Analysis for Dataset across ALL models
        print(f"\nRunning Global Statistical Analysis for {dataset}...")
        # Use sys.executable to ensure we use the same python environment
        cmd = f'"{sys.executable}" scopus_analytics_core.py --dir "{dataset_output_dir}" --metric mAP50'
        os.system(cmd)

if __name__ == "__main__":
    main()
