import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import matplotlib.image as mpimg

# Set global plot parameters for better readability
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'lines.linewidth': 3
})

RESULT_DIRS = {
    "dataset_judi_online_yolo_5fold": Path("/home/ftib/ultralytics/judol/result/results_dataset_judi_online_yolo_5fold_12_03_2026_12_55_wib"),
    "Judol-Detection-v2-1_5fold": Path("/home/ftib/ultralytics/judol/result/results_Judol-Detection-v2-1_5fold_12_03_2026_11_42_wib"),
}

def generate_rekap_grafik(ds_name, root_dir):
    rekap_dir = root_dir / "rekap" / "grafik"
    rekap_dir.mkdir(parents=True, exist_ok=True)
    
    folds = range(5)
    colors = sns.color_palette("husl", 5)
    
    # 1. Training & Validation Losses (Box, Cls, DFL)
    # 3 rows (Box, Cls, DFL) x 2 columns (Train, Val)
    fig_loss, axes_loss = plt.subplots(3, 2, figsize=(20, 18))
    
    # metrics/mAP50(B) and metrics/mAP50-95(B)
    fig_metrics, axes_metrics = plt.subplots(2, 1, figsize=(16, 14))
    
    # Curves from XLSX
    fig_f1, ax_f1 = plt.subplots(figsize=(14, 10))
    fig_pr, ax_pr = plt.subplots(figsize=(12, 12))
    
    for i, fold in enumerate(folds):
        fold_dir = root_dir / f"fold_{fold}"
        if not fold_dir.exists(): continue
        
        # --- Process results.csv ---
        res_csv = fold_dir / "results.csv"
        if res_csv.exists():
            df = pd.read_csv(res_csv)
            df.columns = [c.strip() for c in df.columns]
            
            # Loss Plots
            axes_loss[0, 0].plot(df['epoch'], df['train/box_loss'], label=f'Fold {fold}', color=colors[i])
            axes_loss[0, 1].plot(df['epoch'], df['val/box_loss'], label=f'Fold {fold}', color=colors[i])
            axes_loss[1, 0].plot(df['epoch'], df['train/cls_loss'], label=f'Fold {fold}', color=colors[i])
            axes_loss[1, 1].plot(df['epoch'], df['val/cls_loss'], label=f'Fold {fold}', color=colors[i])
            axes_loss[2, 0].plot(df['epoch'], df['train/dfl_loss'], label=f'Fold {fold}', color=colors[i])
            axes_loss[2, 1].plot(df['epoch'], df['val/dfl_loss'], label=f'Fold {fold}', color=colors[i])
            
            # Metric Plots
            axes_metrics[0].plot(df['epoch'], df['metrics/mAP50(B)'], label=f'Fold {fold}', color=colors[i])
            axes_metrics[1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label=f'Fold {fold}', color=colors[i])
            
        # --- Process Curves (XLSX) ---
        f1_xlsx = fold_dir / "BoxF1_curve.xlsx"
        if f1_xlsx.exists():
            df_f1 = pd.read_excel(f1_xlsx)
            ax_f1.plot(df_f1['Confidence'], df_f1['mean_raw'], label=f'Fold {fold}', color=colors[i])
            
        pr_xlsx = fold_dir / "BoxPR_curve.xlsx"
        if pr_xlsx.exists():
            df_pr = pd.read_excel(pr_xlsx)
            ax_pr.plot(df_pr['Recall'], df_pr['mean_precision'], label=f'Fold {fold}', color=colors[i])
            
    # Finalize Loss Plots
    titles = ["Train Box Loss", "Val Box Loss", "Train Class Loss", "Val Class Loss", "Train DFL Loss", "Val DFL Loss"]
    for idx, ax in enumerate(axes_loss.flatten()):
        ax.set_title(titles[idx])
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig_loss.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_loss.savefig(rekap_dir / "rekap_loss_curves.png", dpi=300)
    
    # Finalize Metrics Plots
    axes_metrics[0].set_title("mAP@50")
    axes_metrics[1].set_title("mAP@50-95")
    for ax in axes_metrics:
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig_metrics.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_metrics.savefig(rekap_dir / "rekap_map_metrics.png", dpi=300)
    
    # Finalize F1 Plot
    ax_f1.set_title(f"F1-Confidence Curve")
    ax_f1.set_xlabel("Confidence")
    ax_f1.set_ylabel("F1")
    ax_f1.set_xlim(0, 1)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)
    fig_f1.savefig(rekap_dir / "rekap_f1_curve.png", dpi=300)
    
    # Finalize PR Plot
    ax_pr.set_title(f"Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend()
    ax_pr.grid(True, alpha=0.3)
    fig_pr.savefig(rekap_dir / "rekap_pr_curve.png", dpi=300)
    
    plt.close('all')
    print(f"Generated base rekap grafik for {ds_name}")

def create_composite_rekap(ds_name, root_dir):
    rekap_dir = root_dir / "rekap" / "grafik"
    
    img_paths = {
        "Average Confusion Matrix": rekap_dir / "rekap_confusion_matrix_average.png",
        "PR Curve": rekap_dir / "rekap_pr_curve.png",
        "F1 Curve": rekap_dir / "rekap_f1_curve.png",
        "mAP Metrics": rekap_dir / "rekap_map_metrics.png"
    }
    
    fig = plt.figure(figsize=(24, 20))
    
    for i, (caption, path) in enumerate(img_paths.items()):
        if not path.exists():
            print(f"  ⚠ Image missing for composite: {path}")
            continue
        
        ax = fig.add_subplot(2, 2, i+1)
        img = mpimg.imread(str(path))
        ax.imshow(img)
        ax.set_title(caption, fontsize=30, fontweight='bold', pad=15)
        ax.axis('off')
        
    plt.tight_layout()
    save_path = rekap_dir / "rekap_composite_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created composite: {save_path}")

if __name__ == "__main__":
    for name, path in RESULT_DIRS.items():
        if path.exists():
            generate_rekap_grafik(name, path)
            create_composite_rekap(name, path)
        else:
            print(f"Path not found: {path}")
