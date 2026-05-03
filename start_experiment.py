from main import run_full_pipeline
from prepare_kfold import prepare_kfold
import os
from datetime import datetime, timedelta, timezone

# ==============================================================================
# 0. DATASET PREPARATION (K-FOLD)
# ==============================================================================
PREPARE_KFOLD = True  
DATASET_MAPPINGS = [
    {"source": "test_dataset", "output": "hehe"},
]

# ==============================================================================
# 1. KONFIGURASI LENGKAP (HYPERPARAMETERS) - TEST MODE (1 Epoch, 64 Imgsz)
# ==============================================================================
CONFIG = {
    "epochs": 1,            # Test mode: 1 epoch
    "imgsz": 64,            # Test mode: 64 imgsz
    "batch": 16,
    "patience": 50,
    "optimizer": "auto",
    "device": 0,
    "workers": 0,
    "cache": "disk",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "val": True,
    "save": True,
    "exist_ok": True,
}

# ==============================================================================
# 2. DAFTAR MODEL
# ==============================================================================
MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
]

# ==============================================================================
# 3. DAFTAR DATASET (TARGET TRAINING)
# ==============================================================================
# Menggabungkan dataset hasil konversi DAN dataset yang sudah ada di folder
DATASETS = []
if PREPARE_KFOLD:
    DATASETS.extend([m["output"] for m in DATASET_MAPPINGS])

# Tambahkan dataset yang sudah berbentuk kfold manual di sini
MANUAL_KFOLD_DATASETS = ["Judol-Detection-v2-1_5fold"]
DATASETS.extend(MANUAL_KFOLD_DATASETS)

# ==============================================================================
# MAIN RUNNER WITH TIMESTAMP
# ==============================================================================
if __name__ == "__main__":
    jakarta_now = datetime.now(timezone(timedelta(hours=7)))
    day = jakarta_now.day
    ts_suffix = jakarta_now.strftime(f"{day}%b%Y_%HH%MM%SS")
    
    print("\n" + "="*50)
    print("      ADVANCED ANALYTICS: AUTOMATED RUNNER")
    print(f"      START TIME: {jakarta_now.strftime('%Y-%m-%d %H:%M:%S')} (WIB)")
    print("="*50)

    # Langkah 1: K-Fold Preparation (Multi-Dataset)
    if PREPARE_KFOLD:
        print(f"\n[STEP 1] Mempersiapkan {len(DATASET_MAPPINGS)} Stratified K-Fold Datasets...")
        for mapping in DATASET_MAPPINGS:
            src, out = mapping["source"], mapping["output"]
            print(f"\n>>> Processing: {src} -> {out}")
            if os.path.exists(out):
                print(f"    (Folder {out} sudah ada, menimpa...)")
            prepare_kfold(src, out, n_splits=5)
        print(f"\n[SUCCESS] Seluruh Persiapan Dataset Selesai.")

    # Langkah 2: Training & Analysis Pipeline
    print(f"\n[STEP 2] Memulai Pipeline Eksperimen...")
    print(f"Total Datasets : {len(DATASETS)} ({DATASETS})")
    print(f"Total Models   : {len(MODELS)} ({MODELS})")
    print("="*50 + "\n")
    
    for dataset in DATASETS:
        for model in MODELS:
            print(f"\n>>> [RUNNING] Dataset: {dataset} | Model: {model} | Suffix: {ts_suffix}")
            run_full_pipeline(
                model_weights=model,
                dataset_name=dataset,
                output_suffix=ts_suffix,
                **CONFIG
            )
    
    print("\n" + "="*50)
    print(f" [SUCCESS] Eksperimen Selesai. Suffix: {ts_suffix}")
    print("="*50)
