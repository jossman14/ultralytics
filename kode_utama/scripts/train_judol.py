"""
YOLOv12 5-Fold Cross Validation Training Script

Runs 5-Fold CV on multiple datasets:
- dataset_judi_online_yolo_5fold
- Judol-Detection-v2-1_5fold

Saves results to: judol/result/results_{dataset}_{timestamp}/fold_{k}
Aggregates metrics across folds into a single Excel report per dataset.
"""

import os
import yaml
import datetime
from datetime import timezone, timedelta
import logging
import traceback
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import pandas as pd
import psutil
import torch
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from ultralytics import YOLO

# ==============================================================================
# LOGGING SETUP
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
SCRIPT_DIR = Path(__file__).resolve().parent  # /home/ftib/ultralytics
JUDOL_DIR = SCRIPT_DIR / "judol"

DATASETS = [
    # "Judol-Detection-v2-1_5fold",
    "dataset_judi_online_yolo_5fold",
]
NUM_FOLDS = 5

# Model - YOLOv12s
MODEL_CFG = SCRIPT_DIR / "ultralytics" / "cfg" / "models" / "12" / "yolo12.yaml"
MODEL_SCALE = "s"  # Use small scale
RESULT_DIR = JUDOL_DIR / "result"

def get_jakarta_time() -> datetime.datetime:
    return datetime.datetime.now(timezone(timedelta(hours=7)))

@dataclass
class TrainingConfig:
    batch_size: int = 16
    device: str = "0"
    img_size: int = 480
    epochs: int = 100
    workers: int = 8
    patience: int = 30
    optimizer: str = "adamW"
    cos_lr: bool = True
    close_mosaic: int = 10
    amp: bool = True

@dataclass
class FoldMetrics:
    fold_idx: int = 0
    model: str = ""
    start_time: str = "-"
    end_time: str = "-"
    duration: str = "-"
    last_epoch: int = 0
    status: str = "Pending"
    output_dir: str = "-"

    # Overall metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0

    # Model performance
    gflops: float = 0.0
    parameters: int = 0

    # Timing metrics (ms)
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    nms_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0

    # Loss values
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0

    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_system_info() -> Dict[str, Any]:
    info = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_gpu_name"] = torch.cuda.get_device_name(0)
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info["gpu"] = ", ".join(f"{g.name} (VRAM: {g.memoryTotal}MB)" for g in gpus)
        except Exception:
            pass
    return info

def get_scaled_model_path(model_cfg_path: Path, scale: str) -> str:
    import re
    stem = model_cfg_path.stem   
    suffix = model_cfg_path.suffix  
    parent = model_cfg_path.parent
    pattern = re.compile(r'^(yolo[ev]?)(\d+)(.*?)$', re.IGNORECASE)
    match = pattern.match(stem)
    if match:
        prefix, version, rest = match.group(1), match.group(2), match.group(3)
        scaled_stem = f"{prefix}{version}{scale}{rest}"
        scaled_path = parent / f"{scaled_stem}{suffix}"
        return str(scaled_path)
    return str(model_cfg_path)

# ==============================================================================
# METRICS EXTRACTION
# ==============================================================================

def extract_training_metrics(results, model, metrics: FoldMetrics) -> FoldMetrics:
    try:
        results_dict = {}
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict

        def get_val(keys, default=0.0):
            for k in keys:
                if k in results_dict:
                    return float(results_dict[k])
            return default

        metrics.precision = get_val(['metrics/precision(B)', 'precision'])
        metrics.recall = get_val(['metrics/recall(B)', 'recall'])
        metrics.map50 = get_val(['metrics/mAP50(B)', 'map50'])
        metrics.map50_95 = get_val(['metrics/mAP50-95(B)', 'map50-95', 'map'])

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)

        metrics.box_loss = get_val(['train/box_loss', 'box_loss'])
        metrics.cls_loss = get_val(['train/cls_loss', 'cls_loss'])
        metrics.dfl_loss = get_val(['train/dfl_loss', 'dfl_loss'])

        if hasattr(results, 'epoch'):
            metrics.last_epoch = int(results.epoch)

        logger.info(f"  • Metrics: P={metrics.precision:.4f}, R={metrics.recall:.4f}, "
                     f"mAP50={metrics.map50:.4f}, mAP50-95={metrics.map50_95:.4f}")
    except Exception as e:
        logger.warning(f"Error extracting training metrics: {e}")
    return metrics

def extract_model_performance(model, metrics: FoldMetrics, img_size: int) -> FoldMetrics:
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            metrics.parameters = sum(p.numel() for p in model.model.parameters())
        if hasattr(model, 'info'):
            try:
                info = model.info(verbose=False, imgsz=img_size)
                if isinstance(info, tuple) and len(info) >= 4:
                    metrics.gflops = float(info[3]) if info[3] else 0.0
            except Exception:
                pass
        logger.info(f"  • Model: {metrics.parameters:,} params, {metrics.gflops:.2f} GFLOPs")
    except Exception as e:
        logger.warning(f"Error extracting model performance: {e}")
    return metrics

def extract_validation_metrics(model, val_results, metrics: FoldMetrics) -> FoldMetrics:
    try:
        if val_results is None:
            return metrics

        if hasattr(val_results, 'speed'):
            speed = val_results.speed
            metrics.preprocess_ms = float(speed.get('preprocess', 0.0))
            metrics.inference_ms = float(speed.get('inference', 0.0))
            metrics.nms_ms = float(speed.get('nms', 0.0)) if 'nms' in speed else 0.0
            metrics.postprocess_ms = float(speed.get('postprocess', 0.0))
            metrics.total_ms = metrics.preprocess_ms + metrics.inference_ms + metrics.nms_ms + metrics.postprocess_ms

        if hasattr(val_results, 'box'):
            box = val_results.box
            names = getattr(val_results, 'names', getattr(model, 'names', {}))

            if hasattr(box, 'ap50') and hasattr(box, 'ap'):
                ap50_per_class = box.ap50
                ap_per_class = box.ap
                p_per_class = box.p if hasattr(box, 'p') else [0.0] * len(ap50_per_class)
                r_per_class = box.r if hasattr(box, 'r') else [0.0] * len(ap50_per_class)

                for i, (ap50, ap, p, r) in enumerate(zip(ap50_per_class, ap_per_class, p_per_class, r_per_class)):
                    class_name = names.get(i, f"class_{i}")
                    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                    metrics.per_class_metrics[class_name] = {
                        'precision': float(p),
                        'recall': float(r),
                        'f1_score': float(f1),
                        'mAP50': float(ap50),
                        'mAP50-95': float(ap),
                    }
    except Exception as e:
        logger.warning(f"Error extracting validation metrics: {e}")
    return metrics

# ==============================================================================
# EXCEL LOGGING
# ==============================================================================

def save_cv_log(all_metrics: List[FoldMetrics], excel_file: Path, system_info: Dict) -> None:
    try:
        summary_rows = []
        metrics_rows = []
        perf_rows = []
        
        # Calculate averages for metrics
        avg_precision = np.mean([m.precision for m in all_metrics if str(m.precision) != 'nan'])
        avg_recall = np.mean([m.recall for m in all_metrics if str(m.recall) != 'nan'])
        avg_f1 = np.mean([m.f1_score for m in all_metrics if str(m.f1_score) != 'nan'])
        avg_map50 = np.mean([m.map50 for m in all_metrics if str(m.map50) != 'nan'])
        avg_map50_95 = np.mean([m.map50_95 for m in all_metrics if str(m.map50_95) != 'nan'])

        for m in all_metrics:
            summary_rows.append({
                'Fold': f"Fold {m.fold_idx}",
                'Status': m.status,
                'Start Time': m.start_time,
                'Duration': m.duration,
                'Last Epoch': m.last_epoch,
                'Output Dir': m.output_dir,
            })
            metrics_rows.append({
                'Fold': f"Fold {m.fold_idx}",
                'Precision': m.precision,
                'Recall': m.recall,
                'F1-Score': m.f1_score,
                'mAP50': m.map50,
                'mAP50-95': m.map50_95,
            })
            perf_rows.append({
                'Fold': f"Fold {m.fold_idx}",
                'GFLOPs': m.gflops,
                'Parameters': m.parameters,
                'Total (ms)': m.total_ms,
                'Box Loss': m.box_loss,
                'Cls Loss': m.cls_loss,
            })

        # Add Average Row
        metrics_rows.append({
            'Fold': "AVERAGE (CV)",
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1,
            'mAP50': avg_map50,
            'mAP50-95': avg_map50_95,
        })
        
        # Aggregate per-class metrics
        class_metrics_acc = {}
        for m in all_metrics:
            for cls_name, vals in m.per_class_metrics.items():
                if cls_name not in class_metrics_acc:
                    class_metrics_acc[cls_name] = {'p':[], 'r':[], 'f1':[], 'map50':[], 'map5095':[]}
                class_metrics_acc[cls_name]['p'].append(vals.get('precision', 0.0))
                class_metrics_acc[cls_name]['r'].append(vals.get('recall', 0.0))
                class_metrics_acc[cls_name]['f1'].append(vals.get('f1_score', 0.0))
                class_metrics_acc[cls_name]['map50'].append(vals.get('mAP50', 0.0))
                class_metrics_acc[cls_name]['map5095'].append(vals.get('mAP50-95', 0.0))

        per_class_rows = []
        for cls_name, lists in class_metrics_acc.items():
            per_class_rows.append({
                'Class': cls_name,
                'Avg Precision': np.mean(lists['p']),
                'Avg Recall': np.mean(lists['r']),
                'Avg F1-Score': np.mean(lists['f1']),
                'Avg mAP50': np.mean(lists['map50']),
                'Avg mAP50-95': np.mean(lists['map5095']),
            })

        sys_rows = [{'Key': k, 'Value': str(v)} for k, v in system_info.items()]

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
            pd.DataFrame(metrics_rows).to_excel(writer, sheet_name='Metrics', index=False)
            pd.DataFrame(per_class_rows).to_excel(writer, sheet_name='Per-Class CV Average', index=False)
            pd.DataFrame(perf_rows).to_excel(writer, sheet_name='Performance', index=False)
            pd.DataFrame(sys_rows).to_excel(writer, sheet_name='System Info', index=False)

        logger.info(f"📊 CV Excel log saved: {excel_file}")
    except Exception as e:
        logger.error(f"Error saving Excel: {e}")
        traceback.print_exc()

# ==============================================================================
# MAIN TRAINING
# ==============================================================================

def train_dataset(dataset_name: str, config: TrainingConfig, system_info: Dict):
    jakarta_time = get_jakarta_time()
    # Format: tanggal_bulan_tahun_jam_menit_wib (e.g., 12_03_2026_07_15_wib)
    time_str = jakarta_time.strftime('%d_%m_%Y_%H_%M_wib')
    
    experiment_name = f"results_{dataset_name}_{time_str}"
    dataset_result_dir = RESULT_DIR / experiment_name
    dataset_result_dir.mkdir(parents=True, exist_ok=True)
    
    # File logging for this dataset
    log_file = dataset_result_dir / "execution.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"🚀 STARTING 5-FOLD CV ON: {dataset_name}")
    logger.info(f"📁 Output Dir: {dataset_result_dir}")
    logger.info("=" * 60)

    dataset_root = JUDOL_DIR / dataset_name
    if not dataset_root.exists():
        logger.error(f"❌ Dataset not found: {dataset_root}")
        logger.removeHandler(file_handler)
        return

    scaled_model_path = get_scaled_model_path(MODEL_CFG, MODEL_SCALE)
    all_metrics = []

    for fold in range(NUM_FOLDS):
        logger.info(f"\n{'-' * 40}")
        logger.info(f"⚡ TRAINING FOLD {fold}/{NUM_FOLDS-1}")
        logger.info(f"{'-' * 40}")
        
        data_yaml = dataset_root / f"fold_{fold}" / "data.yaml"
        if not data_yaml.exists():
            logger.error(f"❌ data.yaml not found: {data_yaml}")
            continue

        fold_metrics = FoldMetrics(fold_idx=fold, model="yolo12s")
        start_time = get_jakarta_time()
        fold_metrics.start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            model = YOLO(scaled_model_path)
            fold_metrics = extract_model_performance(model, fold_metrics, config.img_size)

            results = model.train(
                data=str(data_yaml),
                epochs=config.epochs,
                batch=config.batch_size,
                imgsz=config.img_size,
                device=config.device,
                workers=config.workers,
                name=f"fold_{fold}",  # saves inside the project dir as fold_0, fold_1, etc.
                project=str(dataset_result_dir),
                exist_ok=True,
                patience=config.patience,
                optimizer=config.optimizer,
                cos_lr=config.cos_lr,
                close_mosaic=config.close_mosaic,
                amp=config.amp,
            )

            if results:
                fold_metrics = extract_training_metrics(results, model, fold_metrics)

            logger.info(f"📏 Running validation for Fold {fold}...")
            val_results = model.val(data=str(data_yaml), imgsz=config.img_size)
            fold_metrics = extract_validation_metrics(model, val_results, fold_metrics)

            fold_metrics.status = "Success"
            fold_metrics.output_dir = str(dataset_result_dir / f"fold_{fold}")
            logger.info(f"✅ FOLD {fold} COMPLETED.")

        except Exception as e:
            fold_metrics.status = f"Error: {str(e)}"
            logger.error(f"❌ FOLD {fold} ERROR: {e}")
            traceback.print_exc()

        end_time = get_jakarta_time()
        duration = end_time - start_time
        fold_metrics.end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        fold_metrics.duration = str(duration)
        all_metrics.append(fold_metrics)

    # Save aggregated Excel log
    excel_file = dataset_result_dir / f"CV_Report_{dataset_name}.xlsx"
    save_cv_log(all_metrics, excel_file, system_info)
    
    logger.info(f"\n🏁 ALL {NUM_FOLDS} FOLDS COMPLETED FOR {dataset_name}.")
    logger.info(f"📊 CV Excel Report: {excel_file}")
    
    logger.removeHandler(file_handler)

def main():
    logger.info("=" * 60)
    logger.info("YOLOv12s MULTI-DATASET 5-FOLD CV TRAINING")
    logger.info("=" * 60)

    system_info = get_system_info()
    for k, v in system_info.items():
        logger.info(f"  • {k}: {v}")

    config = TrainingConfig()
    
    for dataset in DATASETS:
        train_dataset(dataset, config, system_info)

if __name__ == "__main__":
    main()
