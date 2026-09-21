"""
Unified YOLO Evaluation Script for Multiple Frameworks

Script untuk evaluasi berbagai model YOLO dari framework berbeda:
- Ultralytics (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, RT-DETR)
- YOLOv7
- YOLOv6

Menghasilkan Excel dengan metrik lengkap:
- Accuracy, Precision, Recall, F1, mAP50, mAP50-95 (per kelas & semua kelas)
- GFLOPs, Parameters, Pre-processing, Inference, NMS, Loss, Post-Process, Total (ms)
"""

import os
import glob
import re
import datetime
import logging
import traceback
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum

import pandas as pd
import torch

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
# ENUMS & CONSTANTS
# ==============================================================================
class ModelFramework(Enum):
    ULTRALYTICS = "ultralytics"
    YOLOV7 = "yolov7"
    YOLOV6 = "yolov6"


# Base directories for each framework
ULTRALYTICS_BASE = Path("/home/ftib/ultralytics")
YOLOV7_BASE = Path("/home/ftib/yolov7")
YOLOV6_BASE = Path("/home/ftib/YOLOv6")

# Default search paths for models
MODEL_SEARCH_PATHS = [
    # train_all output
    Path("/home/ftib/ultralytics/train_all"),
    # Ultralytics default
    Path("/home/ftib/ultralytics/runs/detect"),
    Path("/home/ftib/ultralytics/run/train"),
    # YOLOv7 default
    Path("/home/ftib/yolov7/runs/train"),
    # YOLOv6 default
    Path("/home/ftib/YOLOv6/runs/train"),
]


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    data_dir: str = "data"
    batch_size: int = 16
    img_size: int = 480
    conf_threshold: float = 0.25
    iou_threshold: float = 0.6
    device: str = "0"


@dataclass
class EvalResult:
    """Result from a single evaluation."""
    model_name: str
    model_path: str
    fold: str
    framework: str
    eval_time: datetime.datetime
    duration: datetime.timedelta
    status: str
    # Metrics (all classes)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0
    map75: float = 0.0
    map50_95: float = 0.0
    # Model info
    gflops: float = 0.0
    parameters: int = 0
    n_layers: int = 0
    # Speed metrics (ms)
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    nms_ms: float = 0.0
    loss_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    # Per-class metrics
    per_class_metrics: List[Dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def detect_framework(model_path: Path) -> ModelFramework:
    """Detect which framework the model belongs to based on path or file structure."""
    path_str = str(model_path).lower()
    
    # Check path patterns
    if "yolov7" in path_str or str(YOLOV7_BASE).lower() in path_str:
        return ModelFramework.YOLOV7
    elif "yolov6" in path_str or str(YOLOV6_BASE).lower() in path_str:
        return ModelFramework.YOLOV6
    else:
        # Check for YOLOv6 specific checkpoint names
        if model_path.name in ["best_ckpt.pt", "best_stop_aug_ckpt.pt", "last_ckpt.pt"]:
            return ModelFramework.YOLOV6
        return ModelFramework.ULTRALYTICS


def extract_model_name(model_path: Path) -> str:
    """Extract model name from path."""
    # Get parent folder name (usually experiment name)
    parent = model_path.parent
    if parent.name == "weights":
        parent = parent.parent
    
    return parent.name


def extract_fold_name(model_path: Path) -> Optional[str]:
    """Extract fold name from path."""
    path_str = str(model_path)
    match = re.search(r'fold_\d+', path_str)
    return match.group(0) if match else None


def find_all_models(search_paths: List[Path]) -> List[Path]:
    """Find all model checkpoint files in search paths."""
    model_files = []
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        # Search for common checkpoint filenames
        patterns = ["**/best.pt", "**/best_ckpt.pt", "**/weights/best.pt"]
        
        for pattern in patterns:
            for model_file in base_path.glob(pattern):
                if model_file.is_file():
                    model_files.append(model_file)
    
    # Remove duplicates and sort
    model_files = list(set(model_files))
    model_files.sort()
    
    return model_files


def get_data_yaml_for_fold(fold: str, data_dir: str = "data") -> Optional[Path]:
    """Get data.yml path for a specific fold."""
    # Check multiple possible locations
    possible_paths = [
        Path(data_dir) / fold / "data.yml",
        Path(data_dir) / fold / "data.yaml",
        ULTRALYTICS_BASE / data_dir / fold / "data.yml",
        YOLOV7_BASE / data_dir / fold / "data.yml",
        YOLOV6_BASE / data_dir / fold / "data.yml",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


# ==============================================================================
# ULTRALYTICS EVALUATION
# ==============================================================================
def eval_ultralytics(
    model_path: Path,
    data_yaml: Path,
    config: EvalConfig
) -> EvalResult:
    """Evaluate an Ultralytics model."""
    from ultralytics import YOLO
    
    start_time = datetime.datetime.now()
    model_name = extract_model_name(model_path)
    fold = extract_fold_name(model_path) or "unknown"
    
    result = EvalResult(
        model_name=model_name,
        model_path=str(model_path),
        fold=fold,
        framework=ModelFramework.ULTRALYTICS.value,
        eval_time=start_time,
        duration=datetime.timedelta(),
        status="Pending"
    )
    
    try:
        logger.info(f"  Loading model: {model_path}")
        model = YOLO(str(model_path))
        
        # Get model info
        model_info = model.info(verbose=True)
        if model_info:
            result.n_layers = model_info[0] if len(model_info) > 0 else 0
            result.parameters = model_info[1] if len(model_info) > 1 else 0
            result.gflops = model_info[3] if len(model_info) > 3 else 0.0
        
        logger.info(f"  📊 Model Info: {result.parameters:,} params, {result.gflops:.1f} GFLOPs")
        
        # Run validation
        metrics = model.val(
            data=str(data_yaml),
            imgsz=config.img_size,
            batch=config.batch_size,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            device=config.device,
            verbose=True,
            save_json=True,
            plots=True,
            save_txt=True,
            workers=8,
            half=False,
            augment=False,
            agnostic_nms=False,
            single_cls=False,
        )
        
        # Extract speed metrics
        if hasattr(metrics, 'speed'):
            speed = metrics.speed
            result.preprocess_ms = speed.get('preprocess', 0)
            result.inference_ms = speed.get('inference', 0)
            result.loss_ms = speed.get('loss', 0)
            result.postprocess_ms = speed.get('postprocess', 0)
            result.total_ms = (result.preprocess_ms + result.inference_ms + 
                              result.loss_ms + result.postprocess_ms)
        
        # Extract overall metrics
        if hasattr(metrics, 'box'):
            result.map50_95 = float(metrics.box.map) if metrics.box.map else 0.0
            result.map50 = float(metrics.box.map50) if metrics.box.map50 else 0.0
            result.map75 = float(metrics.box.map75) if metrics.box.map75 else 0.0
            result.precision = float(metrics.box.mp) if metrics.box.mp else 0.0
            result.recall = float(metrics.box.mr) if metrics.box.mr else 0.0
            result.f1 = float(metrics.box.f1.mean()) if len(metrics.box.f1) > 0 else 0.0
        
        # Extract per-class metrics
        if hasattr(metrics, 'box') and hasattr(metrics, 'names'):
            for i in range(len(metrics.box.p)):
                class_idx = metrics.box.ap_class_index[i] if i < len(metrics.box.ap_class_index) else i
                class_name = metrics.names.get(class_idx, f"class_{class_idx}")
                
                result.per_class_metrics.append({
                    'class': class_name,
                    'precision': float(metrics.box.p[i]) if i < len(metrics.box.p) else 0,
                    'recall': float(metrics.box.r[i]) if i < len(metrics.box.r) else 0,
                    'f1': float(metrics.box.f1[i]) if i < len(metrics.box.f1) else 0,
                    'map50': float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0,
                    'map50_95': float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0
                })
        
        result.status = "Success"
        logger.info(f"  ✅ Evaluation completed")
        logger.info(f"     mAP50-95: {result.map50_95:.4f}")
        logger.info(f"     mAP50: {result.map50:.4f}")
        logger.info(f"     Precision: {result.precision:.4f}")
        logger.info(f"     Recall: {result.recall:.4f}")
        logger.info(f"     F1: {result.f1:.4f}")
        
    except Exception as e:
        result.status = f"Error: {str(e)}"
        logger.error(f"  ❌ Evaluation failed: {e}")
        logger.debug(traceback.format_exc())
    
    end_time = datetime.datetime.now()
    result.duration = end_time - start_time
    
    return result


# ==============================================================================
# YOLOV7 EVALUATION
# ==============================================================================
def eval_yolov7(
    model_path: Path,
    data_yaml: Path,
    config: EvalConfig
) -> EvalResult:
    """Evaluate a YOLOv7 model via subprocess."""
    start_time = datetime.datetime.now()
    model_name = extract_model_name(model_path)
    fold = extract_fold_name(model_path) or "unknown"
    
    result = EvalResult(
        model_name=model_name,
        model_path=str(model_path),
        fold=fold,
        framework=ModelFramework.YOLOV7.value,
        eval_time=start_time,
        duration=datetime.timedelta(),
        status="Pending"
    )
    
    try:
        # Build command
        cmd = [
            "python", "test.py",
            "--weights", str(model_path),
            "--data", str(data_yaml),
            "--batch-size", str(config.batch_size),
            "--img-size", str(config.img_size),
            "--conf-thres", str(config.conf_threshold),
            "--iou-thres", str(config.iou_threshold),
            "--task", "val",
            "--device", config.device,
            "--name", model_name,
            "--project", "runs/val",
            "--exist-ok",
            "--save-json",
            "--verbose"
        ]
        
        logger.info(f"  Running: {' '.join(cmd[:10])}...")
        
        # Run evaluation
        process = subprocess.run(
            cmd,
            cwd=str(YOLOV7_BASE),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        output = process.stdout + process.stderr
        
        if process.returncode == 0:
            result.status = "Success"
            
            # Parse metrics from output
            # Look for patterns like: "all       100        50     0.889     0.567     0.578     0.446"
            ap_pattern = r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            match = re.search(ap_pattern, output)
            if match:
                result.precision = float(match.group(1))
                result.recall = float(match.group(2))
                result.map50 = float(match.group(3))
                result.map50_95 = float(match.group(4))
            
            # Parse speed metrics
            speed_pattern = r"Speed:\s+([\d.]+)ms\s+pre-process,\s+([\d.]+)ms\s+inference,\s+([\d.]+)ms\s+NMS"
            speed_match = re.search(speed_pattern, output)
            if speed_match:
                result.preprocess_ms = float(speed_match.group(1))
                result.inference_ms = float(speed_match.group(2))
                result.nms_ms = float(speed_match.group(3))
                result.total_ms = result.preprocess_ms + result.inference_ms + result.nms_ms
            
            # Parse per-class metrics
            class_pattern = r"(\w+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            class_matches = re.findall(class_pattern, output)
            for match in class_matches:
                if match[0] != "all":
                    result.per_class_metrics.append({
                        'class': match[0],
                        'precision': float(match[1]),
                        'recall': float(match[2]),
                        'map50': float(match[3]),
                        'map50_95': float(match[4]),
                        'f1': 2 * float(match[1]) * float(match[2]) / (float(match[1]) + float(match[2]) + 1e-6)
                    })
            
            # Calculate F1
            if result.precision > 0 or result.recall > 0:
                result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall + 1e-6)
            
            logger.info(f"  ✅ Evaluation completed")
            logger.info(f"     mAP50-95: {result.map50_95:.4f}")
            logger.info(f"     mAP50: {result.map50:.4f}")
        else:
            result.status = f"Error: Exit code {process.returncode}"
            logger.error(f"  ❌ Evaluation failed")
        
    except subprocess.TimeoutExpired:
        result.status = "Error: Timeout"
        logger.error(f"  ❌ Evaluation timed out")
    except Exception as e:
        result.status = f"Error: {str(e)}"
        logger.error(f"  ❌ Evaluation failed: {e}")
        logger.debug(traceback.format_exc())
    
    end_time = datetime.datetime.now()
    result.duration = end_time - start_time
    
    return result


# ==============================================================================
# YOLOV6 EVALUATION
# ==============================================================================
def eval_yolov6(
    model_path: Path,
    data_yaml: Path,
    config: EvalConfig
) -> EvalResult:
    """Evaluate a YOLOv6 model via subprocess."""
    start_time = datetime.datetime.now()
    model_name = extract_model_name(model_path)
    fold = extract_fold_name(model_path) or "unknown"
    
    result = EvalResult(
        model_name=model_name,
        model_path=str(model_path),
        fold=fold,
        framework=ModelFramework.YOLOV6.value,
        eval_time=start_time,
        duration=datetime.timedelta(),
        status="Pending"
    )
    
    try:
        # Build command
        cmd = [
            "python", "tools/eval.py",
            "--data", str(data_yaml),
            "--weights", str(model_path),
            "--device", config.device,
            "--task", "val",
            "--conf-thres", str(config.conf_threshold),
            "--iou-thres", str(config.iou_threshold),
            "--do_pr_metric", "True",
            "--plot_curve", "True",
            "--plot_confusion_matrix",
            "--height", str(config.img_size),
            "--width", str(config.img_size),
            "--name", model_name
        ]
        
        logger.info(f"  Running: {' '.join(cmd[:10])}...")
        
        # Run evaluation
        process = subprocess.run(
            cmd,
            cwd=str(YOLOV6_BASE),
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        output = process.stdout + process.stderr
        
        if process.returncode == 0:
            result.status = "Success"
            
            # Parse metrics from output
            # Different YOLOv6 versions may have different output formats
            map_pattern = r"mAP@0\.5\s*[=:]\s*([\d.]+)"
            match = re.search(map_pattern, output)
            if match:
                result.map50 = float(match.group(1))
            
            map95_pattern = r"mAP@0\.5:0\.95\s*[=:]\s*([\d.]+)"
            match = re.search(map95_pattern, output)
            if match:
                result.map50_95 = float(match.group(1))
            
            # Look for precision/recall
            pr_pattern = r"Precision\s*[=:]\s*([\d.]+)"
            match = re.search(pr_pattern, output)
            if match:
                result.precision = float(match.group(1))
            
            rec_pattern = r"Recall\s*[=:]\s*([\d.]+)"
            match = re.search(rec_pattern, output)
            if match:
                result.recall = float(match.group(1))
            
            # Calculate F1
            if result.precision > 0 or result.recall > 0:
                result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall + 1e-6)
            
            logger.info(f"  ✅ Evaluation completed")
            logger.info(f"     mAP50-95: {result.map50_95:.4f}")
            logger.info(f"     mAP50: {result.map50:.4f}")
        else:
            result.status = f"Error: Exit code {process.returncode}"
            logger.error(f"  ❌ Evaluation failed")
        
    except subprocess.TimeoutExpired:
        result.status = "Error: Timeout"
        logger.error(f"  ❌ Evaluation timed out")
    except Exception as e:
        result.status = f"Error: {str(e)}"
        logger.error(f"  ❌ Evaluation failed: {e}")
        logger.debug(traceback.format_exc())
    
    end_time = datetime.datetime.now()
    result.duration = end_time - start_time
    
    return result


# ==============================================================================
# EVALUATION ROUTER
# ==============================================================================
def evaluate_model(
    model_path: Path,
    config: EvalConfig
) -> Optional[EvalResult]:
    """Route evaluation to appropriate framework."""
    framework = detect_framework(model_path)
    fold = extract_fold_name(model_path)
    
    if not fold:
        logger.warning(f"  Could not detect fold for: {model_path}")
        return None
    
    # Find data yaml
    data_yaml = get_data_yaml_for_fold(fold, config.data_dir)
    if not data_yaml:
        logger.warning(f"  Could not find data.yml for fold: {fold}")
        return None
    
    model_name = extract_model_name(model_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"Framework: {framework.value}")
    logger.info(f"Fold: {fold}")
    logger.info(f"Data: {data_yaml}")
    logger.info(f"{'='*60}")
    
    if framework == ModelFramework.ULTRALYTICS:
        return eval_ultralytics(model_path, data_yaml, config)
    elif framework == ModelFramework.YOLOV7:
        return eval_yolov7(model_path, data_yaml, config)
    elif framework == ModelFramework.YOLOV6:
        return eval_yolov6(model_path, data_yaml, config)
    else:
        logger.error(f"  Unknown framework: {framework}")
        return None


# ==============================================================================
# EXCEL EXPORT
# ==============================================================================
def export_results_to_excel(results: List[EvalResult], output_path: Path):
    """Export all evaluation results to Excel with multiple sheets."""
    logger.info(f"\n📊 Exporting results to {output_path}")
    
    def extract_model_base(name: str) -> str:
        """Extract base model name before fold info."""
        match = re.match(r'(.+?)_fold_\d+', name)
        return match.group(1) if match else name
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary (per model per fold)
        summary_data = []
        for r in results:
            if 'Success' in r.status:
                summary_data.append({
                    'Model': r.model_name,
                    'Fold': r.fold,
                    'Framework': r.framework,
                    'GFLOPs': round(r.gflops, 2),
                    'Parameters': r.parameters,
                    'Mean_Precision': round(r.precision, 4),
                    'Mean_Recall': round(r.recall, 4),
                    'Mean_F1': round(r.f1, 4),
                    'mAP50': round(r.map50, 4),
                    'mAP75': round(r.map75, 4),
                    'mAP50-95': round(r.map50_95, 4),
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            logger.info(f"   ✅ Sheet 'Summary' ({len(summary_data)} rows)")
            
            # Sheet 2: Model Average
            df_summary['Model_Base'] = df_summary['Model'].apply(extract_model_base)
            numeric_cols = ['GFLOPs', 'Parameters', 'Mean_Precision', 'Mean_Recall', 
                          'Mean_F1', 'mAP50', 'mAP75', 'mAP50-95']
            df_model_avg = df_summary.groupby('Model_Base').agg({
                col: 'mean' for col in numeric_cols
            }).reset_index()
            df_model_avg['Num_Folds'] = df_summary.groupby('Model_Base').size().values
            
            for col in numeric_cols:
                if col in ['GFLOPs', 'Parameters']:
                    df_model_avg[col] = df_model_avg[col].round(2)
                else:
                    df_model_avg[col] = df_model_avg[col].round(4)
            
            df_model_avg = df_model_avg.rename(columns={'Model_Base': 'Model'})
            cols_order = ['Model', 'Num_Folds', 'GFLOPs', 'Parameters', 'Mean_Precision', 
                         'Mean_Recall', 'Mean_F1', 'mAP50', 'mAP75', 'mAP50-95']
            df_model_avg = df_model_avg[cols_order]
            df_model_avg.to_excel(writer, sheet_name='Model Average', index=False)
            logger.info(f"   ✅ Sheet 'Model Average' ({len(df_model_avg)} rows)")
        
        # Sheet 3: Per-Class Metrics
        per_class_data = []
        for r in results:
            if 'Success' in r.status and r.per_class_metrics:
                for pm in r.per_class_metrics:
                    per_class_data.append({
                        'Model': r.model_name,
                        'Fold': r.fold,
                        'Class': pm['class'],
                        'Precision': round(pm['precision'], 4),
                        'Recall': round(pm['recall'], 4),
                        'F1': round(pm['f1'], 4),
                        'mAP50': round(pm['map50'], 4),
                        'mAP50-95': round(pm['map50_95'], 4),
                    })
        
        if per_class_data:
            df_per_class = pd.DataFrame(per_class_data)
            df_per_class.to_excel(writer, sheet_name='Per-Class Metrics', index=False)
            logger.info(f"   ✅ Sheet 'Per-Class Metrics' ({len(per_class_data)} rows)")
            
            # Sheet 4: Per-Class Average
            df_per_class['Model_Base'] = df_per_class['Model'].apply(extract_model_base)
            class_numeric_cols = ['Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95']
            df_class_avg = df_per_class.groupby(['Model_Base', 'Class']).agg({
                col: 'mean' for col in class_numeric_cols
            }).reset_index()
            df_class_avg['Num_Folds'] = df_per_class.groupby(['Model_Base', 'Class']).size().values
            
            for col in class_numeric_cols:
                df_class_avg[col] = df_class_avg[col].round(4)
            
            df_class_avg = df_class_avg.rename(columns={'Model_Base': 'Model'})
            cols_order = ['Model', 'Class', 'Num_Folds', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95']
            df_class_avg = df_class_avg[cols_order]
            df_class_avg.to_excel(writer, sheet_name='Per-Class Average', index=False)
            logger.info(f"   ✅ Sheet 'Per-Class Average' ({len(df_class_avg)} rows)")
        
        # Sheet 5: Model Performance (timing)
        perf_data = []
        for r in results:
            if 'Success' in r.status:
                perf_data.append({
                    'Model': r.model_name,
                    'Fold': r.fold,
                    'Framework': r.framework,
                    'GFLOPs': round(r.gflops, 2),
                    'Parameters': r.parameters,
                    'Preprocess_ms': round(r.preprocess_ms, 4),
                    'Inference_ms': round(r.inference_ms, 4),
                    'NMS_ms': round(r.nms_ms, 4),
                    'Loss_ms': round(r.loss_ms, 4),
                    'Postprocess_ms': round(r.postprocess_ms, 4),
                    'Total_ms': round(r.total_ms, 4),
                })
        
        if perf_data:
            df_performance = pd.DataFrame(perf_data)
            df_performance.to_excel(writer, sheet_name='Model Performance', index=False)
            logger.info(f"   ✅ Sheet 'Model Performance' ({len(perf_data)} rows)")
            
            # Sheet 6: Performance Average
            df_performance['Model_Base'] = df_performance['Model'].apply(extract_model_base)
            perf_numeric_cols = ['GFLOPs', 'Parameters', 'Preprocess_ms', 'Inference_ms', 
                                'NMS_ms', 'Loss_ms', 'Postprocess_ms', 'Total_ms']
            df_perf_avg = df_performance.groupby('Model_Base').agg({
                col: 'mean' for col in perf_numeric_cols
            }).reset_index()
            df_perf_avg['Num_Folds'] = df_performance.groupby('Model_Base').size().values
            
            for col in perf_numeric_cols:
                df_perf_avg[col] = df_perf_avg[col].round(4)
            
            df_perf_avg = df_perf_avg.rename(columns={'Model_Base': 'Model'})
            cols_order = ['Model', 'Num_Folds', 'GFLOPs', 'Parameters', 'Preprocess_ms', 
                         'Inference_ms', 'NMS_ms', 'Loss_ms', 'Postprocess_ms', 'Total_ms']
            df_perf_avg = df_perf_avg[cols_order]
            df_perf_avg.to_excel(writer, sheet_name='Performance Average', index=False)
            logger.info(f"   ✅ Sheet 'Performance Average' ({len(df_perf_avg)} rows)")
        
        # Sheet 7: Evaluation Log (all results including errors)
        log_data = []
        for r in results:
            log_data.append({
                'Model': r.model_name,
                'Model Path': r.model_path,
                'Fold': r.fold,
                'Framework': r.framework,
                'Eval Time': r.eval_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Duration': str(r.duration),
                'Status': r.status
            })
        
        if log_data:
            df_log = pd.DataFrame(log_data)
            df_log.to_excel(writer, sheet_name='Evaluation Log', index=False)
            logger.info(f"   ✅ Sheet 'Evaluation Log' ({len(log_data)} rows)")
    
    logger.info(f"\n🎉 Results exported to: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    """Main evaluation loop."""
    logger.info("\n" + "="*80)
    logger.info("UNIFIED YOLO EVALUATION SCRIPT")
    logger.info("="*80)
    
    # Initialize config
    config = EvalConfig()
    
    # Find all models
    logger.info("\n🔍 Searching for trained models...")
    model_files = find_all_models(MODEL_SEARCH_PATHS)
    
    if not model_files:
        logger.error("No trained models found!")
        logger.info("Searched paths:")
        for path in MODEL_SEARCH_PATHS:
            logger.info(f"  - {path}")
        return
    
    logger.info(f"Found {len(model_files)} models to evaluate")
    
    # Filter to only yolov8 models for demo (remove this line to eval all)
    # model_files = [m for m in model_files if 'yolov8' in str(m).lower()]
    
    # Evaluate each model
    all_results: List[EvalResult] = []
    
    for model_path in model_files:
        result = evaluate_model(model_path, config)
        if result:
            all_results.append(result)
    
    # Export results
    if all_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"eval_results_comprehensive_{timestamp}.xlsx")
        export_results_to_excel(all_results, output_path)
    else:
        logger.warning("No evaluation results to export")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    
    successful = sum(1 for r in all_results if 'Success' in r.status)
    failed = len(all_results) - successful
    
    logger.info(f"Total evaluations: {len(all_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if all_results:
        total_duration = sum([r.duration for r in all_results], datetime.timedelta())
        avg_duration = total_duration / len(all_results)
        logger.info(f"Total duration: {total_duration}")
        logger.info(f"Average duration per model: {avg_duration}")


if __name__ == "__main__":
    main()
