"""
Unified YOLO Training Script for Multiple Frameworks

Script untuk melatih berbagai model YOLO dari framework berbeda:
- Ultralytics (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, RT-DETR)
- YOLOv7
- YOLOv6

Semua hasil disimpan ke folder train_all dengan format:
{model_name}_{fold}_{day}_{date}_{month}_{timestamp}
"""

import os
import sys
import yaml
import json
import datetime
import logging
import traceback
import subprocess
import platform
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum

import pandas as pd
import psutil
import torch

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

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


DAYS_ID = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
MONTHS_ID = ['', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
             'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

# Base directories for each framework
ULTRALYTICS_BASE = Path("/home/ftib/ultralytics")
YOLOV7_BASE = Path("/home/ftib/yolov7")
YOLOV6_BASE = Path("/home/ftib/YOLOv6")

# Output directory
OUTPUT_BASE = Path("/home/ftib/ultralytics/train_all")


# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================
@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    framework: ModelFramework
    config_path: str  # Relative path to config file
    weights_path: Optional[str] = None  # Pretrained weights
    extra_args: Dict[str, Any] = field(default_factory=dict)


# Define all models to train
MODEL_CONFIGS: List[ModelConfig] = [
    # Ultralytics Models
    ModelConfig(
        name="yolov8n",
        framework=ModelFramework.ULTRALYTICS,
        config_path="ultralytics/cfg/models/v8/yolov8.yaml",
        weights_path="yolov8n.pt"
    ),
    # ModelConfig(
    #     name="yolov8s",
    #     framework=ModelFramework.ULTRALYTICS,
    #     config_path="ultralytics/cfg/models/v8/yolov8.yaml",
    #     weights_path="yolov8s.pt"
    # ),
    # ModelConfig(
    #     name="yolov8n-hematology",
    #     framework=ModelFramework.ULTRALYTICS,
    #     config_path="ultralytics/cfg/models/v8/yolov8n-hematology.yaml",
    #     weights_path=None
    # ),
    # ModelConfig(
    #     name="yolov9s",
    #     framework=ModelFramework.ULTRALYTICS,
    #     config_path="ultralytics/cfg/models/v9/yolov9s.yaml",
    #     weights_path="yolov9s.pt"
    # ),
    # ModelConfig(
    #     name="yolov10s",
    #     framework=ModelFramework.ULTRALYTICS,
    #     config_path="ultralytics/cfg/models/v10/yolov10s.yaml",
    #     weights_path="yolov10s.pt"
    # ),
    # ModelConfig(
    #     name="yolo11n",
    #     framework=ModelFramework.ULTRALYTICS,
    #     config_path="ultralytics/cfg/models/11/yolo11.yaml",
    #     weights_path="yolo11n.pt"
    # ),
    # ModelConfig(
    #     name="yolo12n",
    #     framework=ModelFramework.ULTRALYTICS,
    #     config_path="ultralytics/cfg/models/12/yolo12.yaml",
    #     weights_path="yolo12n.pt"
    # ),
    # YOLOv7
    ModelConfig(
        name="yolov7",
        framework=ModelFramework.YOLOV7,
        config_path="cfg/training/yolov7.yaml",
        weights_path="weights/yolov7.pt",
        extra_args={"hyp": "data/hyp.scratch.custom.yaml"}
    ),
    # ModelConfig(
    #     name="yolov7-tiny",
    #     framework=ModelFramework.YOLOV7,
    #     config_path="cfg/training/yolov7-tiny.yaml",
    #     weights_path="weights/yolov7-tiny.pt",
    #     extra_args={"hyp": "data/hyp.scratch.tiny.yaml"}
    # ),
    # YOLOv6
    ModelConfig(
        name="yolov6n",
        framework=ModelFramework.YOLOV6,
        config_path="configs/yolov6n_finetune.py",
        weights_path=None
    ),
    # ModelConfig(
    #     name="yolov6s",
    #     framework=ModelFramework.YOLOV6,
    #     config_path="configs/yolov6s_finetune.py",
    #     weights_path=None
    # ),
]


@dataclass
class TrainingConfig:
    """Global training configuration."""
    base_data_dir: str = "data"
    batch_size: int = 32
    device: str = "0"
    img_size: int = 480
    epochs: int = 2
    workers: int = 8
    patience: int = 30
    optimizer: str = "auto"
    cos_lr: bool = True
    close_mosaic: int = 10
    amp: bool = True


@dataclass
class TrainingResult:
    """Result from a single training run."""
    model_name: str
    fold: str
    framework: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: datetime.timedelta
    status: str
    output_dir: str
    last_epoch: int = 0
    epochs_before_patience: int = 0
    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    # Model info
    gflops: float = 0.0
    parameters: int = 0
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
def get_system_info() -> Dict[str, Any]:
    """Collect system information for logging."""
    info: Dict[str, Any] = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info["gpu"] = ", ".join([f"{gpu.name} ({gpu.memoryTotal}MB)" for gpu in gpus])
        except Exception:
            pass
    
    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_gpu_name"] = torch.cuda.get_device_name(0)
    
    return info


def get_fold_directories(base_dir: str) -> List[str]:
    """Find all fold directories in base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    return sorted([
        d.name for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith('fold_')
    ])


def generate_experiment_name(model_name: str, fold: str, timestamp: datetime.datetime) -> str:
    """Generate experiment name with standard format."""
    day_name = DAYS_ID[timestamp.weekday()]
    month_name = MONTHS_ID[timestamp.month]
    time_str = timestamp.strftime('%H_%M')
    return f"{model_name}_{fold}_{day_name}_{timestamp.day}_{month_name}_{timestamp.year}_{time_str}"


def ensure_output_dir():
    """Ensure output directory exists."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# ULTRALYTICS TRAINING
# ==============================================================================
def train_ultralytics(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    fold: str,
    fold_path: Path,
    experiment_name: str
) -> TrainingResult:
    """Train an Ultralytics model."""
    from ultralytics import YOLO
    
    start_time = datetime.datetime.now()
    yaml_file = fold_path / "data.yml"
    output_dir = OUTPUT_BASE / experiment_name
    
    result = TrainingResult(
        model_name=model_config.name,
        fold=fold,
        framework=model_config.framework.value,
        start_time=start_time,
        end_time=start_time,
        duration=datetime.timedelta(),
        status="Pending",
        output_dir=str(output_dir)
    )
    
    try:
        logger.info(f"  Loading model config: {model_config.config_path}")
        config_path = ULTRALYTICS_BASE / model_config.config_path
        model = YOLO(str(config_path))
        
        # Load pretrained weights if available
        if model_config.weights_path:
            weight_path = Path(model_config.weights_path)
            if weight_path.exists():
                logger.info(f"  Loading weights: {weight_path}")
                model = model.load(str(weight_path))
            else:
                logger.warning(f"  Weights not found: {weight_path}")
        
        # Train
        logger.info(f"  Starting training...")
        train_results = model.train(
            data=str(yaml_file),
            epochs=training_config.epochs,
            batch=training_config.batch_size,
            imgsz=training_config.img_size,
            device=training_config.device,
            workers=training_config.workers,
            name=experiment_name,
            project=str(OUTPUT_BASE),
            exist_ok=True,
            patience=training_config.patience,
            optimizer=training_config.optimizer,
            cos_lr=training_config.cos_lr,
            close_mosaic=training_config.close_mosaic,
            amp=training_config.amp
        )
        
        # Extract metrics from results
        if hasattr(train_results, 'results_dict'):
            rd = train_results.results_dict
            result.precision = rd.get('metrics/precision(B)', 0.0)
            result.recall = rd.get('metrics/recall(B)', 0.0)
            result.map50 = rd.get('metrics/mAP50(B)', 0.0)
            result.map50_95 = rd.get('metrics/mAP50-95(B)', 0.0)
        
        # Get model info
        model_info = model.info(verbose=False)
        if model_info:
            result.parameters = model_info[1] if len(model_info) > 1 else 0
            result.gflops = model_info[3] if len(model_info) > 3 else 0.0
        
        # Get speed metrics from validation
        try:
            val_metrics = model.val(data=str(yaml_file), imgsz=training_config.img_size, device=training_config.device)
            if hasattr(val_metrics, 'speed'):
                speed = val_metrics.speed
                result.preprocess_ms = speed.get('preprocess', 0)
                result.inference_ms = speed.get('inference', 0)
                result.loss_ms = speed.get('loss', 0)
                result.postprocess_ms = speed.get('postprocess', 0)
                result.total_ms = result.preprocess_ms + result.inference_ms + result.loss_ms + result.postprocess_ms
            
            # Per-class metrics
            if hasattr(val_metrics, 'box') and hasattr(val_metrics, 'names'):
                for i in range(len(val_metrics.box.p)):
                    class_idx = val_metrics.box.ap_class_index[i] if i < len(val_metrics.box.ap_class_index) else i
                    class_name = val_metrics.names.get(class_idx, f"class_{class_idx}")
                    result.per_class_metrics.append({
                        'class': class_name,
                        'precision': float(val_metrics.box.p[i]),
                        'recall': float(val_metrics.box.r[i]),
                        'f1': float(val_metrics.box.f1[i]),
                        'map50': float(val_metrics.box.ap50[i]),
                        'map50_95': float(val_metrics.box.ap[i])
                    })
                
                # Calculate mean F1
                result.f1 = float(val_metrics.box.f1.mean()) if len(val_metrics.box.f1) > 0 else 0.0
        except Exception as e:
            logger.warning(f"  Could not get validation metrics: {e}")
        
        # Try to get last epoch info
        try:
            results_csv = output_dir / "results.csv"
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                result.last_epoch = len(df)
                result.epochs_before_patience = len(df)
        except Exception:
            pass
        
        result.status = "Success"
        logger.info(f"  ✅ Training completed for {model_config.name}")
        
    except Exception as e:
        result.status = f"Error: {str(e)}"
        logger.error(f"  ❌ Training failed: {e}")
        logger.debug(traceback.format_exc())
    
    end_time = datetime.datetime.now()
    result.end_time = end_time
    result.duration = end_time - start_time
    
    return result


# ==============================================================================
# YOLOV7 TRAINING
# ==============================================================================
def train_yolov7(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    fold: str,
    fold_path: Path,
    experiment_name: str
) -> TrainingResult:
    """Train a YOLOv7 model via subprocess."""
    start_time = datetime.datetime.now()
    yaml_file = fold_path / "data.yml"
    output_dir = OUTPUT_BASE / experiment_name
    
    result = TrainingResult(
        model_name=model_config.name,
        fold=fold,
        framework=model_config.framework.value,
        start_time=start_time,
        end_time=start_time,
        duration=datetime.timedelta(),
        status="Pending",
        output_dir=str(output_dir)
    )
    
    try:
        # Build command
        cmd = [
            "python", "train.py",
            "--workers", str(training_config.workers),
            "--device", training_config.device,
            "--batch-size", str(training_config.batch_size),
            "--data", str(yaml_file),
            "--img-size", str(training_config.img_size), str(training_config.img_size),
            "--cfg", model_config.config_path,
            "--name", experiment_name,
            "--project", str(OUTPUT_BASE),
            "--epochs", str(training_config.epochs),
            "--exist-ok"
        ]
        
        # Add weights if available
        if model_config.weights_path:
            weights_path = YOLOV7_BASE / model_config.weights_path
            if weights_path.exists():
                cmd.extend(["--weights", str(weights_path)])
        
        # Add hyperparameter file
        if "hyp" in model_config.extra_args:
            hyp_path = YOLOV7_BASE / model_config.extra_args["hyp"]
            if hyp_path.exists():
                cmd.extend(["--hyp", str(hyp_path)])
        
        logger.info(f"  Running: {' '.join(cmd)}")
        
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=str(YOLOV7_BASE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Capture output
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            logger.info(f"    [YOLOv7] {line}")
        
        process.wait()
        
        # Check for success based on weights file existence (more reliable than returncode)
        best_pt = output_dir / "weights" / "best.pt"
        last_pt = output_dir / "weights" / "last.pt"
        training_succeeded = best_pt.exists() or last_pt.exists()
        
        if training_succeeded or process.returncode == 0:
            result.status = "Success"
            logger.info(f"  ✅ Training completed for {model_config.name}")
            
            # Try to extract metrics from output or results file
            try:
                results_file = output_dir / "results.txt"
                if results_file.exists():
                    with open(results_file) as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip().split()
                            if len(last_line) >= 7:
                                result.last_epoch = int(float(last_line[0]))
                                result.precision = float(last_line[4])
                                result.recall = float(last_line[5])
                                result.map50 = float(last_line[6])
                                result.map50_95 = float(last_line[7]) if len(last_line) > 7 else 0.0
            except Exception as e:
                logger.warning(f"  Could not parse results: {e}")
        else:
            output_text = '\n'.join(output_lines[-20:])  # Last 20 lines for debugging
            result.status = f"Error: Exit code {process.returncode}"
            logger.error(f"  ❌ Training failed. Last output:\n{output_text}")
        
    except Exception as e:
        result.status = f"Error: {str(e)}"
        logger.error(f"  ❌ Training failed: {e}")
        logger.debug(traceback.format_exc())
    
    end_time = datetime.datetime.now()
    result.end_time = end_time
    result.duration = end_time - start_time
    
    return result


# ==============================================================================
# YOLOV6 TRAINING
# ==============================================================================
def train_yolov6(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    fold: str,
    fold_path: Path,
    experiment_name: str
) -> TrainingResult:
    """Train a YOLOv6 model via subprocess."""
    start_time = datetime.datetime.now()
    yaml_file = fold_path / "data.yml"
    output_dir = OUTPUT_BASE / experiment_name
    
    result = TrainingResult(
        model_name=model_config.name,
        fold=fold,
        framework=model_config.framework.value,
        start_time=start_time,
        end_time=start_time,
        duration=datetime.timedelta(),
        status="Pending",
        output_dir=str(output_dir)
    )
    
    try:
        # Build command
        cmd = [
            "python", "tools/train.py",
            "--batch-size", str(training_config.batch_size),
            "--conf-file", model_config.config_path,
            "--data-path", str(yaml_file),
            "--fuse_ab",
            "--device", training_config.device,
            "--workers", str(training_config.workers),
            "--name", experiment_name,
            "--output-dir", str(OUTPUT_BASE),
            "--img-size", str(training_config.img_size),
            "--epochs", str(training_config.epochs),
            "--eval-final-only"
        ]
        
        logger.info(f"  Running: {' '.join(cmd)}")
        
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=str(YOLOV6_BASE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Capture output
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            logger.info(f"    [YOLOv6] {line}")
        
        process.wait()
        output_text = '\n'.join(output_lines)
        
        if process.returncode == 0:
            result.status = "Success"
            logger.info(f"  ✅ Training completed for {model_config.name}")
            
            # Try to extract metrics from output
            try:
                # Look for mAP patterns in output
                map_match = re.search(r'mAP@0\.5\s*[:\s]+([\d.]+)', output_text)
                if map_match:
                    result.map50 = float(map_match.group(1))
                
                map95_match = re.search(r'mAP@0\.5:0\.95\s*[:\s]+([\d.]+)', output_text)
                if map95_match:
                    result.map50_95 = float(map95_match.group(1))
            except Exception as e:
                logger.warning(f"  Could not parse metrics: {e}")
        else:
            last_output = '\n'.join(output_lines[-10:])  # Last 10 lines
            result.status = f"Error: Exit code {process.returncode}"
            logger.error(f"  ❌ Training failed. Last output:\n{last_output}")        
    except Exception as e:
        result.status = f"Error: {str(e)}"
        logger.error(f"  ❌ Training failed: {e}")
        logger.debug(traceback.format_exc())
    
    end_time = datetime.datetime.now()
    result.end_time = end_time
    result.duration = end_time - start_time
    
    return result


# ==============================================================================
# TRAINING ROUTER
# ==============================================================================
def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    fold: str,
    fold_path: Path
) -> TrainingResult:
    """Route training to appropriate framework."""
    start_time = datetime.datetime.now()
    experiment_name = generate_experiment_name(model_config.name, fold, start_time)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_config.name} | Fold: {fold}")
    logger.info(f"Framework: {model_config.framework.value}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"{'='*60}")
    
    if model_config.framework == ModelFramework.ULTRALYTICS:
        return train_ultralytics(model_config, training_config, fold, fold_path, experiment_name)
    elif model_config.framework == ModelFramework.YOLOV7:
        return train_yolov7(model_config, training_config, fold, fold_path, experiment_name)
    elif model_config.framework == ModelFramework.YOLOV6:
        return train_yolov6(model_config, training_config, fold, fold_path, experiment_name)
    else:
        raise ValueError(f"Unknown framework: {model_config.framework}")


# ==============================================================================
# EXCEL EXPORT
# ==============================================================================
def export_results_to_excel(results: List[TrainingResult], output_path: Path):
    """Export all training results to Excel with multiple sheets."""
    logger.info(f"\n📊 Exporting results to {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Training Summary
        summary_data = []
        for r in results:
            summary_data.append({
                'Model': r.model_name,
                'Fold': r.fold,
                'Framework': r.framework,
                'Start Time': r.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'End Time': r.end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Duration': str(r.duration),
                'Duration (seconds)': r.duration.total_seconds(),
                'Last Epoch': r.last_epoch,
                'Epochs Before Patience': r.epochs_before_patience,
                'Status': r.status,
                'Output Dir': r.output_dir
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Training Summary', index=False)
        logger.info(f"   ✅ Sheet 'Training Summary' ({len(summary_data)} rows)")
        
        # Sheet 2: Metrics Summary
        metrics_data = []
        for r in results:
            if 'Success' in r.status:
                metrics_data.append({
                    'Model': r.model_name,
                    'Fold': r.fold,
                    'Precision': round(r.precision, 4),
                    'Recall': round(r.recall, 4),
                    'F1': round(r.f1, 4),
                    'mAP50': round(r.map50, 4),
                    'mAP50-95': round(r.map50_95, 4),
                    'GFLOPs': round(r.gflops, 2),
                    'Parameters': r.parameters
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_excel(writer, sheet_name='Metrics Summary', index=False)
            logger.info(f"   ✅ Sheet 'Metrics Summary' ({len(metrics_data)} rows)")
            
            # Sheet 3: Model Average (grouped by model name)
            numeric_cols = ['Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95', 'GFLOPs', 'Parameters']
            df_avg = df_metrics.groupby('Model').agg({col: 'mean' for col in numeric_cols}).reset_index()
            df_avg['Num_Folds'] = df_metrics.groupby('Model').size().values
            for col in numeric_cols:
                df_avg[col] = df_avg[col].round(4)
            df_avg.to_excel(writer, sheet_name='Model Average', index=False)
            logger.info(f"   ✅ Sheet 'Model Average' ({len(df_avg)} rows)")
        
        # Sheet 4: Per-Class Metrics
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
                        'mAP50-95': round(pm['map50_95'], 4)
                    })
        
        if per_class_data:
            df_per_class = pd.DataFrame(per_class_data)
            df_per_class.to_excel(writer, sheet_name='Per-Class Metrics', index=False)
            logger.info(f"   ✅ Sheet 'Per-Class Metrics' ({len(per_class_data)} rows)")
            
            # Per-Class Average
            class_numeric_cols = ['Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95']
            df_class_avg = df_per_class.groupby(['Model', 'Class']).agg({
                col: 'mean' for col in class_numeric_cols
            }).reset_index()
            for col in class_numeric_cols:
                df_class_avg[col] = df_class_avg[col].round(4)
            df_class_avg.to_excel(writer, sheet_name='Per-Class Average', index=False)
            logger.info(f"   ✅ Sheet 'Per-Class Average' ({len(df_class_avg)} rows)")
        
        # Sheet 5: Model Performance (timing)
        perf_data = []
        for r in results:
            if 'Success' in r.status:
                perf_data.append({
                    'Model': r.model_name,
                    'Fold': r.fold,
                    'GFLOPs': round(r.gflops, 2),
                    'Parameters': r.parameters,
                    'Preprocess (ms)': round(r.preprocess_ms, 4),
                    'Inference (ms)': round(r.inference_ms, 4),
                    'NMS (ms)': round(r.nms_ms, 4),
                    'Loss (ms)': round(r.loss_ms, 4),
                    'Postprocess (ms)': round(r.postprocess_ms, 4),
                    'Total (ms)': round(r.total_ms, 4)
                })
        
        if perf_data:
            df_perf = pd.DataFrame(perf_data)
            df_perf.to_excel(writer, sheet_name='Model Performance', index=False)
            logger.info(f"   ✅ Sheet 'Model Performance' ({len(perf_data)} rows)")
            
            # Performance Average
            perf_numeric_cols = ['GFLOPs', 'Parameters', 'Preprocess (ms)', 'Inference (ms)', 
                                'NMS (ms)', 'Loss (ms)', 'Postprocess (ms)', 'Total (ms)']
            df_perf_avg = df_perf.groupby('Model').agg({col: 'mean' for col in perf_numeric_cols}).reset_index()
            df_perf_avg['Num_Folds'] = df_perf.groupby('Model').size().values
            for col in perf_numeric_cols:
                df_perf_avg[col] = df_perf_avg[col].round(4)
            df_perf_avg.to_excel(writer, sheet_name='Performance Average', index=False)
            logger.info(f"   ✅ Sheet 'Performance Average' ({len(df_perf_avg)} rows)")
    
    logger.info(f"\n🎉 Results exported to: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    """Main training loop."""
    logger.info("\n" + "="*80)
    logger.info("UNIFIED YOLO TRAINING SCRIPT")
    logger.info("="*80)
    
    # Print system info
    sys_info = get_system_info()
    logger.info(f"System: {sys_info.get('os', 'Unknown')}")
    logger.info(f"Python: {sys_info.get('python_version', 'Unknown')}")
    logger.info(f"PyTorch: {sys_info.get('torch_version', 'Unknown')}")
    logger.info(f"GPU: {sys_info.get('torch_gpu_name', 'Not available')}")
    
    # Initialize
    training_config = TrainingConfig()
    ensure_output_dir()
    
    # Find fold directories
    fold_dirs = get_fold_directories(training_config.base_data_dir)
    if not fold_dirs:
        logger.error(f"No fold directories found in {training_config.base_data_dir}")
        return
    
    logger.info(f"\nFound {len(fold_dirs)} folds: {fold_dirs}")
    logger.info(f"Training {len(MODEL_CONFIGS)} models")
    
    # Training results
    all_results: List[TrainingResult] = []
    
    # Train each model on each fold
    for model_config in MODEL_CONFIGS:
        for fold in fold_dirs:
            fold_path = Path(training_config.base_data_dir).resolve() / fold
            
            # Validate data.yml exists
            yaml_file = fold_path / "data.yml"
            if not yaml_file.exists():
                logger.warning(f"Skipping {fold}: data.yml not found")
                continue
            
            # Train
            result = train_model(model_config, training_config, fold, fold_path)
            all_results.append(result)
            
            # Save intermediate results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = OUTPUT_BASE / f"training_log_{timestamp}.xlsx"
            export_results_to_excel(all_results, excel_path)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    
    successful = sum(1 for r in all_results if 'Success' in r.status)
    failed = len(all_results) - successful
    
    logger.info(f"Total runs: {len(all_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if all_results:
        total_duration = sum([r.duration for r in all_results], datetime.timedelta())
        avg_duration = total_duration / len(all_results)
        logger.info(f"Total duration: {total_duration}")
        logger.info(f"Average duration: {avg_duration}")


if __name__ == "__main__":
    main()
