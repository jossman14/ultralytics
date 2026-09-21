"""
YOLO Cross-Validation Training Script

Script untuk melatih berbagai model YOLO dengan cross-validation (fold-based)
dan logging ke Comet ML serta Excel dengan metrik komprehensif.
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
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import psutil
import torch
import numpy as np
from itertools import combinations

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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

# Suppress Comet ML verbose logging
if COMET_AVAILABLE:
    try:
        comet_ml.get_logger().setLevel(logging.ERROR)
    except Exception:
        pass

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
MODEL_CFG_BASE_DIR = Path("/home/ftib/ultralytics/ultralytics/cfg/models")

# Model configurations - All equivalent to YOLOv8s (~11M params, ~28 GFLOPs)
MODEL_CONFIGS: List[str] = [
    # === COMMENTED OUT - Previous models ===
    # "v3/yolov3.yaml",           # YOLOv3 base (no scales - uses depth/width_multiple)
    # "v5/yolov5.yaml",           # YOLOv5s (scale 's')
    # "v6/yolov6.yaml",           # YOLOv6s (scale 's')
    # "v8/yolov8.yaml",           # YOLOv8s (scale 's')
    # "v9/yolov9s.yaml",          # YOLOv9s (dedicated file)
    # "v10/yolov10s.yaml",        # YOLOv10s (dedicated file)
    # "11/yolo11.yaml",           # YOLO11s (scale 's')
    # "12/yolo12.yaml",           # YOLO12 base (scale 's')
    # "rt-detr/rtdetr-l.yaml",    # RT-DETR-l (dedicated file)
    # "12/yolo12-ghost-gam.yaml", # YOLO12-Ghost-GAM custom (scale 's')
    # "v8/yolov8-ghost.yaml",     # YOLOv8-Ghost (scale 's')
    # "v8/yolov8-ghost-gam.yaml", # YOLOv8-Ghost-GAM custom (scale 's')
    # "v8/yolov8-hematology.yaml", # YOLOv8-Hematology custom (scale 's')
    # # SCL-YOLO variants
    # "v8/yolov8-starnet.yaml",   # YOLOv8 with StarNet backbone (scale 's')
    # "v8/yolov8-capffn.yaml",    # YOLOv8 with CAPFFN neck (scale 's')
    # "v8/yolov8-ledh.yaml",      # YOLOv8 with LEDHead (scale 's')
    # "v8/yolov8-scl.yaml",       # Complete SCL-YOLO (scale 's')
    # "v8/yolov8-scl-gam.yaml",   # SCL-YOLO with GAM (scale 's')
    
    # # Previous optimized models
    "v8/yolov8s-ghost-capffn.yaml",   # Best balance of size and accuracy
    # "v8/yolov8s-ghost-kd.yaml",       # Train with KD from yolov8-capffn teacher
    # "v8/yolov8s-ghost-eca.yaml",      # Minimal params increase with attention
    
    # === NEW: Swin Transformer Backbone ===
    # "v8/yolov8-swin.yaml",            # Custom Swin implementation
    # "v8/yolov8-swin-timm.yaml",         # Swin from timm with pre-trained weights
    
    # === Ablation Study: Swin Transformer Variants ===
    # "v8/yolov8-ghost-swin.yaml",        # Ghost + Swin backbone
    # "v8/yolov8-swin-capffn.yaml",       # Swin + CAPFFN neck
    # "v8/yolov8-swin-gam.yaml",          # Swin + GAM attention
    # "v8/yolov8-swin-eca.yaml",          # Swin + ECA attention
    "v8/yolov8-ghost-swin-capffn.yaml", # Full hybrid: Ghost + Swin + CAPFFN
]

# Scale mapping for models that use compound scaling
# Models not in this dict use their default scales or dedicated files
MODEL_SCALES: Dict[str, str] = {
    # Previous models (commented)
    # "yolov5": "s",
    # "yolov6": "s",
    # "yolov8": "s",
    # "yolo11": "s",
    # "yolo12": "s",
    # "yolo12-ghost-gam": "s",
    # "yolov8-ghost": "s",
    # "yolov8-ghost-gam": "s",
    # "yolov8-hematology": "s",
    # "yolov8s-hematology": "s",
    # "yolov8n-hematology": "n",
    # "yolov8m-hematology": "m",
    # # SCL-YOLO variants
    # "yolov8-starnet": "s",
    # "yolov8-capffn": "s",
    # "yolov8-ledh": "s",
    # "yolov8-scl": "s",
    # "yolov8-scl-gam": "s",
    "yolov8-ghost-capffn": "s",
    # "yolov8-ghost-kd": "s",
    # "yolov8-ghost-eca": "s",
    
    # New Swin Transformer models
    # "yolov8-swin": "s",              # Custom Swin implementation
    # "yolov8-swin-timm": "s",           # Swin from timm with pre-trained weights
    
    # Ablation Study: Swin variants
    # "yolov8-ghost-swin": "s",          # Ghost + Swin backbone
    # "yolov8-swin-capffn": "s",         # Swin + CAPFFN neck
    # "yolov8-swin-gam": "s",            # Swin + GAM attention
    # "yolov8-swin-eca": "s",            # Swin + ECA attention
    "yolov8-ghost-swin-capffn": "s",   # Full hybrid
}

# Knowledge Distillation Configuration
# Maps student model to teacher model weights path
KD_CONFIG: Dict[str, Dict[str, Any]] = {
    "yolov8s-ghost-kd": {
        "teacher_model": "v8/yolov8s-capffn.yaml",  # Teacher config (train first if no weights)
        "teacher_weights_pattern": "**/yolov8s-capffn*/**/best.pt",  # Pattern to find teacher weights
        "alpha": 0.5,  # Weight for hard loss (0.5 = 50% hard, 50% soft)
        "temperature": 4.0,  # Temperature for soft predictions
    },
    "yolov8s-sota-fusion-kd": {
        "teacher_model": "v8/yolov8l-sota-fusion-a.yaml",  # Larger scale of the same architecture
        "teacher_weights_pattern": "**/yolov8l-sota-fusion-a*/**/best.pt",
        "alpha": 0.5,
        "temperature": 4.0,
    },
}

DAYS_ID = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
MONTHS_ID = ['', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
             'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']


def get_jakarta_time() -> datetime.datetime:
    """Get current time in Jakarta (UTC+7)."""
    return datetime.datetime.now(timezone(timedelta(hours=7)))


def _get_timestamped_excel_log() -> str:
    """Generate timestamped excel log filename using Jakarta time."""
    # This will be updated in main() to be inside experiment dir
    timestamp = get_jakarta_time().strftime("%Y%m%d_%H%M%S")
    return f"training_log_{timestamp}.xlsx"


@dataclass
class TrainingConfig:
    """Konfigurasi untuk training YOLO."""
    base_data_dir: str = "data"
    batch_size: int = 32
    device: str = "0"
    # output_dir will be set in main()
    experiment_dir: Path = field(default_factory=lambda: Path("result"))
    excel_log: str = field(default_factory=_get_timestamped_excel_log)
    img_size: int = 480
    epochs: int = 5
    workers: int = 8
    patience: int = 30
    optimizer: str = "auto"
    cos_lr: bool = True
    close_mosaic: int = 10
    amp: bool = True
    # Knowledge Distillation settings
    kd_alpha: float = 0.5
    kd_temperature: float = 4.0
    kd_enabled: bool = True  # Enable automatic KD for supported models


@dataclass
class TrainingMetrics:
    """Metrik hasil training."""
    # Training info
    model: str = ""
    fold: str = ""
    start_time: str = "-"
    end_time: str = "-"
    duration: str = "-"
    last_epoch: int = 0
    status: str = "Pending"
    output_dir: str = "-"
    comet_enabled: str = "No"
    
    # Overall metrics
    accuracy: float = 0.0
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
    
    # Per-class metrics stored separately
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dict for Excel."""
        return {
            'Model': self.model,
            'Fold': self.fold,
            'Start Time': self.start_time,
            'End Time': self.end_time,
            'Duration': self.duration,
            'Last Epoch': self.last_epoch,
            'Status': self.status,
            'Output Dir': self.output_dir,
            'Comet Enabled': self.comet_enabled,
        }
    
    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert to metrics dict for Excel."""
        return {
            'Model': self.model,
            'Fold': self.fold,
            'Accuracy (All)': self.accuracy,
            'Precision (All)': self.precision,
            'Recall (All)': self.recall,
            'F1-Score (All)': self.f1_score,
            'mAP50 (All)': self.map50,
            'mAP50-95 (All)': self.map50_95,
        }
    
    def to_performance_dict(self) -> Dict[str, Any]:
        """Convert to performance dict for Excel."""
        return {
            'Model': self.model,
            'Fold': self.fold,
            'GFLOPs': self.gflops,
            'Parameters': self.parameters,
            'Pre-process (ms)': self.preprocess_ms,
            'Inference (ms)': self.inference_ms,
            'NMS (ms)': self.nms_ms,
            'Post-process (ms)': self.postprocess_ms,
            'Total (ms)': self.total_ms,
            'Box Loss': self.box_loss,
            'Cls Loss': self.cls_loss,
            'DFL Loss': self.dfl_loss,
        }


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_system_info() -> Dict[str, Any]:
    """Kumpulkan informasi sistem untuk logging."""
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    # GPU information
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = [f"{gpu.name} (VRAM: {gpu.memoryTotal}MB)" for gpu in gpus]
                info["gpu"] = ", ".join(gpu_info)
            else:
                info["gpu"] = "Tidak ditemukan GPU CUDA"
        except Exception as e:
            info["gpu"] = f"Error mendapatkan info GPU: {e}"
    else:
        info["gpu"] = "GPUtil tidak tersedia"
    
    # PyTorch information
    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_gpu_name"] = torch.cuda.get_device_name(0)
    
    return info


def get_fold_directories(base_dir: str) -> List[str]:
    """Cari semua direktori fold di base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    return sorted([
        d.name for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith('fold_')
    ])


def find_pretrained_weights(model_base: str, model_dir: Path) -> Optional[Path]:
    """Cari file pretrained weights di berbagai lokasi."""
    weight_name = f"{model_base}.pt"
    weight_locations = [
        Path(weight_name),
        Path("weights") / weight_name,
        Path("yolov8/weights") / weight_name,
        MODEL_CFG_BASE_DIR / model_dir / weight_name,
    ]
    
    for path in weight_locations:
        if path.exists():
            return path
    return None


def generate_experiment_name(model_base: str, fold: str, timestamp: datetime.datetime) -> str:
    """Generate nama eksperimen dengan format standar."""
    # Use simple numeric format for easier sorting: YYYYMMDD_HHMM
    time_str = timestamp.strftime('%Y%m%d_%H%M')
    return f"{model_base}_{fold}_{time_str}"


def get_model_scale(model_base: str) -> Optional[str]:
    """Get the scale for a model based on MODEL_SCALES mapping."""
    for key, scale in MODEL_SCALES.items():
        if key in model_base.lower():
            return scale
    return None


def get_scaled_model_path(model_cfg_path: Path, model_base: str) -> str:
    """
    Construct the model path with scale suffix for compound scaling models.
    
    For models that use compound scaling (like yolo12.yaml with scales n/s/m/l/x),
    this constructs the proper path to select the 's' scale by modifying the
    filename (e.g., yolo12.yaml -> yolo12s.yaml).
    
    Args:
        model_cfg_path: Path to the model config YAML file
        model_base: Base name of the model config (without extension)
    
    Returns:
        Model path string to use for YOLO initialization
    """
    import re
    
    scale = get_model_scale(model_base)
    
    # If no scale mapping, use the original path
    if not scale:
        return str(model_cfg_path)
    
    # Check if the model already has a scale suffix in the name
    # Pattern matches: yolo12, yolo11, yolov8, etc. without existing scale suffix
    has_scale_suffix = re.search(r'yolo(e-)?[v]?\d+[nslmx]', model_base.lower())
    
    if has_scale_suffix:
        # Already has a scale suffix, use as-is
        return str(model_cfg_path)
    
    # For models like yolo12.yaml, yolov8-ghost.yaml, construct scaled path
    # The YOLO library uses the filename to extract scale, so we need to
    # modify the path to include the scale
    
    # For base models (yolo12, yolo11, yolov8), append scale before .yaml
    # The library will use this to select the correct scale from the config
    stem = model_cfg_path.stem  # e.g., 'yolo12' or 'yolov8-ghost'
    suffix = model_cfg_path.suffix  # .yaml
    parent = model_cfg_path.parent
    
    # Insert scale into the model name
    # Handle patterns like: yolo12 -> yolo12s, yolov8-ghost -> yolov8s-ghost
    # The key is inserting the scale after the version number
    
    # Try to match common patterns
    pattern = re.compile(r'^(yolo[ev]?)(\d+)(.*?)$', re.IGNORECASE)
    match = pattern.match(stem)
    
    if match:
        prefix = match.group(1)  # 'yolo' or 'yolov'
        version = match.group(2)  # '12', '8', '11'
        rest = match.group(3)  # '-ghost', '-hematology', or ''
        
        scaled_stem = f"{prefix}{version}{scale}{rest}"  # yolo12s, yolov8s-ghost
        scaled_path = parent / f"{scaled_stem}{suffix}"
        
        # Check if the scaled file exists, otherwise use original with scale in name
        if scaled_path.exists():
            logger.info(f"  • Using scaled model config: {scaled_path}")
            return str(scaled_path)
        else:
            # Construct a virtual path that signals the scale
            # The YOLO library will parse the filename to get the scale
            logger.info(f"  • Using model with scale '{scale}': {model_cfg_path}")
            return str(parent / f"{scaled_stem}{suffix}")
    
    # Fallback: return original path
    return str(model_cfg_path)


# ==============================================================================
# METRICS EXTRACTION FUNCTIONS
# ==============================================================================

def extract_training_metrics(results: Any, model: Any, metrics: TrainingMetrics) -> TrainingMetrics:
    """
    Extract comprehensive training metrics from YOLO training results.
    
    Args:
        results: Training results object from model.train()
        model: The YOLO model object
        metrics: TrainingMetrics object to populate
    
    Returns:
        Updated TrainingMetrics object
    """
    try:
        # Get results dict if available
        results_dict = {}
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
        elif hasattr(results, 'results'):
            # Some versions might store it directly or differently
            logger.warning("results.results_dict missing, checking results.results")
            results_dict = results.results if isinstance(results.results, dict) else {}
        else:
            logger.warning(f"Could not find results dict in {type(results)}")
            # Log available attributes for debugging
            logger.debug(f"Available attributes: {dir(results)}")

        if not results_dict:
            logger.warning("  ⚠️  Training results dictionary is empty or not found!")
        
        # KEY MAPPING: Handle different YOLO versions key naming
        # Standardize to what we want
        keys_map = {
            'metrics/precision(B)': ['metrics/precision(B)', 'val/box_loss', 'precision'], # Fallbacks
            'metrics/recall(B)': ['metrics/recall(B)', 'recall'],
            'metrics/mAP50(B)': ['metrics/mAP50(B)', 'map50'],
            'metrics/mAP50-95(B)': ['metrics/mAP50-95(B)', 'map50-95', 'map'],
            'train/box_loss': ['train/box_loss', 'box_loss'],
            'train/cls_loss': ['train/cls_loss', 'cls_loss'],
            'train/dfl_loss': ['train/dfl_loss', 'dfl_loss'],
        }
        
        def get_val(keys_list, default=0.0):
            for k in keys_list:
                if k in results_dict:
                    return float(results_dict[k])
            return default

        # Extract overall metrics
        metrics.precision = get_val(keys_map['metrics/precision(B)'])
        metrics.recall = get_val(keys_map['metrics/recall(B)'])
        metrics.map50 = get_val(keys_map['metrics/mAP50(B)'])
        metrics.map50_95 = get_val(keys_map['metrics/mAP50-95(B)'])
        
        # Calculate F1 score if not available
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        
        # Accuracy approximation (using mAP50-95 as proxy)
        metrics.accuracy = metrics.map50_95
        
        # Loss values
        metrics.box_loss = get_val(keys_map['train/box_loss'])
        metrics.cls_loss = get_val(keys_map['train/cls_loss'])
        metrics.dfl_loss = get_val(keys_map['train/dfl_loss'])
        
        # Get last epoch
        if hasattr(results, 'epoch'):
            metrics.last_epoch = int(results.epoch)
        elif hasattr(model, 'trainer') and hasattr(model.trainer, 'epoch'):
             metrics.last_epoch = int(model.trainer.epoch)
        
        logger.info(f"  • Extracted metrics: P={metrics.precision:.4f}, R={metrics.recall:.4f}, "
                   f"mAP50={metrics.map50:.4f}, mAP50-95={metrics.map50_95:.4f}")
        
    except Exception as e:
        logger.warning(f"Error extracting training metrics: {e}")
        traceback.print_exc()
    
    return metrics


def extract_model_performance(model: Any, metrics: TrainingMetrics, img_size: int = 480) -> TrainingMetrics:
    """
    Extract model performance metrics (GFLOPs, parameters, timing).
    
    Args:
        model: The YOLO model object
        metrics: TrainingMetrics object to populate
        img_size: Image size for profiling
    
    Returns:
        Updated TrainingMetrics object
    """
    try:
        # Get model info
        if hasattr(model, 'model'):
            # Get parameters count
            if hasattr(model.model, 'parameters'):
                metrics.parameters = sum(p.numel() for p in model.model.parameters())
            
            # Get GFLOPs from model info
            # model.info() returns tuple: (n_layers, n_params, n_gradients, flops)
            # flops is at index 3
            if hasattr(model, 'info'):
                try:
                    info = model.info(verbose=False, imgsz=img_size)
                    if isinstance(info, tuple) and len(info) >= 4:
                        # GFLOPs is at index 3 (4th element)
                        metrics.gflops = float(info[3]) if info[3] else 0.0
                    elif isinstance(info, tuple) and len(info) >= 2:
                        # Fallback for older format
                        metrics.gflops = float(info[1]) if info[1] else 0.0
                except Exception as e:
                    logger.debug(f"Could not extract GFLOPs from model.info(): {e}")
                    # Try alternative method using get_flops directly
                    try:
                        from ultralytics.utils.torch_utils import get_flops
                        metrics.gflops = get_flops(model.model, imgsz=img_size)
                    except Exception as e2:
                        logger.debug(f"Could not extract GFLOPs using get_flops(): {e2}")
        
        logger.info(f"  • Model performance: {metrics.parameters:,} params, {metrics.gflops:.2f} GFLOPs")
        
    except Exception as e:
        logger.warning(f"Error extracting model performance: {e}")
    
    return metrics


def extract_validation_metrics(model: Any, val_results: Any, metrics: TrainingMetrics) -> TrainingMetrics:
    """
    Extract validation metrics including per-class metrics and timing.
    
    Args:
        model: The YOLO model object
        val_results: Validation results from model.val()
        metrics: TrainingMetrics object to populate
    
    Returns:
        Updated TrainingMetrics object
    """
    try:
        if val_results is None:
            return metrics
        
        # Extract timing metrics
        if hasattr(val_results, 'speed'):
            speed = val_results.speed
            metrics.preprocess_ms = float(speed.get('preprocess', 0.0))
            metrics.inference_ms = float(speed.get('inference', 0.0))
            metrics.nms_ms = float(speed.get('nms', 0.0)) if 'nms' in speed else 0.0
            metrics.postprocess_ms = float(speed.get('postprocess', 0.0))
            metrics.total_ms = metrics.preprocess_ms + metrics.inference_ms + metrics.nms_ms + metrics.postprocess_ms
            
            logger.info(f"  • Timing: pre={metrics.preprocess_ms:.2f}ms, inf={metrics.inference_ms:.2f}ms, "
                       f"nms={metrics.nms_ms:.2f}ms, post={metrics.postprocess_ms:.2f}ms, total={metrics.total_ms:.2f}ms")
        
        # Extract per-class metrics
        if hasattr(val_results, 'box'):
            box = val_results.box
            
            # Get class names
            if hasattr(val_results, 'names'):
                names = val_results.names
            elif hasattr(model, 'names'):
                names = model.names
            else:
                names = {}
            
            # Per-class AP
            if hasattr(box, 'ap50') and hasattr(box, 'ap'):
                ap50_per_class = box.ap50
                ap_per_class = box.ap
                
                # Get precision and recall per class if available
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
                
                logger.info(f"  • Extracted per-class metrics for {len(metrics.per_class_metrics)} classes")
        
    except Exception as e:
        logger.warning(f"Error extracting validation metrics: {e}")
    
    return metrics


# ==============================================================================
# COMET ML FUNCTIONS
# ==============================================================================

def setup_comet_ml(project_name: str, experiment_name: str) -> Optional[Any]:
    """
    Setup Comet ML dengan pengecekan API key.
    
    Returns:
        Comet ML experiment object atau None jika tidak tersedia.
    """
    # DISABLED: Comet upload too slow
    return None
    
    if not COMET_AVAILABLE:
        logger.info("Comet ML package tidak terinstal")
        return None
    
    # Get API key from environment variable (NEVER hardcode!)
    api_key = os.environ.get("COMET_API_KEY", "")
    if not api_key or len(api_key) < 10:
        logger.info("Comet ML: API key tidak ditemukan atau invalid")
        logger.info("  Set COMET_API_KEY environment variable untuk mengaktifkan logging")
        return None
    
    try:
        logger.info(f"✅ Comet ML: Menghubungkan ke project '{project_name}'")
        
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=None,
            log_code=True,
            log_graph=True,
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_weight_logging=True,
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_output_logging="simple",
            display_summary_level=1
        )
        
        experiment.set_name(experiment_name)
        return experiment
        
    except Exception as e:
        logger.error(f"❌ Gagal menghubungkan ke Comet ML: {e}")
        return None


def log_comet_metrics(experiment: Any, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics ke Comet ML dengan penanganan error."""
    if experiment is None:
        return
    
    try:
        for metric_name, value in metrics.items():
            if value is not None and metric_name != 'fitness':
                experiment.log_metric(metric_name, value, step=step)
    except Exception as e:
        logger.warning(f"Gagal log metrics ke Comet ML: {e}")


def log_comet_parameters(experiment: Any, params: Dict[str, Any]) -> None:
    """Log hyperparameters ke Comet ML."""
    if experiment is None:
        return
    
    try:
        experiment.log_parameters(params)
    except Exception as e:
        logger.warning(f"Gagal log parameters ke Comet ML: {e}")


def log_comet_model(experiment: Any, model_path: str, model_name: str = "YOLOv8") -> None:
    """Log model ke Comet ML."""
    if experiment is None or not Path(model_path).exists():
        return
    
    try:
        experiment.log_model(model_name, model_path)
        logger.info(f"  • Model berhasil di-log ke Comet ML: {model_name}")
    except Exception as e:
        logger.warning(f"Gagal log model ke Comet ML: {e}")


def finalize_comet_experiment(
    experiment: Any,
    config: TrainingConfig,
    model_cfg_path: str,
    pretrained_weights: str,
    model_base: str,
    fold: str,
    comet_project: str,
    duration: datetime.timedelta,
    output_dir: str,
    status: str,
    model_path: Optional[Path],
    metrics: TrainingMetrics
) -> None:
    """Finalize dan log semua data ke Comet ML experiment."""
    if experiment is None:
        return
    
    try:
        # Log hyperparameters
        hyper_params = {
            "model_cfg": model_cfg_path,
            "pretrained_weights": pretrained_weights,
            "batch_size": config.batch_size,
            "img_size": config.img_size,
            "epochs": config.epochs,
            "device": config.device,
            "workers": config.workers,
            "fold": fold,
            "comet_project": comet_project,
            "model_type": model_base
        }
        log_comet_parameters(experiment, hyper_params)
        
        # Log training metrics
        experiment.log_metric("training_status", 1 if "Success" in status else 0)
        experiment.log_metric("duration_seconds", duration.total_seconds())
        experiment.log_metric("precision", metrics.precision)
        experiment.log_metric("recall", metrics.recall)
        experiment.log_metric("f1_score", metrics.f1_score)
        experiment.log_metric("mAP50", metrics.map50)
        experiment.log_metric("mAP50_95", metrics.map50_95)
        experiment.log_metric("gflops", metrics.gflops)
        experiment.log_metric("parameters", metrics.parameters)
        
        experiment.log_other("output_dir", output_dir)
        experiment.log_other("final_status", status)
        
        # Log system info
        system_info = get_system_info()
        for key, value in system_info.items():
            experiment.log_other(key, value)
        
        # Log model if available
        if model_path and model_path.exists():
            log_comet_model(experiment, str(model_path), model_name=f"best_{fold}")
        
        # Add tags
        if "Success" in status:
            experiment.add_tag("Success")
        else:
            experiment.add_tag("Failed")
            experiment.add_tag("Error")
        
        logger.info("  • Logging ke Comet ML selesai")
        
    except Exception as e:
        logger.warning(f"Gagal menyelesaikan logging ke Comet ML: {e}")
    
    finally:
        try:
            experiment.end()
            logger.info("  • Koneksi Comet ML ditutup")
        except Exception:
            pass


# ==============================================================================
# EXCEL LOGGING
# ==============================================================================

def save_comprehensive_log(
    training_logs: List[TrainingMetrics],
    excel_file: str
) -> None:
    """
    Simpan seluruh training log ke file Excel dengan multiple sheets.
    Termasuk sheet rata-rata per model dari semua fold.
    
    Args:
        training_logs: List of TrainingMetrics objects
        excel_file: Path ke file Excel output
    """
    if not training_logs:
        logger.warning("Tidak ada log untuk disimpan")
        return
    
    try:
        # Create DataFrames for each sheet
        summary_data = [m.to_summary_dict() for m in training_logs]
        metrics_data = [m.to_metrics_dict() for m in training_logs]
        performance_data = [m.to_performance_dict() for m in training_logs]
        
        df_summary = pd.DataFrame(summary_data)
        df_metrics = pd.DataFrame(metrics_data)
        df_performance = pd.DataFrame(performance_data)
        
        # Create per-class metrics DataFrame
        per_class_rows = []
        for m in training_logs:
            for class_name, class_metrics in m.per_class_metrics.items():
                row = {
                    'Model': m.model,
                    'Fold': m.fold,
                    'Class': class_name,
                    'Precision': class_metrics.get('precision', 0.0),
                    'Recall': class_metrics.get('recall', 0.0),
                    'F1-Score': class_metrics.get('f1_score', 0.0),
                    'mAP50': class_metrics.get('mAP50', 0.0),
                    'mAP50-95': class_metrics.get('mAP50-95', 0.0),
                }
                per_class_rows.append(row)
        
        df_per_class = pd.DataFrame(per_class_rows) if per_class_rows else pd.DataFrame()
        
        # =====================================================================
        # Create AVERAGE sheets (per model, across all folds)
        # =====================================================================
        
        # Summary Average - group by Model, calculate mean for numeric columns
        summary_avg_data = []
        for model in df_summary['Model'].unique():
            model_data = df_summary[df_summary['Model'] == model]
            
            # Calculate average duration
            durations = []
            for d in model_data['Duration']:
                try:
                    td = pd.to_timedelta(d)
                    durations.append(td.total_seconds())
                except Exception:
                    pass
            avg_duration_sec = sum(durations) / len(durations) if durations else 0
            avg_duration = str(datetime.timedelta(seconds=int(avg_duration_sec)))
            
            # Calculate average last epoch
            epochs = model_data['Last Epoch'].tolist()
            avg_epoch = sum(epochs) / len(epochs) if epochs else 0
            
            # Count success rate
            success_count = sum(1 for s in model_data['Status'] if 'Success' in str(s))
            total_count = len(model_data)
            
            summary_avg_data.append({
                'Model': model,
                'Fold Count': total_count,
                'Success Count': success_count,
                'Success Rate (%)': round(success_count / total_count * 100, 2) if total_count > 0 else 0,
                'Avg Duration': avg_duration,
                'Avg Last Epoch': round(avg_epoch, 2),
            })
        
        df_summary_avg = pd.DataFrame(summary_avg_data)
        
        # Metrics Average - group by Model
        metrics_avg_data = []
        for model in df_metrics['Model'].unique():
            model_data = df_metrics[df_metrics['Model'] == model]
            metrics_avg_data.append({
                'Model': model,
                'Fold Count': len(model_data),
                'Avg Accuracy (All)': round(model_data['Accuracy (All)'].mean(), 4),
                'Std Accuracy (All)': round(model_data['Accuracy (All)'].std(), 4),
                'Avg Precision (All)': round(model_data['Precision (All)'].mean(), 4),
                'Std Precision (All)': round(model_data['Precision (All)'].std(), 4),
                'Avg Recall (All)': round(model_data['Recall (All)'].mean(), 4),
                'Std Recall (All)': round(model_data['Recall (All)'].std(), 4),
                'Avg F1-Score (All)': round(model_data['F1-Score (All)'].mean(), 4),
                'Std F1-Score (All)': round(model_data['F1-Score (All)'].std(), 4),
                'Avg mAP50 (All)': round(model_data['mAP50 (All)'].mean(), 4),
                'Std mAP50 (All)': round(model_data['mAP50 (All)'].std(), 4),
                'Avg mAP50-95 (All)': round(model_data['mAP50-95 (All)'].mean(), 4),
                'Std mAP50-95 (All)': round(model_data['mAP50-95 (All)'].std(), 4),
            })
        
        df_metrics_avg = pd.DataFrame(metrics_avg_data)
        
        # Performance Average - group by Model
        performance_avg_data = []
        for model in df_performance['Model'].unique():
            model_data = df_performance[df_performance['Model'] == model]
            performance_avg_data.append({
                'Model': model,
                'Fold Count': len(model_data),
                'GFLOPs': round(model_data['GFLOPs'].mean(), 2),
                'Parameters': int(model_data['Parameters'].mean()),
                'Avg Pre-process (ms)': round(model_data['Pre-process (ms)'].mean(), 2),
                'Std Pre-process (ms)': round(model_data['Pre-process (ms)'].std(), 2),
                'Avg Inference (ms)': round(model_data['Inference (ms)'].mean(), 2),
                'Std Inference (ms)': round(model_data['Inference (ms)'].std(), 2),
                'Avg NMS (ms)': round(model_data['NMS (ms)'].mean(), 2),
                'Std NMS (ms)': round(model_data['NMS (ms)'].std(), 2),
                'Avg Post-process (ms)': round(model_data['Post-process (ms)'].mean(), 2),
                'Std Post-process (ms)': round(model_data['Post-process (ms)'].std(), 2),
                'Avg Total (ms)': round(model_data['Total (ms)'].mean(), 2),
                'Std Total (ms)': round(model_data['Total (ms)'].std(), 2),
                'Avg Box Loss': round(model_data['Box Loss'].mean(), 4),
                'Std Box Loss': round(model_data['Box Loss'].std(), 4),
                'Avg Cls Loss': round(model_data['Cls Loss'].mean(), 4),
                'Std Cls Loss': round(model_data['Cls Loss'].std(), 4),
                'Avg DFL Loss': round(model_data['DFL Loss'].mean(), 4),
                'Std DFL Loss': round(model_data['DFL Loss'].std(), 4),
            })
        
        df_performance_avg = pd.DataFrame(performance_avg_data)
        
        # Per-Class Metrics Average - group by Model and Class
        per_class_avg_data = []
        if not df_per_class.empty:
            for model in df_per_class['Model'].unique():
                model_data = df_per_class[df_per_class['Model'] == model]
                for class_name in model_data['Class'].unique():
                    class_data = model_data[model_data['Class'] == class_name]
                    per_class_avg_data.append({
                        'Model': model,
                        'Class': class_name,
                        'Fold Count': len(class_data),
                        'Avg Precision': round(class_data['Precision'].mean(), 4),
                        'Std Precision': round(class_data['Precision'].std(), 4),
                        'Avg Recall': round(class_data['Recall'].mean(), 4),
                        'Std Recall': round(class_data['Recall'].std(), 4),
                        'Avg F1-Score': round(class_data['F1-Score'].mean(), 4),
                        'Std F1-Score': round(class_data['F1-Score'].std(), 4),
                        'Avg mAP50': round(class_data['mAP50'].mean(), 4),
                        'Std mAP50': round(class_data['mAP50'].std(), 4),
                        'Avg mAP50-95': round(class_data['mAP50-95'].mean(), 4),
                        'Std mAP50-95': round(class_data['mAP50-95'].std(), 4),
                    })
        
        df_per_class_avg = pd.DataFrame(per_class_avg_data) if per_class_avg_data else pd.DataFrame()
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Raw data sheets
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            df_performance.to_excel(writer, sheet_name='Performance', index=False)
            if not df_per_class.empty:
                df_per_class.to_excel(writer, sheet_name='Per-Class Metrics', index=False)
            
            # Average sheets
            df_summary_avg.to_excel(writer, sheet_name='Summary Average', index=False)
            df_metrics_avg.to_excel(writer, sheet_name='Metrics Average', index=False)
            df_performance_avg.to_excel(writer, sheet_name='Performance Average', index=False)
            if not df_per_class_avg.empty:
                df_per_class_avg.to_excel(writer, sheet_name='Per-Class Average', index=False)
        
        logger.info(f"Log komprehensif disimpan ke: {os.path.abspath(excel_file)}")
        logger.info(f"  • Summary sheet: {len(df_summary)} rows")
        logger.info(f"  • Metrics sheet: {len(df_metrics)} rows")
        logger.info(f"  • Performance sheet: {len(df_performance)} rows")
        logger.info(f"  • Per-Class Metrics sheet: {len(df_per_class)} rows")
        logger.info(f"  • Summary Average sheet: {len(df_summary_avg)} models")
        logger.info(f"  • Metrics Average sheet: {len(df_metrics_avg)} models")
        logger.info(f"  • Performance Average sheet: {len(df_performance_avg)} models")
        logger.info(f"  • Per-Class Average sheet: {len(df_per_class_avg)} rows")
        
    except Exception as e:
        logger.error(f"Error saat menyimpan ke Excel: {e}")
        traceback.print_exc()
    
    # Generate statistical analysis
    try:
        generate_statistical_analysis(training_logs, excel_file)
    except Exception as e:
        logger.warning(f"Error generating statistical analysis: {e}")
        traceback.print_exc()


# ==============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ==============================================================================

def generate_statistical_analysis(
    training_logs: List[TrainingMetrics],
    excel_file: str
) -> None:
    """
    Generate comprehensive statistical analysis for model comparison.
    
    Creates a statistical_analysis/ directory with:
    - mcnemar_test_results.csv
    - mcnemar_pvalue_matrix.csv
    - mcnemar_heatmap.png
    - paired_ttest_results.csv
    - friedman_test_results.txt
    - effect_size_analysis.csv
    - statistical_significance_summary.txt
    - confidence_intervals_plot.png
    
    Args:
        training_logs: List of TrainingMetrics objects
        excel_file: Path to the Excel log file (for directory reference)
    """
    if not training_logs:
        logger.warning("No training logs for statistical analysis")
        return
    
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available - skipping statistical analysis")
        return
    
    # Create output directory
    excel_path = Path(excel_file)
    stat_dir = excel_path.parent / "statistical_analysis"
    stat_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*50}")
    logger.info("GENERATING STATISTICAL ANALYSIS")
    logger.info(f"{'='*50}")
    logger.info(f"  • Output directory: {stat_dir}")
    
    # Prepare data - group by model
    df_metrics = pd.DataFrame([m.to_metrics_dict() for m in training_logs])
    models = df_metrics['Model'].unique().tolist()
    
    if len(models) < 2:
        logger.info("  • At least 2 models required for comparison - skipping")
        return
    
    # Get fold-wise metrics for each model
    model_metrics = {}
    for model in models:
        model_data = df_metrics[df_metrics['Model'] == model]
        model_metrics[model] = {
            'accuracy': model_data['Accuracy (All)'].values,
            'precision': model_data['Precision (All)'].values,
            'recall': model_data['Recall (All)'].values,
            'f1': model_data['F1-Score (All)'].values,
            'map50': model_data['mAP50 (All)'].values,
            'map50_95': model_data['mAP50-95 (All)'].values,
        }
    
    # 1. McNemar Test (approximated using accuracy differences)
    _generate_mcnemar_analysis(models, model_metrics, stat_dir)
    
    # 2. Paired t-test
    _generate_paired_ttest(models, model_metrics, stat_dir)
    
    # 3. Friedman Test
    _generate_friedman_test(models, model_metrics, stat_dir)
    
    # 4. Effect Size Analysis (Cohen's d)
    _generate_effect_size_analysis(models, model_metrics, stat_dir)
    
    # 5. Statistical Significance Summary
    _generate_significance_summary(models, model_metrics, stat_dir)
    
    # 6. Confidence Intervals Plot
    if MATPLOTLIB_AVAILABLE:
        _generate_confidence_intervals_plot(models, model_metrics, stat_dir)
    
    logger.info("  • Statistical analysis completed")


def _generate_mcnemar_analysis(
    models: List[str],
    model_metrics: Dict[str, Dict[str, np.ndarray]],
    stat_dir: Path
) -> None:
    """Generate McNemar test results and heatmap."""
    try:
        # Create pairwise comparison matrix for McNemar-like analysis
        # Since we don't have individual predictions, approximate using accuracy
        n_models = len(models)
        pvalue_matrix = np.ones((n_models, n_models))
        results = []
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:
                    continue
                
                acc1 = model_metrics[model1]['map50_95']
                acc2 = model_metrics[model2]['map50_95']
                
                # Use paired t-test as approximation for McNemar
                if len(acc1) > 1 and len(acc2) > 1 and len(acc1) == len(acc2):
                    try:
                        stat, pvalue = stats.ttest_rel(acc1, acc2)
                        pvalue_matrix[i, j] = pvalue
                        pvalue_matrix[j, i] = pvalue
                        
                        results.append({
                            'Model 1': model1,
                            'Model 2': model2,
                            'Mean Diff': np.mean(acc1) - np.mean(acc2),
                            'Statistic': stat,
                            'P-Value': pvalue,
                            'Significant (p<0.05)': 'Yes' if pvalue < 0.05 else 'No'
                        })
                    except Exception:
                        pass
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(stat_dir / 'mcnemar_test_results.csv', index=False)
            
            # Save p-value matrix
            df_pvalue = pd.DataFrame(pvalue_matrix, index=models, columns=models)
            df_pvalue.to_csv(stat_dir / 'mcnemar_pvalue_matrix.csv')
            
            # Generate heatmap
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    pvalue_matrix, 
                    annot=True, 
                    fmt='.4f',
                    xticklabels=models,
                    yticklabels=models,
                    cmap='RdYlGn_r',
                    vmin=0, 
                    vmax=0.1,
                    center=0.05
                )
                plt.title('McNemar-like Test P-Value Matrix\n(Values < 0.05 indicate significant difference)')
                plt.tight_layout()
                plt.savefig(stat_dir / 'mcnemar_heatmap.png', dpi=150)
                plt.close()
            
            logger.info("  • McNemar test results saved")
    
    except Exception as e:
        logger.warning(f"Error in McNemar analysis: {e}")


def _generate_paired_ttest(
    models: List[str],
    model_metrics: Dict[str, Dict[str, np.ndarray]],
    stat_dir: Path
) -> None:
    """Generate paired t-test results for all model pairs."""
    try:
        results = []
        metrics_to_test = ['accuracy', 'precision', 'recall', 'f1', 'map50', 'map50_95']
        
        for metric in metrics_to_test:
            for model1, model2 in combinations(models, 2):
                data1 = model_metrics[model1][metric]
                data2 = model_metrics[model2][metric]
                
                if len(data1) > 1 and len(data2) > 1 and len(data1) == len(data2):
                    try:
                        stat, pvalue = stats.ttest_rel(data1, data2)
                        results.append({
                            'Metric': metric,
                            'Model 1': model1,
                            'Model 2': model2,
                            'Mean 1': np.mean(data1),
                            'Mean 2': np.mean(data2),
                            'Mean Diff': np.mean(data1) - np.mean(data2),
                            'T-Statistic': stat,
                            'P-Value': pvalue,
                            'Significant (p<0.05)': 'Yes' if pvalue < 0.05 else 'No'
                        })
                    except Exception:
                        pass
        
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(stat_dir / 'paired_ttest_results.csv', index=False)
            logger.info("  • Paired t-test results saved")
    
    except Exception as e:
        logger.warning(f"Error in paired t-test: {e}")


def _generate_friedman_test(
    models: List[str],
    model_metrics: Dict[str, Dict[str, np.ndarray]],
    stat_dir: Path
) -> None:
    """Generate Friedman test results."""
    try:
        results_text = []
        results_text.append("FRIEDMAN TEST RESULTS")
        results_text.append("=" * 50)
        results_text.append(f"Models compared: {', '.join(models)}")
        results_text.append("")
        
        metrics_to_test = ['accuracy', 'precision', 'recall', 'f1', 'map50', 'map50_95']
        
        for metric in metrics_to_test:
            # Get data for all models
            data_arrays = []
            for model in models:
                data_arrays.append(model_metrics[model][metric])
            
            # Check if all arrays have same length
            lengths = [len(arr) for arr in data_arrays]
            if len(set(lengths)) == 1 and lengths[0] > 1:
                try:
                    stat, pvalue = stats.friedmanchisquare(*data_arrays)
                    results_text.append(f"\nMetric: {metric.upper()}")
                    results_text.append(f"  Chi-square statistic: {stat:.4f}")
                    results_text.append(f"  P-value: {pvalue:.6f}")
                    results_text.append(f"  Significant (p<0.05): {'Yes' if pvalue < 0.05 else 'No'}")
                    
                    # Add model means
                    results_text.append("  Model means:")
                    for model, data in zip(models, data_arrays):
                        results_text.append(f"    - {model}: {np.mean(data):.4f} ± {np.std(data):.4f}")
                except Exception as e:
                    results_text.append(f"\nMetric: {metric.upper()}")
                    results_text.append(f"  Error: {str(e)}")
            else:
                results_text.append(f"\nMetric: {metric.upper()}")
                results_text.append(f"  Skipped: Unequal sample sizes or insufficient data")
        
        # Write results
        with open(stat_dir / 'friedman_test_results.txt', 'w') as f:
            f.write('\n'.join(results_text))
        
        logger.info("  • Friedman test results saved")
    
    except Exception as e:
        logger.warning(f"Error in Friedman test: {e}")


def _compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def _generate_effect_size_analysis(
    models: List[str],
    model_metrics: Dict[str, Dict[str, np.ndarray]],
    stat_dir: Path
) -> None:
    """Generate effect size (Cohen's d) analysis."""
    try:
        results = []
        metrics_to_test = ['accuracy', 'precision', 'recall', 'f1', 'map50', 'map50_95']
        
        for metric in metrics_to_test:
            for model1, model2 in combinations(models, 2):
                data1 = model_metrics[model1][metric]
                data2 = model_metrics[model2][metric]
                
                if len(data1) > 1 and len(data2) > 1:
                    try:
                        cohens_d = _compute_cohens_d(data1, data2)
                        results.append({
                            'Metric': metric,
                            'Model 1': model1,
                            'Model 2': model2,
                            'Mean 1': np.mean(data1),
                            'Mean 2': np.mean(data2),
                            "Cohen's d": cohens_d,
                            'Effect Size': _interpret_cohens_d(cohens_d),
                            'Favors': model1 if cohens_d > 0 else model2
                        })
                    except Exception:
                        pass
        
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(stat_dir / 'effect_size_analysis.csv', index=False)
            logger.info("  • Effect size analysis saved")
    
    except Exception as e:
        logger.warning(f"Error in effect size analysis: {e}")


def _generate_significance_summary(
    models: List[str],
    model_metrics: Dict[str, Dict[str, np.ndarray]],
    stat_dir: Path
) -> None:
    """Generate statistical significance summary."""
    try:
        summary = []
        summary.append("STATISTICAL SIGNIFICANCE SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Models compared: {len(models)}")
        summary.append(f"Model names: {', '.join(models)}")
        summary.append("")
        
        # Overall best model based on mAP50-95
        best_model = None
        best_score = -1
        for model in models:
            mean_score = np.mean(model_metrics[model]['map50_95'])
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        
        summary.append(f"Best performing model (by mAP50-95): {best_model}")
        summary.append(f"Best mAP50-95: {best_score:.4f}")
        summary.append("")
        
        # Model rankings
        summary.append("MODEL RANKINGS (by mAP50-95):")
        summary.append("-" * 40)
        rankings = []
        for model in models:
            mean_score = np.mean(model_metrics[model]['map50_95'])
            std_score = np.std(model_metrics[model]['map50_95'])
            rankings.append((model, mean_score, std_score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        for rank, (model, mean, std) in enumerate(rankings, 1):
            summary.append(f"  {rank}. {model}: {mean:.4f} ± {std:.4f}")
        
        summary.append("")
        summary.append("SIGNIFICANT PAIRWISE DIFFERENCES (p < 0.05):")
        summary.append("-" * 40)
        
        # Check pairwise significance
        significant_pairs = []
        for model1, model2 in combinations(models, 2):
            data1 = model_metrics[model1]['map50_95']
            data2 = model_metrics[model2]['map50_95']
            
            if len(data1) > 1 and len(data2) > 1 and len(data1) == len(data2):
                try:
                    _, pvalue = stats.ttest_rel(data1, data2)
                    if pvalue < 0.05:
                        diff = np.mean(data1) - np.mean(data2)
                        better = model1 if diff > 0 else model2
                        significant_pairs.append(f"  {model1} vs {model2}: p={pvalue:.4f}, {better} is better")
                except Exception:
                    pass
        
        if significant_pairs:
            summary.extend(significant_pairs)
        else:
            summary.append("  No significant pairwise differences found")
        
        # Write summary
        with open(stat_dir / 'statistical_significance_summary.txt', 'w') as f:
            f.write('\n'.join(summary))
        
        logger.info("  • Statistical significance summary saved")
    
    except Exception as e:
        logger.warning(f"Error in significance summary: {e}")


def _generate_confidence_intervals_plot(
    models: List[str],
    model_metrics: Dict[str, Dict[str, np.ndarray]],
    stat_dir: Path
) -> None:
    """Generate confidence intervals plot for all metrics."""
    try:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'map50', 'map50_95']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP50-95']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[idx]
            
            means = []
            cis = []
            
            for model in models:
                data = model_metrics[model][metric]
                if len(data) > 1:
                    mean = np.mean(data)
                    sem = stats.sem(data)
                    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
                    means.append(mean)
                    cis.append(ci)
                else:
                    means.append(data[0] if len(data) > 0 else 0)
                    cis.append(0)
            
            x = np.arange(len(models))
            ax.bar(x, means, yerr=cis, capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(label)
            ax.set_title(f'{label}\n(95% CI)')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance with 95% Confidence Intervals', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(stat_dir / 'confidence_intervals_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("  • Confidence intervals plot saved")
    
    except Exception as e:
        logger.warning(f"Error generating confidence intervals plot: {e}")


# ==============================================================================
# KNOWLEDGE DISTILLATION HELPER FUNCTIONS
# ==============================================================================

def is_kd_model(model_base: str) -> bool:
    """Check if model requires Knowledge Distillation training."""
    return model_base in KD_CONFIG


def find_teacher_weights(model_base: str, pattern: str = None) -> Optional[Path]:
    """
    Find teacher model weights for Knowledge Distillation.
    
    Args:
        model_base: Base name of student model (to lookup in KD_CONFIG)
        pattern: Optional glob pattern override
        
    Returns:
        Path to teacher weights if found, None otherwise
    """
    import glob
    
    if model_base not in KD_CONFIG:
        return None
    
    kd_cfg = KD_CONFIG[model_base]
    search_pattern = pattern or kd_cfg.get("teacher_weights_pattern", "**/best.pt")
    
    # Search in runs/detect for teacher weights
    search_dirs = [
        Path("runs/detect"),
        Path("."),
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        matches = list(search_dir.glob(search_pattern))
        if matches:
            # Sort by modification time (newest first)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            logger.info(f"  • Found teacher weights: {matches[0]}")
            return matches[0]
    
    return None


def train_teacher_if_needed(
    config: TrainingConfig,
    model_base: str,
    fold: str,
    fold_path: Path
) -> Optional[Path]:
    """
    Train teacher model if no weights found.
    
    Args:
        config: Training configuration
        model_base: Student model base name
        fold: Current fold name
        fold_path: Path to fold data
        
    Returns:
        Path to teacher weights if available/trained, None otherwise
    """
    if model_base not in KD_CONFIG:
        return None
    
    kd_cfg = KD_CONFIG[model_base]
    
    # First try to find existing weights
    teacher_weights = find_teacher_weights(model_base)
    if teacher_weights and teacher_weights.exists():
        return teacher_weights
    
    # No weights found - train teacher first
    logger.info(f"\n{'='*60}")
    logger.info(f"KNOWLEDGE DISTILLATION: Training teacher model first")
    logger.info(f"{'='*60}")
    
    teacher_cfg = kd_cfg.get("teacher_model", "v8/yolov8s-capffn.yaml")
    teacher_cfg_path = MODEL_CFG_BASE_DIR / teacher_cfg
    teacher_base = Path(teacher_cfg).stem
    teacher_dir = Path(teacher_cfg).parent
    
    if not teacher_cfg_path.exists():
        logger.error(f"  ❌ Teacher config not found: {teacher_cfg_path}")
        return None
    
    logger.info(f"  • Teacher model: {teacher_base}")
    logger.info(f"  • Teacher config: {teacher_cfg_path}")
    
    # Train teacher (recursive call without KD)
    # Temporarily disable KD
    old_kd = config.kd_enabled
    config.kd_enabled = False
    
    teacher_metrics = train_single_fold(
        config, teacher_cfg_path, teacher_base, teacher_dir, fold, fold_path
    )
    
    config.kd_enabled = old_kd
    
    # Check if training was successful
    if "Success" in teacher_metrics.status:
        # Find the weights
        teacher_weights = find_teacher_weights(model_base)
        return teacher_weights
    
    logger.error("  ❌ Teacher training failed")
    return None


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_single_fold(
    config: TrainingConfig,
    model_cfg_path: Path,
    model_base: str,
    model_dir: Path,
    fold: str,
    fold_path: Path
) -> TrainingMetrics:
    """
    Train model pada single fold.
    
    Returns:
        TrainingMetrics dengan hasil training
    """
    metrics = TrainingMetrics(model=model_base, fold=fold)
    
    # Validate data.yml
    yaml_file = fold_path / "data.yml"
    if not yaml_file.exists():
        logger.error(f"  ❌ File data.yml tidak ditemukan di {fold_path}")
        metrics.status = "Error: data.yml missing"
        return metrics
    
    # Record start time
    start_time = get_jakarta_time()
    metrics.start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate experiment name
    experiment_name = generate_experiment_name(model_base, fold, start_time)
    comet_project = experiment_name
    
    # Output dir structure: result/experiment_NAME/fold/run_name
    # But YOLO handles 'project' as base and 'name' as subdir
    # We want: config.experiment_dir / fold / experiment_name
    # So project = config.experiment_dir / fold
    # name = experiment_name
    
    fold_specific_dir = config.experiment_dir / fold
    output_dir = fold_specific_dir / experiment_name
    metrics.output_dir = str(output_dir)
    
    logger.info(f"  • Waktu mulai: {metrics.start_time}")
    logger.info(f"  • Nama eksperimen: {experiment_name}")
    logger.info(f"  • Output Dir: {output_dir}")
    
    # Setup Comet ML
    comet_experiment = setup_comet_ml(comet_project, experiment_name)
    metrics.comet_enabled = "Yes" if comet_experiment else "No"
    
    # Log training parameters
    pretrained_weights = f"{model_base}.pt"
    logger.info(f"\nMenjalankan pelatihan menggunakan Ultralytics YOLO...")
    logger.info(f"  • Data YAML: {yaml_file}")
    logger.info(f"  • Model config: {model_cfg_path}")
    logger.info(f"  • Epochs: {config.epochs}")
    logger.info(f"  • Batch size: {config.batch_size}")
    logger.info(f"  • Image size: {config.img_size}")
    
    # Check for Knowledge Distillation
    teacher_weights = None
    is_kd = is_kd_model(model_base) and config.kd_enabled
    
    if is_kd:
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 KNOWLEDGE DISTILLATION MODE DETECTED")
        logger.info(f"{'='*60}")
        logger.info(f"  • Student model: {model_base}")
        
        kd_cfg = KD_CONFIG.get(model_base, {})
        logger.info(f"  • KD Alpha: {kd_cfg.get('alpha', config.kd_alpha)}")
        logger.info(f"  • KD Temperature: {kd_cfg.get('temperature', config.kd_temperature)}")
        
        # Find or train teacher
        teacher_weights = train_teacher_if_needed(config, model_base, fold, fold_path)
        
        if teacher_weights:
            logger.info(f"  • Teacher weights: {teacher_weights}")
        else:
            logger.warning("  ⚠️  Teacher weights not found. Training student without KD.")
            is_kd = False
    
    # Training
    status = "Success"
    model_path: Optional[Path] = None
    model = None
    results = None
    
    try:
        # Construct scaled model path if necessary (for compound scaling)
        scaled_model_path = get_scaled_model_path(model_cfg_path, model_base)
        
        logger.info(f"\nMemuat konfigurasi model: {scaled_model_path}")
        model = YOLO(scaled_model_path)
        
        # Find and load pretrained weights
        weight_path = find_pretrained_weights(model_base, model_dir)
        if weight_path:
            logger.info(f"  • Weight ditemukan di: {weight_path}")
            model = model.load(str(weight_path))
        else:
            logger.warning("  ⚠️  Weight pre-trained tidak ditemukan, menggunakan inisialisasi acak")
        
        # Extract model performance before training
        metrics = extract_model_performance(model, metrics, config.img_size)
        
        # Train
        logger.info("\nMemulai proses pelatihan...")
        
        # Prepare KD args
        train_args = {
            "data": str(yaml_file),
            "epochs": config.epochs,
            "batch": config.batch_size,
            "imgsz": config.img_size,
            "device": config.device,
            "workers": config.workers,
            "name": experiment_name,
            "project": str(fold_specific_dir), # Save inside fold dir
            "exist_ok": True,
            "patience": config.patience,
            "optimizer": config.optimizer,
            "cos_lr": config.cos_lr,
            "close_mosaic": config.close_mosaic,
            "amp": config.amp,
        }
        
        # Add teacher for KD if applicable
        if is_kd and teacher_weights:
            logger.info(f"  • Activating Knowledge Distillation with teacher: {teacher_weights}")
            train_args["teacher"] = str(teacher_weights)
            train_args["keras"] = False # Ensure pytorch mode
            
        results = model.train(**train_args)
        
        # Correct path to weights
        model_path = fold_specific_dir / experiment_name / "weights" / "best.pt"
        
        # Extract training metrics
        if results:
            metrics = extract_training_metrics(results, model, metrics)
        
        # Run validation to get detailed metrics
        logger.info("\nMenjalankan validasi untuk metrik detail...")
        try:
            val_results = model.val(data=str(yaml_file), imgsz=config.img_size)
            metrics = extract_validation_metrics(model, val_results, metrics)
        except Exception as val_e:
            logger.warning(f"Error saat validasi: {val_e}")
        
        logger.info(f"\n✅ PELATIHAN SELESAI UNTUK {fold.upper()} - MODEL: {model_base}")
        
        # Log KD summary if applicable
        if is_kd and teacher_weights:
            logger.info(f"\n📚 Knowledge Distillation Summary:")
            logger.info(f"  • Teacher model: {teacher_weights}")
            logger.info(f"  • Student model: {model_base}")
            logger.info(f"  • KD Alpha: {KD_CONFIG.get(model_base, {}).get('alpha', config.kd_alpha)}")
            logger.info(f"  • KD Temperature: {KD_CONFIG.get(model_base, {}).get('temperature', config.kd_temperature)}")
        
    except Exception as e:
        status = f"Error: {str(e)}"
        logger.error(f"\n❌ ERROR PADA {fold} - MODEL {model_base}: {e}")
        logger.debug(traceback.format_exc())
    
    # Record end time
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    metrics.end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    metrics.duration = str(duration)
    metrics.status = status
    
    # Finalize Comet ML
    finalize_comet_experiment(
        comet_experiment, config, str(model_cfg_path), pretrained_weights,
        model_base, fold, comet_project, duration, output_dir, status, model_path, metrics
    )
    
    return metrics


def main() -> None:
    """Main training loop."""
    config = TrainingConfig()
    
    # SETUP EXPERIMENT DIRECTORY (Jakarta Time)
    jakarta_time = get_jakarta_time()
    day_name = DAYS_ID[jakarta_time.weekday()]
    month_name = MONTHS_ID[jakarta_time.month]
    time_str = jakarta_time.strftime('%H_%M')
    # Folder name: eksperimen_senin_26_januari_2026_21_18
    experiment_folder_name = f"eksperimen_{day_name}_{jakarta_time.day}_{month_name}_{jakarta_time.year}_{time_str}".lower()
    
    # Create main result directory
    config.experiment_dir = Path("result") / experiment_folder_name
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Update excel log path to be inside experiment directory
    config.excel_log = str(config.experiment_dir / f"training_log_{jakarta_time.strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    # Setup FileLogging to file inside experiment dir
    log_file = config.experiment_dir / "execution.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"📍 Output Directory: {os.path.abspath(config.experiment_dir)}")
    logger.info(f"📍 Excel Log: {os.path.abspath(config.excel_log)}")
    
    # Find fold directories
    fold_dirs = get_fold_directories(config.base_data_dir)
    if not fold_dirs:
        logger.error(f"Error: Tidak ditemukan fold di {config.base_data_dir}/fold_*")
        return
    
    logger.info(f"Menemukan {len(fold_dirs)} fold: {fold_dirs}")
    
    # Training log
    training_logs: List[TrainingMetrics] = []
    
    # Loop through each model config
    for model_cfg in MODEL_CONFIGS:
        model_base = Path(model_cfg).stem
        model_dir = Path(model_cfg).parent
        model_cfg_path = MODEL_CFG_BASE_DIR / model_cfg
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"MEMULAI PELATIHAN MODEL: {model_base}")
        logger.info(f"  • Model Config Path: {model_cfg_path}")
        
        # Check for scale
        scale = get_model_scale(model_base)
        if scale:
            logger.info(f"  • Model Scale: {scale}")
        
        logger.info(f"{'#'*80}")
        
        # Validate model config exists
        if not model_cfg_path.exists():
            logger.error(f"  ❌ File konfigurasi model {model_cfg_path} tidak ditemukan")
            metrics = TrainingMetrics(
                model=model_base,
                fold="N/A",
                status="Error: Model config missing"
            )
            training_logs.append(metrics)
            continue
        
        # Loop through each fold
        for fold in fold_dirs:
            logger.info(f"\n{'='*50}")
            logger.info(f"MEMULAI PELATIHAN UNTUK {fold.upper()} - MODEL: {model_base}")
            logger.info(f"{'='*50}")
            
            fold_path = Path(config.base_data_dir).resolve() / fold
            
            # Pass experiment_dir as prefix for output
            # We modify train_single_fold to accept output_basedir if needed, 
            # OR we just handle it by setting the 'project' arg in train_single_fold
            
            # Since train_single_fold relies on 'runs/detect', we need to check if we can redirect it
            # YOLO 'project' arg sets the root dir.
            
            metrics = train_single_fold(
                config, model_cfg_path, model_base, model_dir, fold, fold_path
            )
            training_logs.append(metrics)
            
            # Save to Excel after each fold
            save_comprehensive_log(training_logs, config.excel_log)
    
    # Print summary
    if training_logs:
        logger.info(f"\n{'='*50}")
        logger.info(f"LOG PELATIHAN TERSIMPAN DI: {os.path.abspath(config.excel_log)}")
        logger.info(f"Total model: {len(MODEL_CONFIGS)}")
        logger.info(f"Total fold: {len(fold_dirs)}")
        logger.info(f"Total pelatihan: {len(training_logs)}")
        
        # Calculate statistics
        successful = [m for m in training_logs if 'Success' in m.status]
        failed = [m for m in training_logs if 'Error' in m.status]
        
        logger.info(f"Berhasil: {len(successful)}")
        logger.info(f"Gagal: {len(failed)}")
        
        # Calculate average metrics for successful runs
        if successful:
            avg_map50 = sum(m.map50 for m in successful) / len(successful)
            avg_map50_95 = sum(m.map50_95 for m in successful) / len(successful)
            avg_precision = sum(m.precision for m in successful) / len(successful)
            avg_recall = sum(m.recall for m in successful) / len(successful)
            
            logger.info(f"\nRata-rata metrik (successful runs):")
            logger.info(f"  • Precision: {avg_precision:.4f}")
            logger.info(f"  • Recall: {avg_recall:.4f}")
            logger.info(f"  • mAP50: {avg_map50:.4f}")
            logger.info(f"  • mAP50-95: {avg_map50_95:.4f}")
        
        logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()