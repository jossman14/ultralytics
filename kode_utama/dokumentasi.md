# Dokumentasi Kode Utama - YOLO/Ultralytics Customization

## Daftar Isi
1. [Struktur Folder](#struktur-folder)
2. [Deskripsi File](#deskripsi-file)
3. [Panduan Penggunaan](#panduan-penggunaan)
4. [Training dengan Dataset Baru](#training-dengan-dataset-baru)

---

## Struktur Folder

```
kode_utama/
├── dokumentasi.md                    # File dokumentasi ini
├── original/                         # Kode original Ultralytics (tidak dimodifikasi)
│   └── ultralytics/                   # Library Ultralytics lengkap
│       ├── cfg/models/               # Konfigurasi model YOLO (v3, v5, v6, v8, v9, v10, v11, v12, rt-detr)
│       ├── nn/modules/               # Modul neural network (block.py, conv.py, head.py, transformer.py)
│       ├── models/                    # Implementasi model YOLO
│       ├── engine/                    # Engine training, validation, export
│       ├── data/                      # Data loaders, augmentasi
│       ├── utils/                     # Utility functions
│       └── solutions/                # Solusi tambahan (tracking, etc.)
│
├── modified/                          # Kode yang telah dimodifikasi/custom
│   ├── model_configs/                 # Konfigurasi model custom
│   │   ├── v3/                        # YOLOv3 configs
│   │   ├── v5/                        # YOLOv5 configs
│   │   ├── v6/                        # YOLOv6 configs
│   │   ├── v8/                        # YOLOv8 configs (original + custom)
│   │   ├── v9/                        # YOLOv9 configs
│   │   ├── v10/                       # YOLOv10 configs
│   │   ├── v11/                       # YOLOv11 configs
│   │   ├── v12/                       # YOLOv12 configs
│   │   └── rt-detr/                   # RT-DETR configs
│   │
│   ├── nn_modules/                    # Modul neural network custom
│   │   ├── block.py                    # Custom blocks (GAM_Attention, C2fGhost, C2fSwin, etc.)
│   │   ├── conv.py                     # Custom convolutions (GhostConv, ECA, etc.)
│   │   ├── head.py                     # Detection heads
│   │   └── transformer.py            # Transformer modules
│   │
│   ├── training_scripts/              # Script training
│   │   ├── main.py                     # Main training script dengan CV
│   │   ├── train_judol.py             # Training khusus dataset Judol
│   │   ├── train_all.py               # Training semua model
│   │   └── train_kd.py                # Knowledge Distillation training
│   │
│   ├── evaluation_scripts/            # Script evaluasi
│   │   ├── eval_all.py                # Evaluasi universal untuk semua framework
│   │   ├── main_eval.py               # Evaluasi main
│   │   ├── analyze_metrics.py         # Analisis metrik
│   │   └── analyze_results.py         # Analisis hasil
│   │
│   └── utility_scripts/               # Script utility
│       ├── gradcam_rekap.py           # Grad-CAM visualization
│       ├── generate_rekap_grafik.py   # Generate grafik rekap
│       ├── check_sheets.py            # Check Excel sheets
│       ├── merge_results.py            # Merge hasil evaluasi
│       └── ... (script utility lainnya)
│
├── scripts/                            # Script eksekusi utama
│
├── konfigurasi/                        # File konfigurasi umum
│
├── utils/                              # Utility functions custom
│
└── dataset_template/                   # Template struktur dataset
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

## Deskripsi File

### 1. Model Configuration Files (`modified/model_configs/`)

#### YOLOv8 Custom Models (`v8/`):

| File | Deskripsi | Modifikasi |
|------|-----------|------------|
| `yolov8-ghost.yaml` | YOLOv8 dengan GhostConv | Ghost bottleneck untuk efisiensi |
| `yolov8-ghost-gam.yaml` | YOLOv8 + Ghost + GAM | GhostConv + Global Attention Mechanism |
| `yolov8-ghost-capffn.yaml` | YOLOv8 + Ghost + CAPFFN | GhostConv + Cross-Axis Parallel FFN |
| `yolov8-ghost-swin.yaml` | YOLOv8 + Ghost + Swin | GhostConv + Swin Transformer backbone |
| `yolov8-ghost-swin-capffn.yaml` | Full Hybrid | Ghost + Swin + CAPFFN |
| `yolov8-swin.yaml` | YOLOv8 dengan Swin Transformer | Swin backbone custom |
| `yolov8-swin-timm.yaml` | YOLOv8 + Swin (timm) | Menggunakan timm pretrained weights |
| `yolov8-swin-capffn.yaml` | Swin + CAPFFN neck | Transformer backbone dengan FFN custom |
| `yolov8-swin-gam.yaml` | Swin + GAM | Transformer + Global Attention |
| `yolov8-swin-eca.yaml` | Swin + ECA | Transformer + Efficient Channel Attention |
| `yolov8-capffn.yaml` | YOLOv8 dengan CAPFFN neck | Cross-Axis Parallel FFN |
| `yolov8-scl.yaml` | SCL-YOLO complete | Semantic Context Learning |
| `yolov8-scl-gam.yaml` | SCL-YOLO + GAM | SCL dengan attention |
| `yolov8-starnet.yaml` | YOLOv8 + StarNet | StarNet backbone |
| `yolov8-ledh.yaml` | YOLOv8 + LEDHead | Lightweight Efficient Detection Head |
| `yolov8-hematology.yaml` | YOLOv8 untuk hematology | Optimized untuk sel darah |
| `yolov8s-ghost-capffn.yaml` | YOLOv8s + Ghost + CAPFFN | Small scale variant |
| `yolov8s-ghost-eca.yaml` | YOLOv8s + Ghost + ECA | Small scale dengan ECA |
| `yolov8s-ghost-kd.yaml` | YOLOv8s + Ghost + KD | Small scale dengan Knowledge Distillation |
| `yolov8-ghost-eca.yaml` | YOLOv8 + Ghost + ECA | GhostConv dengan ECA attention |
| `yolov8-ghost-kd.yaml` | YOLOv8 + Ghost + KD | Ghost dengan Knowledge Distillation |

#### YOLOv12 Models (`v12/`):

| File | Deskripsi | Modifikasi |
|------|-----------|------------|
| `yolo12-ghost-gam.yaml` | YOLOv12 + Ghost + GAM | Latest version dengan custom modules |

### 2. Neural Network Modules (`modified/nn_modules/`)

#### `block.py` - Custom Building Blocks

| Class | Fungsi | Kegunaan |
|-------|--------|----------|
| `GAM_Attention` | Global Attention Mechanism | Attention 3D untuk feature maps |
| `C2fGhost` | C2f dengan GhostBottleneck | Lightweight C2f variant |
| `C2fSwin` | C2f dengan Swin Transformer | Transformer-based C2f |
| `SwinBackbone` | Swin Transformer Backbone | Alternative to CNN backbone |
| `SwinTransformerBlock` | Swin Transformer Block | Window-based attention |
| `WindowAttention` | Window-based Multi-head Attention | Self-attention dengan windowing |
| `CFCGLU` | Cross-Feature Channel Gated Linear Unit | Custom FFN variant |
| `FBM` | Fuse Block Multi | Feature fusion block |
| `PCE` | Pyramid Context Extraction | Multi-scale context |
| `C2fAttn` | C2f dengan Attention | C2f dengan PSA block |
| `PSA` | Position-Sensitive Attention | Attention mekanisme |
| `C2PSA` | C2 dengan PSA | C2 variant dengan PSA |
| `A2C2f` | Area-Attention C2f | Area-based attention |
| `AAttn` | Area Attention | Spatial attention |
| `ABlock` | Area Block | Building block dengan attention |
| `SwiGLUFFN` | SwiGLU Feed-Forward Network | Alternative FFN dengan SwiGLU |

#### `conv.py` - Custom Convolution Modules

| Class | Fungsi | Kegunaan |
|-------|--------|----------|
| `GhostConv` | Ghost Convolution | Reduce parameters 50% |
| `ECA` | Efficient Channel Attention | Channel attention tanpa dimensi reduction |
| `CBAM` | Convolutional Block Attention Module | Channel + Spatial attention |
| `ChannelAttention` | Channel Attention | Squeeze-and-Excitation style |
| `SpatialAttention` | Spatial Attention | Spatial feature emphasis |

#### `head.py` - Detection Heads

| Class | Fungsi | Kegunaan |
|-------|--------|----------|
| `Detect` | Standard Detection Head | YOLO detection head |
| `Segment` | Segmentation Head | Instance segmentation |
| `Pose` | Pose Estimation Head | Keypoint detection |
| `OBB` | Oriented Bounding Box Head | Rotated box detection |
| `LEDHead` | Lightweight Efficient Detection Head | Custom efficient head |

### 3. Training Scripts (`modified/training_scripts/`)

#### `main.py` - Main Training Script

**Fungsi**: Script training utama dengan 5-fold cross-validation untuk semua variant model YOLO.

**Fitur**:
- Multi-model training (v3, v5, v6, v8, v9, v10, v11, v12)
- 5-Fold Cross Validation
- Logging ke Comet ML
- Export metrik ke Excel
- Automatic model scaling (n, s, m, l, x)
- Custom configuration support

**Konfigurasi Utama**:
```python
MODEL_CONFIGS = [
    "v8/yolov8s-ghost-capffn.yaml",      # Best balance
    "v8/yolov8-ghost-swin-capffn.yaml",  # Full hybrid
    # ... dan lainnya
]

TrainingConfig:
    batch_size: 16
    epochs: 100
    img_size: 480
    device: "0"           # GPU device
    optimizer: "adamW"
    patience: 30            # Early stopping patience
```

#### `train_judol.py` - Judol Dataset Training

**Fungsi**: Training khusus untuk dataset deteksi judi online dengan struktur fold-based.

**Dataset yang Didukung**:
- `dataset_judi_online_yolo_5fold`
- `Judol-Detection-v2-1_5fold`

**Output**:
- `judol/result/results_{dataset}_{timestamp}/fold_{k}/`
- Excel report dengan metrik per fold

#### `train_all.py` - Universal Training

**Fungsi**: Training semua model variant secara berurutan.

**Fitur**:
- Batch training multiple models
- Automatic result aggregation
- Resume capability

#### `train_kd.py` - Knowledge Distillation

**Fungsi**: Training dengan Knowledge Distillation dari teacher model.

**Konsep**:
- Teacher model (larger/better) → Student model (smaller)
- Transfer knowledge melalui soft labels
- Gunakan `yolov8s-ghost-kd.yaml`

### 4. Evaluation Scripts (`modified/evaluation_scripts/`)

#### `eval_all.py` - Universal Evaluation

**Fungsi**: Evaluasi model dari berbagai framework (Ultralytics, YOLOv7, YOLOv6).

**Metrik yang Dihasilkan**:
- Precision, Recall, F1-Score
- mAP50, mAP50-95
- GFLOPs, Parameters
- Inference time (preprocess, inference, NMS, postprocess)
- Per-class metrics

**Output**: Excel dengan metrik lengkap

#### `main_eval.py` - Main Evaluation

**Fungsi**: Evaluasi standar untuk model Ultralytics.

**Fitur**:
- Batch evaluation
- Multi-fold evaluation
- Automatic metric extraction

#### `analyze_metrics.py` & `analyze_results.py`

**Fungsi**: Analisis hasil training dan evaluasi.

**Output**:
- Statistical analysis
- Comparison charts
- Performance summary

### 5. Utility Scripts (`modified/utility_scripts/`)

#### `gradcam_rekap.py`

**Fungsi**: Generate Grad-CAM visualizations untuk interpretasi model.

**Fitur**:
- Heatmap generation
- Per-class visualization
- Sample selection

#### `generate_rekap_grafik.py`

**Fungsi**: Generate grafik rekap training (loss curves, metrics).

**Output**:
- Training/Validation loss plots
- Metrics per fold
- Combined visualization

#### `merge_results.py`

**Fungsi**: Merge multiple result files menjadi satu report.

#### `check_sheets.py`

**Fungsi**: Validasi dan check Excel sheets hasil training.

---

## Panduan Penggunaan

### 1. Setup Environment

```bash
# Install dependencies
cd kode_utama/original
pip install -e .

# Atau install requirements
pip install ultralytics pandas openpyxl psutil comet-ml matplotlib seaborn
```

### 2. Persiapan Dataset

Gunakan template di `dataset_template/`:

```
dataset_baru/
├── train/
│   ├── images/          # Place training images here
│   └── labels/          # Place training labels here (YOLO format)
├── valid/
│   ├── images/          # Place validation images here
│   └── labels/          # Place validation labels here
└── test/                # Optional
    ├── images/
    └── labels/
```

**Format Label YOLO**:
```
<class_id> <x_center> <y_center> <width> <height>
```

Contoh (`labels/image001.txt`):
```
0 0.5 0.5 0.3 0.4
1 0.7 0.2 0.1 0.15
```

### 3. Training Model Standar

#### Training Single Model:

```bash
cd kode_utama/modified/training_scripts

# Training YOLOv8s standar
python main.py --model v8/yolov8.yaml --scale s --data /path/to/data.yaml --epochs 100
```

#### Training dengan Cross-Validation:

```bash
# Edit main.py untuk mengatur MODEL_CONFIGS
python main.py
```

**Konfigurasi di `main.py`**:
```python
MODEL_CONFIGS = [
    "v8/yolov8.yaml",           # Standard YOLOv8
    "v8/yolov8-ghost.yaml",     # Ghost variant
]

TrainingConfig:
    batch_size = 16
    epochs = 100
    img_size = 480
```

### 4. Training Model Custom

```bash
# Training dengan konfigurasi custom
cd kode_utama/modified/training_scripts

python main.py --config ../model_configs/v8/yolov8-ghost-swin-capffn.yaml
```

### 5. Evaluasi Model

```bash
cd kode_utama/modified/evaluation_scripts

# Evaluasi semua model
python eval_all.py --data /path/to/data.yaml --weights /path/to/weights

# Evaluasi single model
python main_eval.py --weights best.pt --data data.yaml
```

### 6. Export Model

```bash
# Export ke ONNX
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.export(format='onnx')"

# Export ke TensorRT
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.export(format='engine')"
```

### 7. Inference/Prediction

```python
from ultralytics import YOLO

# Load model
model = YOLO('path/to/best.pt')

# Predict on image
results = model('path/to/image.jpg')

# Predict on video
results = model('path/to/video.mp4', save=True)

# Predict with custom confidence
results = model('image.jpg', conf=0.5, iou=0.6)
```

---

## Training dengan Dataset Baru

### Langkah 1: Siapkan Dataset

1. **Struktur Folder**:
   ```
   dataset_baru/
   ├── train/images/
   ├── train/labels/
   ├── valid/images/
   ├── valid/labels/
   ├── test/images/      # Optional
   └── test/labels/      # Optional
   ```

2. **Format Label**:
   - Format: `<class_id> <x_center> <y_center> <width> <height>`
   - Normalized (0-1)
   - Satu file `.txt` per gambar

3. **Contoh**:
   ```
   # train/labels/image001.txt
   0 0.5 0.5 0.3 0.4
   1 0.7 0.2 0.1 0.15
   ```

### Langkah 2: Buat File Konfigurasi Dataset (YAML)

Buat file `dataset_baru.yaml`:

```yaml
# dataset_baru.yaml
path: /path/to/dataset_baru  # Root direktori dataset
train: train/images          # Path relatif ke train images
val: valid/images            # Path relatif ke validation images
test: test/images            # Optional: test images

# Class names
names:
  0: person
  1: car
  2: dog
  # ... tambahkan class sesuai dataset

# Optional: URL download (jika ada)
# download: https://example.com/dataset.zip
```

**Contoh lengkap**:
```yaml
# dataset_judi_online.yaml
path: /home/user/datasets/judi_online
train: train/images
val: valid/images

names:
  0: judi_online_banner
  1: judi_online_popup
  2: iklan_judi
```

### Langkah 3: Modifikasi Training Script

Edit `main.py` atau buat script baru:

```python
# training_custom.py
import os
import yaml
from pathlib import Path
from ultralytics import YOLO

# Konfigurasi
DATA_YAML = "/path/to/dataset_baru.yaml"
MODEL_CONFIG = "v8/yolov8s-ghost-capffn.yaml"  # Pilih model
EPOCHS = 100
IMG_SIZE = 480
BATCH_SIZE = 16

# Training
model = YOLO(MODEL_CONFIG)
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device="0",
    patience=30,
    optimizer="adamW",
    cos_lr=True,
    close_mosaic=10,
)
```

### Langkah 4: Run Training

```bash
cd kode_utama/modified/training_scripts
python training_custom.py
```

### Langkah 5: Monitoring

**TensorBoard**:
```bash
tensorboard --logdir runs/detect/train
```

**Comet ML** (jika diaktifkan):
- Buka https://www.comet.ml
- Lihat real-time metrics

### Langkah 6: Evaluasi

```bash
cd kode_utama/modified/evaluation_scripts
python eval_all.py --data /path/to/dataset_baru.yaml --weights runs/detect/train/weights/best.pt
```

---

## Tips dan Best Practices

### 1. Pemilihan Model

| Skenario | Model Rekomendasi |
|----------|-------------------|
| Mobile/Edge | `yolov8n-ghost.yaml` (nano ghost) |
| Balanced | `yolov8s-ghost-capffn.yaml` |
| High Accuracy | `yolov8-ghost-swin-capffn.yaml` |
| Real-time | `yolov8n.yaml` atau `yolov8s.yaml` |
| Small Objects | Model dengan P2 layer (`yolov8-ghost-p2.yaml`) |

### 2. Hyperparameter Tuning

```python
# Di training script
TrainingConfig:
    batch_size = 16        # Naikkan jika GPU memori cukup
    img_size = 640         # Naikkan untuk small objects
    epochs = 100           # Naikkan jika underfitting
    patience = 30          # Early stopping
    lr0 = 0.01             # Initial learning rate
    lrf = 0.01             # Final learning rate fraction
    momentum = 0.937       # SGD momentum / Adam beta1
    weight_decay = 0.0005  # Optimizer weight decay
```

### 3. Multi-GPU Training

```python
# Gunakan device="0,1,2,3" untuk 4 GPU
model.train(
    data="data.yaml",
    device="0,1,2,3",
    batch=64,  # Naikkan batch size
)
```

### 4. Resume Training

```python
# Resume dari checkpoint
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)
```

### 5. Troubleshooting

**Out of Memory (OOM)**:
- Turunkan `batch_size`
- Turunkan `img_size`
- Gunakan `amp=True` (Automatic Mixed Precision)

**Overfitting**:
- Naikkan `weight_decay`
- Tambah augmentasi data
- Gunakan early stopping (`patience`)

**Underfitting**:
- Naikkan `epochs`
- Turunkan `weight_decay`
- Naikkan `img_size`

---

## Referensi

- **Ultralytics Docs**: https://docs.ultralytics.com
- **YOLO Paper**: https://arxiv.org/abs/1506.02640
- **GhostNet**: https://arxiv.org/abs/1911.11907
- **Swin Transformer**: https://arxiv.org/abs/2103.14030
- **GAM Attention**: https://arxiv.org/abs/...

---

## Changelog

**Versi 1.0** (Maret 2025):
- Initial documentation
- Dokumentasi struktur folder lengkap
- Panduan training dan evaluasi
- Template dataset

---

*Dokumentasi ini dibuat untuk memudahkan penggunaan dan pemeliharaan kode YOLO/Ultralytics yang telah dimodifikasi.*
