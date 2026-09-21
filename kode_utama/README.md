# Kode Utama - YOLO/Ultralytics Customization

Repositori ini berisi kode utama untuk training dan evaluasi model YOLO dengan berbagai modifikasi dan custom architecture.

## Struktur Folder

```
kode_utama/
├── dokumentasi.md              # Dokumentasi lengkap (BACA INI!)
├── original/                   # Kode Ultralytics original (tidak dimodifikasi)
├── modified/                   # Kode yang telah dimodifikasi
│   ├── model_configs/          # Konfigurasi model custom
│   ├── nn_modules/             # Modul neural network custom
│   ├── training_scripts/       # Script training
│   ├── evaluation_scripts/     # Script evaluasi
│   └── utility_scripts/        # Script utility
├── scripts/                    # Script eksekusi utama
├── konfigurasi/                # File konfigurasi template
└── dataset_template/           # Template struktur dataset
```

## Quick Start

### 1. Training Model Standar

```bash
cd scripts
python train_custom_dataset.py --data /path/to/data.yaml --train
```

### 2. Training Model Custom (dengan modifikasi)

```bash
cd modified/training_scripts
python main.py  # Edit MODEL_CONFIGS di dalam file
```

### 3. Evaluasi Model

```bash
cd modified/evaluation_scripts
python eval_all.py --weights /path/to/best.pt --data /path/to/data.yaml
```

## Fitur Utama

- ✅ 25+ Custom model architectures (Ghost, Swin, GAM, etc.)
- ✅ 5-Fold Cross Validation support
- ✅ Multi-model training
- ✅ Comprehensive evaluation metrics
- ✅ Automatic Excel report generation
- ✅ Comet ML integration
- ✅ Grad-CAM visualization

## Dokumentasi

Baca file `dokumentasi.md` untuk:
- Penjelasan struktur folder lengkap
- Deskripsi semua file dan fungsinya
- Panduan training step-by-step
- Panduan dataset baru
- Tips dan best practices

## Model Custom Tersedia

### GhostConv Models
- yolov8-ghost.yaml
- yolov8s-ghost-capffn.yaml (Best balance)

### Swin Transformer Models
- yolov8-swin.yaml
- yolov8-ghost-swin-capffn.yaml (Full hybrid)

### Attention Models
- yolov8-swin-gam.yaml
- yolov8-ghost-eca.yaml

### Complete Custom
- yolov8-scl.yaml (Semantic Context Learning)
- yolov8-hematology.yaml

## Requirements

```bash
pip install ultralytics
pip install pandas openpyxl
pip install psutil
pip install comet-ml  # Optional
pip install matplotlib seaborn
```

## Support

Jika ada pertanyaan, silakan merujuk ke dokumentasi.md atau hubungi pengembang.

## License

AGPL-3.0 License - https://ultralytics.com/license
