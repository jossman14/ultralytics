# Dokumentasi Pengelompokan Kode Ultralytics

## 📁 Struktur Folder

```
ultralytics/
├── kode_utama/                          # Folder utama kode sumber
│   ├── ultralytics_package/             # Package Ultralytics lengkap (169 file .py)
│   │   └── ultralytics/                 # Isi package lengkap
│   │       ├── cfg/                     # Konfigurasi model YAML
│   │       ├── engine/                  # Engine training, prediction, export
│   │       ├── models/                  # Arsitektur model YOLO
│   │       ├── nn/                      # Neural network modules
│   │       ├── utils/                   # Utility functions
│   │       ├── hub/                     # Ultralytics HUB integration
│   │       ├── data/                    # Dataset utilities
│   │       ├── solutions/               # Computer vision solutions
│   │       └── trackers/                # Object tracking
│   │
│   ├── training_scripts/                # Script training (14 file)
│   │   ├── main.py                      # Script utama cross-validation
│   │   ├── main_old.py                  # Versi lama main script
│   │   ├── main_eval.py                 # Script evaluation
│   │   ├── main copy.py                 # Backup main script
│   │   ├── train_all.py                 # Training semua model
│   │   ├── train_judol.py               # Training YOLOv12 untuk judol
│   │   ├── train_kd.py                  # Training dengan Knowledge Distillation
│   │   ├── kfold_judol_v2_1.py          # K-fold cross validation
│   │   ├── convert_dataset_yolo.py      # Konversi dataset ke YOLO format
│   │   ├── roboflow_download.py         # Download dari Roboflow
│   │   ├── kaggle_download.py           # Download dari Kaggle
│   │   └── ...                          # Script training lainnya
│   │
│   ├── evaluation_scripts/              # Script evaluasi (3 file)
│   │   ├── eval_all.py                  # Evaluasi semua model
│   │   ├── gradcam_judol.py             # GradCAM visualisasi
│   │   └── gradcam_rekap.py             # Rekap GradCAM
│   │
│   └── analysis_scripts/                # Script analisis (7 file + Excel)
│       ├── analyze_metrics.py           # Analisis metrik
│       ├── analyze_results.py           # Analisis hasil
│       ├── join_analysis.py             # Gabung analisis
│       ├── join_analysis_v2.py          # Versi 2 join analysis
│       ├── merge_results.py             # Merge hasil
│       ├── reformat_hasil.py            # Reformat hasil
│       ├── rekap.py                     # Rekapitulasi
│       ├── rekap_semua_1baris.py        # Rekap semua 1 baris
│       └── *.xlsx                       # File Excel hasil
│
└── hasil_modifikasi/                    # Hasil modifikasi custom
    ├── model_configs/                   # Konfigurasi model YAML (80 file)
    │   ├── yolo12-ghost-gam.yaml        # YOLO12 custom dengan Ghost + GAM
    │   ├── yolov8-ghost.yaml            # YOLOv8 dengan GhostNet
    │   ├── yol
    ├── model_configs/                   # Konfigurasi model YAML (80 file)
    │   ├── yolo12-ghost-gam.yaml        # YOLO12 custom dengan Ghost + GAM
    │   ├── yolov8-ghost.yaml            # YOLOv8 dengan GhostNet
    │   ├── yolov8-ghost-gam.yaml        # YOLOv8 Ghost + GAM
    │   ├── yolov8-ghost-kd.yaml         # YOLOv8 Ghost dengan KD
    │   ├── yol
