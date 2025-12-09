import os
import yaml
import datetime
import pandas as pd
from pathlib import Path
from ultralytics import YOLO  # Import Ultralytics YOLO
import comet_ml

def main():
    # Konfigurasi dasar - MODIFIED FOR ULTRALYTICS YOLOv8+
    BASE_DATA_DIR = "data"
    BATCH_SIZE = 32
    DEVICE = "0"
    EXCEL_LOG = "training_log.xlsx"
    COMET_PROJECT = "fold-training"  # Comet ML project name
    
    # --- ULTRALYTICS SPECIFIC CONFIGURATION ---
    PRETRAINED_WEIGHTS = "yolov8s.pt"  # Ultralytics pre-trained weights
    MODEL_CFG = "yolov8s.yaml"             # Ultralytics model config
    HYP_FILE = "data/hyp.scratch.custom.yaml"  # Hyperparameters file
    IMG_SIZE = 480                             # Image size (single value)
    EPOCHS = 1                               # Training epochs
    WORKERS = 8                                # Data loader workers
    model_name = MODEL_CFG.split('.')[0]
    COMET_PROJECT = f"{COMET_PROJECT}_{model_name}"
    # ------------------------------------------
    
    # Set Comet ML environment variables
    comet_ml.login(project_name=COMET_PROJECT)
    os.environ["COMET_PROJECT_NAME"] = COMET_PROJECT
    # os.environ["COMET_MAX_IMAGE_PREDICTIONS"] = "100"  # Log 100 image predictions
    os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "1"  # Log every batch
    os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "true"  # Enable confusion matrix
    
    # Inisialisasi log pelatihan
    training_log = []
    
    # Mapping nama hari dan bulan dalam bahasa Indonesia
    DAYS_ID = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    MONTHS_ID = ['', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    
    # Cari semua direktori fold
    fold_dirs = [
        d for d in os.listdir(BASE_DATA_DIR) 
        if os.path.isdir(os.path.join(BASE_DATA_DIR, d)) 
        and d.startswith('fold_')
    ]
    
    if not fold_dirs:
        print(f"Error: Tidak ditemukan fold di {BASE_DATA_DIR}/fold_*")
        return
    
    print(f"Menemukan {len(fold_dirs)} fold: {fold_dirs}")
    
    for fold in sorted(fold_dirs):
        print(f"\n{'='*50}")
        print(f"MEMULAI PELATIHAN UNTUK {fold.upper()}")
        print(f"{'='*50}")
        
        # Path absolut ke direktori fold
        fold_path = Path(BASE_DATA_DIR).resolve() / fold
        
        # Validasi keberadaan data.yml
        yaml_file = fold_path / "data.yml"
        if not yaml_file.exists():
            print(f"  ❌ File data.yml tidak ditemukan di {fold_path}")
            training_log.append({
                'Fold': fold,
                'Start Time': '-',
                'End Time': '-',
                'Duration': '-',
                'Status': 'Error: data.yml missing',
                'Output Dir': '-'
            })
            # Simpan log sementara meskipun error
            save_to_excel(training_log[-1], EXCEL_LOG)
            continue
        
        # --- CATAT WAKTU MULAI ---
        start_time = datetime.datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format nama eksperimen dengan timestamp
        day_name = DAYS_ID[start_time.weekday()]
        month_name = MONTHS_ID[start_time.month]
        time_str = start_time.strftime('%H_%M')
        experiment_name = f"{fold}_{day_name}_{start_time.day}_{month_name}_{start_time.year}_{time_str}"
        
        print(f"  • Waktu mulai: {start_time_str}")
        print(f"  • Nama eksperimen: {experiment_name}")
        
        # Set output directory (for logging purposes)
        output_dir = f"runs/detect/{experiment_name}"
        
        # --- SIAPKAN ARGUMEN PELATIHAN ULTRALYTICS ---
        print("\nMenjalankan pelatihan menggunakan Ultralytics YOLO...")
        print(f"  • Data YAML: {yaml_file}")
        print(f"  • Model config: {MODEL_CFG}")
        print(f"  • Pretrained weights: {PRETRAINED_WEIGHTS}")
        print(f"  • Epochs: {EPOCHS}")
        print(f"  • Batch size: {BATCH_SIZE}")
        print(f"  • Image size: {IMG_SIZE}")
        print(f"  • Device: {DEVICE}")
        print(f"  • Workers: {WORKERS}")
        print(f"  • Comet Project: {COMET_PROJECT}")
        
        # --- JALANKAN PELATIHAN ---
        status = "Success"
        
        try:
            # Load model configuration and pretrained weights
            model = YOLO(MODEL_CFG)
            model = model.load(PRETRAINED_WEIGHTS)
            
            # Train the model
            results = model.train(
                data=str(yaml_file),
                epochs=EPOCHS,
                batch=BATCH_SIZE,
                imgsz=IMG_SIZE,
                device=DEVICE,
                workers=WORKERS,
                name=experiment_name,
                project="runs/detect",  # Local directory structure
                exist_ok=True,          # Allow overwriting existing runs
                patience=30,            # Early stopping patience
                optimizer='auto',       # Automatic optimizer selection
                cos_lr=True,            # Cosine learning rate scheduler
                close_mosaic=10,        # Disable mosaic in last 10 epochs
                amp=True                # Automatic Mixed Precision
            )
            print(f"\n✅ PELATIHAN SELESAI UNTUK {fold.upper()}")
            print(f"   Hasil disimpan di: {output_dir}")
        except Exception as e:
            status = f"Error: {str(e)}"
            print(f"\n❌ ERROR PADA {fold}: {e}")
        
        # --- CATAT WAKTU SELESAI ---
        end_time = datetime.datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time
        
        # Tambahkan ke log
        log_entry = {
            'Fold': fold,
            'Start Time': start_time_str,
            'End Time': end_time_str,
            'Duration': str(duration),
            'Status': status,
            'Output Dir': output_dir
        }
        training_log.append(log_entry)
        
        # Simpan log ke Excel
        save_to_excel(log_entry, EXCEL_LOG)
        print(f"  • Log disimpan di: {os.path.abspath(EXCEL_LOG)}")
    
    # Tampilkan ringkasan
    if training_log:
        print(f"\n{'='*50}")
        print(f"LOG PELATIHAN TERSIMPAN DI: {os.path.abspath(EXCEL_LOG)}")
        print(f"Total fold: {len(training_log)}")
        
        # Hitung rata-rata durasi hanya untuk yang berhasil
        successful_durations = [
            pd.to_timedelta(entry['Duration']) 
            for entry in training_log 
            if 'Success' in entry['Status']
        ]
        
        if successful_durations:
            avg_duration = sum(successful_durations, datetime.timedelta()) / len(successful_durations)
            print(f"Rata-rata durasi pelatihan: {str(avg_duration).split('.')[0]}")
        print(f"{'='*50}")

def save_to_excel(log_entry, excel_file):
    """Simpan log entry ke file Excel dengan penanganan file yang benar"""
    df = pd.DataFrame([log_entry])
    
    # Jika file Excel belum ada, buat file baru dengan header
    if not os.path.exists(excel_file):
        df.to_excel(excel_file, sheet_name='Training Log', index=False)
    # Jika file sudah ada, append data
    else:
        try:
            # Baca file Excel yang sudah ada
            existing_df = pd.read_excel(excel_file, sheet_name='Training Log')
            # Gabungkan dengan data baru
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            # Simpan kembali
            updated_df.to_excel(excel_file, sheet_name='Training Log', index=False)
        except Exception as e:
            print(f"  ⚠️  Error saat menyimpan ke Excel: {e}")
            # Jika terjadi error, coba buat file baru
            df.to_excel(excel_file, sheet_name='Training Log', index=False)

if __name__ == "__main__":
    main()