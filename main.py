import os
import yaml
import datetime
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import comet_ml
import logging
import traceback
import platform
import psutil
import GPUtil
import torch

# Nonaktifkan logging Comet yang tidak perlu
try:
    comet_ml.get_logger().setLevel(logging.ERROR)
except:
    pass

def get_system_info():
    """Kumpulkan informasi sistem untuk logging"""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    # Coba dapatkan informasi GPU
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_info = []
            for gpu in gpus:
                gpu_info.append(f"{gpu.name} (VRAM: {gpu.memoryTotal}MB)")
            info["gpu"] = ", ".join(gpu_info)
        else:
            info["gpu"] = "Tidak ditemukan GPU CUDA"
    except:
        info["gpu"] = "Error mendapatkan info GPU"
    
    # Informasi PyTorch
    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_gpu_name"] = torch.cuda.get_device_name(0)
    
    return info

def setup_comet_ml(project_name, experiment_name):
    """
    Setup Comet ML dengan pengecekan API key yang ketat
    HANYA akan membuat experiment jika API key benar-benar tersedia
    """
    # Cek 1: Apakah environment variable COMET_API_KEY tersedia?
    api_key = "REDACTED_COMET_API_KEY_REMOVED"
    if not api_key or api_key.strip() == "":
        print("ℹ️  Comet ML: API key tidak ditemukan di environment variables.")
        print("    • Untuk mengaktifkan logging Comet ML, set environment variable:")
        print("      export COMET_API_KEY='your_api_key_here'")
        print("    • Pelatihan akan berjalan TANPA logging Comet ML")
        return None
    
    # Cek 2: Apakah Comet ML terinstal dengan benar?
    try:
        from comet_ml import Experiment
    except ImportError:
        print("ℹ️  Comet ML package tidak terinstal. Install dengan: pip install comet_ml")
        return None
    
    # Cek 3: Validasi API key (hanya cek panjang minimal)
    if len(api_key) < 10:
        print(f"⚠️  Comet ML: API key terdeteksi tapi terlalu pendek ({len(api_key)} karakter)")
        print("    API key Comet ML biasanya memiliki 20+ karakter. Pastikan sudah benar.")
        return None
    
    # Jika semua cek lolos, inisialisasi Comet ML
    try:
        print(f"✅ Comet ML: Menghubungkan ke project '{project_name}' dengan API key yang valid")
        
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=experiment_name,
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
        print(f"❌ Gagal menghubungkan ke Comet ML: {str(e)}")
        print("    Pelatihan akan berjalan TANPA logging Comet ML")
        return None

def log_comet_metrics(experiment, metrics, step=None):
    """Log metrics ke Comet ML dengan penanganan error"""
    if not experiment:
        return
    
    try:
        for metric_name, value in metrics.items():
            if value is not None and metric_name != 'fitness':
                experiment.log_metric(metric_name, value, step=step)
    except Exception as e:
        print(f"⚠️  Gagal log metrics ke Comet ML: {str(e)}")

def log_comet_parameters(experiment, params):
    """Log hyperparameters ke Comet ML dengan penanganan error"""
    if not experiment:
        return
    
    try:
        experiment.log_parameters(params)
    except Exception as e:
        print(f"⚠️  Gagal log parameters ke Comet ML: {str(e)}")

def log_comet_model(experiment, model_path, model_name="YOLOv8"):
    """Log model ke Comet ML dengan penanganan error"""
    if not experiment or not os.path.exists(model_path):
        return
    
    try:
        experiment.log_model(model_name, model_path)
        print(f"  • Model berhasil di-log ke Comet ML: {model_name}")
    except Exception as e:
        print(f"⚠️  Gagal log model ke Comet ML: {str(e)}")

def get_model_output_dir(project, name):
    """Dapatkan path output direktori untuk model berdasarkan project dan name"""
    # Dalam Ultralytics YOLOv8, struktur direktori output adalah:
    # {project}/{name}
    return Path(project) / name

def main():
    # Konfigurasi dasar
    BASE_DATA_DIR = "data"
    BATCH_SIZE = 32
    DEVICE = "0"
    EXCEL_LOG = "training_log.xlsx"
    
    # --- ULTRALYTICS SPECIFIC CONFIGURATION ---
    PRETRAINED_WEIGHTS = "yolov8s.pt"  # Ultralytics pre-trained weights
    MODEL_CFG = "yolov8s.yaml"          # Ultralytics model config
    # HYP_FILE = "data/hyp.scratch.custom.yaml"  # Hyperparameters file
    IMG_SIZE = 480                      # Image size (single value)
    EPOCHS = 300                         # Training epochs
    WORKERS = 8                         # Data loader workers
    model_name = MODEL_CFG.split('.')[0]
    COMET_PROJECT = f"fold-training_{model_name}"
    # ------------------------------------------
    
    # Inisialisasi log pelatihan
    training_log = []
    comet_ml.login()
    # exp = comet_ml.start()

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
        
        # --- SETUP COMET ML ---
        comet_experiment = setup_comet_ml(COMET_PROJECT, experiment_name)
        
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
        
        # --- JALANKAN PELATIHAN ---
        status = "Success"
        model_path = None
        
        try:
            # Load model configuration and pretrained weights
            model = YOLO(MODEL_CFG)
            model = model.load(PRETRAINED_WEIGHTS)
            
            # Train the model
            print("\nMemulai proses pelatihan...")
            
            # Train the model
            results = model.train(
                data=str(yaml_file),
                epochs=EPOCHS,
                batch=BATCH_SIZE,
                imgsz=IMG_SIZE,
                device=DEVICE,
                workers=WORKERS,
                name=experiment_name,
                # project=f"{COMET_PROJECT}_{fold}",
                project=experiment_name,
                exist_ok=True,
                patience=30,
                optimizer='auto',
                cos_lr=True,
                close_mosaic=10,
                amp=True
            )
            
            # Dalam Ultralytics YOLOv8, kita perlu mendapatkan save_dir dengan cara ini
            save_dir = get_model_output_dir("runs/detect", experiment_name)
            model_path = save_dir / "weights" / "best.pt"
            
            print(f"\n✅ PELATIHAN SELESAI UNTUK {fold.upper()}")
            print(f"   Hasil disimpan di: {output_dir}")
        except Exception as e:
            status = f"Error: {str(e)}"
            print(f"\n❌ ERROR PADA {fold}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        
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
            'Output Dir': output_dir,
            'Comet Enabled': "Yes" if comet_experiment else "No"
        }
        training_log.append(log_entry)
        
        # Simpan log ke Excel
        save_to_excel(log_entry, EXCEL_LOG)
        print(f"  • Log disimpan di: {os.path.abspath(EXCEL_LOG)}")
        
        # --- LOGGING KE COMET ML (HANYA JIKA COMET BERHASIL DISETUP) ---
        if comet_experiment:
            try:
                # Log hyperparameters
                hyper_params = {
                    "model_cfg": MODEL_CFG,
                    "pretrained_weights": PRETRAINED_WEIGHTS,
                    "batch_size": BATCH_SIZE,
                    "img_size": IMG_SIZE,
                    "epochs": EPOCHS,
                    "device": DEVICE,
                    "workers": WORKERS,
                    # "hyp_file": HYP_FILE,
                    "fold": fold,
                    "comet_project": COMET_PROJECT
                }
                log_comet_parameters(comet_experiment, hyper_params)
                
                # Log status pelatihan
                comet_experiment.log_metric("training_status", 1 if "Success" in status else 0)
                
                # Log durasi pelatihan
                comet_experiment.log_metric("duration_seconds", duration.total_seconds())
                
                # Log output directory
                comet_experiment.log_other("output_dir", output_dir)
                
                # Log status akhir
                comet_experiment.log_other("final_status", status)
                
                # --- GANTI log_system_info() DENGAN MANUAL LOGGING ---
                system_info = get_system_info()
                for key, value in system_info.items():
                    comet_experiment.log_other(key, value)
                
                print("  • Informasi sistem berhasil di-log ke Comet ML")
                
                # Log model jika tersedia
                if model_path and os.path.exists(model_path):
                    log_comet_model(comet_experiment, str(model_path), model_name=f"best_{fold}")
                
                # Tambahkan tag berdasarkan status
                if "Success" in status:
                    comet_experiment.add_tag("Success")
                else:
                    comet_experiment.add_tag("Failed")
                    comet_experiment.add_tag("Error")
                
                print("  • Logging ke Comet ML selesai")
                
            except Exception as e:
                print(f"⚠️  Gagal menyelesaikan logging ke Comet ML: {str(e)}")
            
            finally:
                # Pastikan experiment ditutup dengan benar
                try:
                    comet_experiment.end()
                    print("  • Koneksi Comet ML ditutup")
                except:
                    pass
    
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