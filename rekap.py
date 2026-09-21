import os
import pandas as pd
from collections import defaultdict

def process_yolo_models():
    main_dir = "eval_results"
    output_dir = "."  # Simpan di direktori saat ini (bisa diubah)
    
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Kumpulkan semua folder model
    subdirs = [d for d in os.listdir(main_dir) 
              if os.path.isdir(os.path.join(main_dir, d))]
    
    # 2. Kelompokkan folder berdasarkan nama model (sebelum '_fold')
    model_groups = defaultdict(list)
    for folder in subdirs:
        # Ekstraksi nama model: ambil bagian sebelum '_fold'
        if '_fold' in folder:
            model_name = folder.split('_fold')[0]
            model_groups[model_name].append(folder)
        else:
            print(f"  ⚠️ Skipping invalid folder: {folder} (missing '_fold' pattern)")
    
    # 3. Proses setiap model
    for model_name, folders in model_groups.items():
        print(f"\nProcessing model: {model_name}")
        
        # Pastikan ada tepat 5 fold
        if len(folders) != 5:
            print(f"  ⚠️ Skipping {model_name}: found {len(folders)} folds (expected 5)")
            continue
        
        # Urutkan folder berdasarkan nomor fold
        def get_fold_number(folder_name):
            """Ekstraksi nomor fold dari nama folder"""
            parts = folder_name.split('_')
            try:
                # Cari indeks 'fold' dan ambil angka setelahnya
                fold_idx = parts.index('fold')
                return int(parts[fold_idx + 1])
            except (ValueError, IndexError):
                # Jika tidak ditemukan, gunakan 0 (tapi seharusnya tidak terjadi)
                return 0
        
        folders.sort(key=get_fold_number)
        
        # --- PROSES INFERENCE SPEED ---
        stage_times = {
            'preprocess': [],
            'inference': [],
            'loss': [],
            'postprocess': []
        }
        
        for folder in folders:
            metrics_path = os.path.join(main_dir, folder, "evaluation_metrics.xlsx")
            try:
                df = pd.read_excel(metrics_path, sheet_name="Inference Speed")
                for stage in stage_times:
                    time_val = df.loc[df['Stage'] == stage, 'Time (ms)'].values[0]
                    stage_times[stage].append(time_val)
            except Exception as e:
                print(f"  ❌ Error reading {metrics_path}: {str(e)}")
                continue
        
        # Hitung rata-rata untuk inference speed
        inference_data = []
        for stage, times in stage_times.items():
            if times:  # Pastikan tidak kosong
                avg_time = sum(times) / len(times)
                inference_data.append({
                    "Stage": stage,
                    "Time (ms)": avg_time
                })
        
        # --- PROSES EVALUATION RESULTS ---
        # Ambil metadata (Images/Labels) dari fold pertama
        first_result_path = os.path.join(main_dir, folders[0], "evaluation_results.xlsx")
        class_metadata = {}
        try:
            df_first = pd.read_excel(first_result_path)
            for _, row in df_first.iterrows():
                class_metadata[row['Class']] = {
                    'Images': row['Images'],
                    'Labels': row['Labels']
                }
        except Exception as e:
            print(f"  ❌ Error reading metadata: {str(e)}")
            continue
        
        # Siapkan struktur untuk kumpulkan metrik
        metric_cols = [
            'metrics/precision(B)',
            'metrics/recall(B)',
            'metrics/mAP50(B)',
            'metrics/mAP50-95(B)'
        ]
        class_metrics = {cls: {col: [] for col in metric_cols} 
                        for cls in class_metadata.keys()}
        
        # Kumpulkan data metrik dari semua fold
        for folder in folders:
            results_path = os.path.join(main_dir, folder, "evaluation_results.xlsx")
            try:
                df = pd.read_excel(results_path)
                for cls in class_metrics:
                    row = df[df['Class'] == cls].iloc[0]
                    for col in metric_cols:
                        class_metrics[cls][col].append(row[col])
            except Exception as e:
                print(f"  ❌ Error reading {results_path}: {str(e)}")
                continue
        
        # Hitung rata-rata metrik
        results_data = []
        for cls, metadata in class_metadata.items():
            row = {
                "Class": cls,
                "Images": metadata['Images'],
                "Labels": metadata['Labels']
            }
            for col in metric_cols:
                if class_metrics[cls][col]:
                    avg_val = sum(class_metrics[cls][col]) / len(class_metrics[cls][col])
                    row[col] = avg_val
                else:
                    row[col] = None
            results_data.append(row)
        df_results = pd.DataFrame(results_data)
        
        # --- SIMPAN HASIL ---
        inference_file = os.path.join(output_dir, f"{model_name}_inference_speed_summary.xlsx")
        results_file = os.path.join(output_dir, f"{model_name}_evaluation_summary.xlsx")
        
        # Pastikan data inference tidak kosong sebelum menyimpan
        if inference_data:
            df_inference = pd.DataFrame(inference_data)
            df_inference.to_excel(inference_file, index=False)
            print(f"  ✅ Inference speed summary: {inference_file}")
        else:
            print("  ❌ No valid inference speed data found")
        
        if not df_results.empty:
            df_results.to_excel(results_file, index=False)
            print(f"  ✅ Evaluation results: {results_file}")
        else:
            print("  ❌ No valid evaluation results found")

if __name__ == "__main__":
    print("Starting YOLO model evaluation summary...")
    print(f"Current time: Thursday, December 18, 2025")
    process_yolo_models()
    print("\nSummary completed successfully!")