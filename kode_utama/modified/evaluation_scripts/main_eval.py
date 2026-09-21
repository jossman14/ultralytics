import glob
import re
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime

# Ambil semua path data.yml dan best.pt
data_yml_paths = glob.glob('data/**/data.yml', recursive=True)
# best_pt_paths = glob.glob('run/train/**/best.pt', recursive=True)
best_pt_paths = [p for p in glob.glob('run/train/**/best.pt', recursive=True) if 'yolo12' in p.lower()]

# Buat dictionary untuk menyimpan pasangan: fold -> [list best.pt]
fold_to_best_pts = {}
for pt_path in best_pt_paths:
    match = re.search(r'fold_\d+', pt_path)
    if match:
        fold_name = match.group(0)
        fold_to_best_pts.setdefault(fold_name, []).append(pt_path)

# Data structures untuk menyimpan semua hasil
all_summaries = []       # Metrik keseluruhan per model
all_per_class = []       # Metrik per kelas
all_performance = []     # GFLOPs, Parameters, Timing

# Jalankan evaluasi YOLO untuk setiap pasangan
print("\n=== Memulai Evaluasi YOLO (Python Native) ===\n")

for yml_path in data_yml_paths:
    # Ekstrak fold dari path data.yml
    match = re.search(r'data/(fold_\d+)/data\.yml', yml_path)
    if not match:
        print(f"⚠️  Skip: Tidak bisa ekstrak fold dari {yml_path}")
        continue
        
    fold_name = match.group(1)
    print(f"🔍 Evaluasi untuk fold: {fold_name} (dari {yml_path})")
    
    # Cari semua best.pt yang terkait dengan fold ini
    if fold_name not in fold_to_best_pts:
        print(f"  ❌ Tidak ada model best.pt untuk fold {fold_name}")
        continue
        
    for best_pt in fold_to_best_pts[fold_name]:
        print(f"\n  🚀 Menjalankan evaluasi dengan model: {best_pt}")
        
        try:
            # Load model
            model = YOLO(best_pt)
            
            # Ekstrak nama folder induk dari best.pt (sebelum /weights/best.pt)
            model_folder = os.path.dirname(os.path.dirname(best_pt))
            project_name = os.path.basename(model_folder)
            print(f"  📁 Project: {project_name}")
            
            # Dapatkan info model (layers, params, gradients, gflops)
            # Note: harus verbose=True karena jika False, fungsi return None
            model_info = model.info(verbose=True)
            if model_info:
                n_layers, n_params, n_gradients, gflops = model_info
            else:
                n_layers, n_params, n_gradients, gflops = 0, 0, 0, 0.0
            
            print(f"  📊 Model Info: {n_params:,} params, {gflops:.1f} GFLOPs")
            
            # Jalankan validasi
            metrics = model.val(
                data=yml_path,
                imgsz=480,
                batch=16,
                conf=0.25,
                iou=0.6,
                device="0",
                verbose=True,
                save_json=True,
                plots=True,
                save_txt=True,
                workers=8,
                half=False,
                augment=False,
                agnostic_nms=False,
                single_cls=False,
                visualize=False,
                project="eval_results",
                name=project_name
            )
            
            # Ekstrak timing metrics (speed dict)
            speed = metrics.speed  # {preprocess, inference, loss, postprocess}
            preprocess_ms = speed.get('preprocess', 0)
            inference_ms = speed.get('inference', 0)
            loss_ms = speed.get('loss', 0)
            postprocess_ms = speed.get('postprocess', 0)
            total_ms = preprocess_ms + inference_ms + loss_ms + postprocess_ms
            
            # Ekstrak metrik keseluruhan
            mAP50_95 = metrics.box.map
            mAP50 = metrics.box.map50
            mAP75 = metrics.box.map75
            mp = metrics.box.mp  # mean precision
            mr = metrics.box.mr  # mean recall
            
            # Hitung mean F1 dari array f1
            mean_f1 = float(metrics.box.f1.mean()) if len(metrics.box.f1) > 0 else 0.0
            
            # Simpan summary per model
            all_summaries.append({
                'Model': project_name,
                'Fold': fold_name,
                'GFLOPs': round(gflops, 2) if gflops else 0,
                'Parameters': n_params,
                'Mean_Precision': round(mp, 4),
                'Mean_Recall': round(mr, 4),
                'Mean_F1': round(mean_f1, 4),
                'mAP50': round(mAP50, 4),
                'mAP75': round(mAP75, 4),
                'mAP50-95': round(mAP50_95, 4),
            })
            
            # Simpan performance metrics per model
            all_performance.append({
                'Model': project_name,
                'Fold': fold_name,
                'GFLOPs': round(gflops, 2) if gflops else 0,
                'Parameters': n_params,
                'Preprocess_ms': round(preprocess_ms, 4),
                'Inference_ms': round(inference_ms, 4),
                'NMS_Loss_ms': round(loss_ms, 4),
                'Postprocess_ms': round(postprocess_ms, 4),
                'Total_ms': round(total_ms, 4),
            })
            
            # Ekstrak metrik per kelas
            class_names = metrics.names
            for i in range(len(metrics.box.p)):
                class_idx = metrics.box.ap_class_index[i] if i < len(metrics.box.ap_class_index) else i
                class_name = class_names.get(class_idx, f"class_{class_idx}")
                
                # Dapatkan metrik per kelas
                precision = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0
                recall = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0
                f1 = float(metrics.box.f1[i]) if i < len(metrics.box.f1) else 0
                ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0
                ap = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0
                
                all_per_class.append({
                    'Model': project_name,
                    'Fold': fold_name,
                    'Class': class_name,
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1': round(f1, 4),
                    'mAP50': round(ap50, 4),
                    'mAP50-95': round(ap, 4),
                })
            
            # Tampilkan ringkasan metrik
            print(f"  ✅ Evaluasi berhasil untuk {best_pt}")
            print(f"     mAP50-95: {mAP50_95:.4f}")
            print(f"     mAP50:    {mAP50:.4f}")
            print(f"     mAP75:    {mAP75:.4f}")
            print(f"     Mean Precision: {mp:.4f}")
            print(f"     Mean Recall:    {mr:.4f}")
            print(f"     Mean F1:        {mean_f1:.4f}")
            print(f"     Speed: {preprocess_ms:.1f}ms preprocess, {inference_ms:.1f}ms inference, {postprocess_ms:.1f}ms postprocess")
            
        except Exception as e:
            print(f"  ❌ Error saat evaluasi model {best_pt}: {e}\n")

# Simpan semua hasil ke Excel
if all_summaries or all_per_class or all_performance:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"eval_results_comprehensive_{timestamp}.xlsx"
    
    print(f"\n📊 Menyimpan hasil ke {excel_path}...")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Summary (per model per fold)
        if all_summaries:
            df_summary = pd.DataFrame(all_summaries)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            print(f"   ✅ Sheet 'Summary' ({len(all_summaries)} rows)")
            
            # Sheet 2: Model Average (rata-rata per model dari semua fold)
            # Ekstrak nama model base sebelum _fold_ (misalnya yolov8-ghost-gam dari yolov8-ghost-gam_fold_5_...)
            def extract_model_base(name):
                match = re.match(r'(.+?)_fold_\d+', name)
                return match.group(1) if match else name
            
            df_summary['Model_Base'] = df_summary['Model'].apply(extract_model_base)
            
            # Grup berdasarkan Model_Base dan hitung rata-rata
            numeric_cols = ['GFLOPs', 'Parameters', 'Mean_Precision', 'Mean_Recall', 
                          'Mean_F1', 'mAP50', 'mAP75', 'mAP50-95']
            df_model_avg = df_summary.groupby('Model_Base').agg({
                col: 'mean' for col in numeric_cols
            }).reset_index()
            
            # Tambahkan kolom jumlah fold
            df_model_avg['Num_Folds'] = df_summary.groupby('Model_Base').size().values
            
            # Bulatkan nilai
            for col in numeric_cols:
                if col in ['GFLOPs', 'Parameters']:
                    df_model_avg[col] = df_model_avg[col].round(2)
                else:
                    df_model_avg[col] = df_model_avg[col].round(4)
            
            # Rename kolom
            df_model_avg = df_model_avg.rename(columns={'Model_Base': 'Model'})
            
            # Reorder columns
            cols_order = ['Model', 'Num_Folds', 'GFLOPs', 'Parameters', 'Mean_Precision', 
                         'Mean_Recall', 'Mean_F1', 'mAP50', 'mAP75', 'mAP50-95']
            df_model_avg = df_model_avg[cols_order]
            
            df_model_avg.to_excel(writer, sheet_name='Model Average', index=False)
            print(f"   ✅ Sheet 'Model Average' ({len(df_model_avg)} rows)")
        
        # Sheet 3: Per-Class Metrics (per model per fold per class)
        if all_per_class:
            df_per_class = pd.DataFrame(all_per_class)
            df_per_class.to_excel(writer, sheet_name='Per-Class Metrics', index=False)
            print(f"   ✅ Sheet 'Per-Class Metrics' ({len(all_per_class)} rows)")
            
            # Sheet 4: Per-Class Average (rata-rata per model per class dari semua fold)
            df_per_class['Model_Base'] = df_per_class['Model'].apply(extract_model_base)
            
            # Grup berdasarkan Model_Base dan Class, hitung rata-rata
            class_numeric_cols = ['Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95']
            df_class_avg = df_per_class.groupby(['Model_Base', 'Class']).agg({
                col: 'mean' for col in class_numeric_cols
            }).reset_index()
            
            # Tambahkan jumlah fold
            df_class_avg['Num_Folds'] = df_per_class.groupby(['Model_Base', 'Class']).size().values
            
            # Bulatkan nilai
            for col in class_numeric_cols:
                df_class_avg[col] = df_class_avg[col].round(4)
            
            # Rename kolom
            df_class_avg = df_class_avg.rename(columns={'Model_Base': 'Model'})
            
            # Reorder columns
            cols_order = ['Model', 'Class', 'Num_Folds', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95']
            df_class_avg = df_class_avg[cols_order]
            
            df_class_avg.to_excel(writer, sheet_name='Per-Class Average', index=False)
            print(f"   ✅ Sheet 'Per-Class Average' ({len(df_class_avg)} rows)")
        
        # Sheet 5: Model Performance (timing per model per fold)
        if all_performance:
            df_performance = pd.DataFrame(all_performance)
            df_performance.to_excel(writer, sheet_name='Model Performance', index=False)
            print(f"   ✅ Sheet 'Model Performance' ({len(all_performance)} rows)")
            
            # Sheet 6: Performance Average (rata-rata timing per model dari semua fold)
            df_performance['Model_Base'] = df_performance['Model'].apply(extract_model_base)
            
            perf_numeric_cols = ['GFLOPs', 'Parameters', 'Preprocess_ms', 'Inference_ms', 
                                'NMS_Loss_ms', 'Postprocess_ms', 'Total_ms']
            df_perf_avg = df_performance.groupby('Model_Base').agg({
                col: 'mean' for col in perf_numeric_cols
            }).reset_index()
            
            # Tambahkan jumlah fold
            df_perf_avg['Num_Folds'] = df_performance.groupby('Model_Base').size().values
            
            # Bulatkan nilai
            for col in perf_numeric_cols:
                df_perf_avg[col] = df_perf_avg[col].round(4)
            
            # Rename kolom
            df_perf_avg = df_perf_avg.rename(columns={'Model_Base': 'Model'})
            
            # Reorder columns
            cols_order = ['Model', 'Num_Folds', 'GFLOPs', 'Parameters', 'Preprocess_ms', 
                         'Inference_ms', 'NMS_Loss_ms', 'Postprocess_ms', 'Total_ms']
            df_perf_avg = df_perf_avg[cols_order]
            
            df_perf_avg.to_excel(writer, sheet_name='Performance Average', index=False)
            print(f"   ✅ Sheet 'Performance Average' ({len(df_perf_avg)} rows)")
    
    print(f"\n🎉 Evaluasi selesai! Hasil disimpan di: {excel_path}")
else:
    print("\n⚠️ Tidak ada hasil evaluasi untuk disimpan.")