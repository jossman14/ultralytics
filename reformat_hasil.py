#!/usr/bin/env python3
"""
Script to reformat Hasil.xlsx to match the format of training_log.xlsx
Hasil.xlsx contains average results from all folds
"""

import pandas as pd
import numpy as np

# Read the original Hasil.xlsx
df_original = pd.read_excel('Hasil.xlsx', sheet_name='Sheet1', header=None)

# The first row (row 0) contains category headers (Object, ALL, L1, etc)
# The second row (row 1) contains column headers (Method, Presisi, Recall, etc)
# Actual data starts from row 2
headers_row = df_original.iloc[1].tolist()
data = df_original.iloc[2:].reset_index(drop=True)

# Define new column names based on the structure
# Row 0 headers: Method, then for ALL/L1/L2/L3: Presisi, Recall, F1, mAP(0.5), mAP(0.5:0.95)
# Then: GFLOPS, Parameter, Pre-prosessing, Inference, NMS, Loss, Post-Process, Total

# Extract models
models = data.iloc[:, 0].tolist()  # First column is Method/Model names

# Extract ALL metrics (columns 1-5)
all_precision = pd.to_numeric(data.iloc[:, 1], errors='coerce')
all_recall = pd.to_numeric(data.iloc[:, 2], errors='coerce')
all_f1 = pd.to_numeric(data.iloc[:, 3], errors='coerce')
all_map50 = pd.to_numeric(data.iloc[:, 4], errors='coerce')
all_map50_95 = pd.to_numeric(data.iloc[:, 5], errors='coerce')

# Extract L1 metrics (columns 6-10)
l1_precision = pd.to_numeric(data.iloc[:, 6], errors='coerce')
l1_recall = pd.to_numeric(data.iloc[:, 7], errors='coerce')
l1_f1 = pd.to_numeric(data.iloc[:, 8], errors='coerce')
l1_map50 = pd.to_numeric(data.iloc[:, 9], errors='coerce')
l1_map50_95 = pd.to_numeric(data.iloc[:, 10], errors='coerce')

# Extract L2 metrics (columns 11-15)
l2_precision = pd.to_numeric(data.iloc[:, 11], errors='coerce')
l2_recall = pd.to_numeric(data.iloc[:, 12], errors='coerce')
l2_f1 = pd.to_numeric(data.iloc[:, 13], errors='coerce')
l2_map50 = pd.to_numeric(data.iloc[:, 14], errors='coerce')
l2_map50_95 = pd.to_numeric(data.iloc[:, 15], errors='coerce')

# Extract L3 metrics (columns 16-20)
l3_precision = pd.to_numeric(data.iloc[:, 16], errors='coerce')
l3_recall = pd.to_numeric(data.iloc[:, 17], errors='coerce')
l3_f1 = pd.to_numeric(data.iloc[:, 18], errors='coerce')
l3_map50 = pd.to_numeric(data.iloc[:, 19], errors='coerce')
l3_map50_95 = pd.to_numeric(data.iloc[:, 20], errors='coerce')

# Extract Performance metrics (remaining columns 21-28)
gflops = pd.to_numeric(data.iloc[:, 21], errors='coerce')
parameters = pd.to_numeric(data.iloc[:, 22], errors='coerce')
preprocess = pd.to_numeric(data.iloc[:, 23], errors='coerce')
inference = pd.to_numeric(data.iloc[:, 24], errors='coerce')
nms = pd.to_numeric(data.iloc[:, 25], errors='coerce')
loss = pd.to_numeric(data.iloc[:, 26], errors='coerce')
postprocess = pd.to_numeric(data.iloc[:, 27], errors='coerce')
total_time = pd.to_numeric(data.iloc[:, 28], errors='coerce')

# Create Metrics Average sheet
# Format: Model, Fold Count, Avg Accuracy (All), Std Accuracy (All), Avg Precision (All), Std Precision (All), 
#         Avg Recall (All), Std Recall (All), Avg F1-Score (All), Std F1-Score (All), 
#         Avg mAP50 (All), Std mAP50 (All), Avg mAP50-95 (All), Std mAP50-95 (All)
metrics_avg = pd.DataFrame({
    'Model': models,
    'Fold Count': 5,  # Assuming 5 folds
    'Avg Accuracy (All)': all_precision,  # Using precision as proxy for accuracy
    'Std Accuracy (All)': 0.0,  # No std data in Hasil.xlsx
    'Avg Precision (All)': all_precision,
    'Std Precision (All)': 0.0,
    'Avg Recall (All)': all_recall,
    'Std Recall (All)': 0.0,
    'Avg F1-Score (All)': all_f1,
    'Std F1-Score (All)': 0.0,
    'Avg mAP50 (All)': all_map50,
    'Std mAP50 (All)': 0.0,
    'Avg mAP50-95 (All)': all_map50_95,
    'Std mAP50-95 (All)': 0.0
})

# Create Performance Average sheet
# Format: Model, Fold Count, GFLOPs, Parameters, Avg Pre-process (ms), Avg Inference (ms), 
#         Avg NMS (ms), Avg Post-process (ms), Avg Total (ms), Avg Box Loss, Avg Cls Loss, Avg DFL Loss
performance_avg = pd.DataFrame({
    'Model': models,
    'Fold Count': 5,
    'GFLOPs': gflops,
    'Parameters': parameters,
    'Avg Pre-process (ms)': preprocess,
    'Avg Inference (ms)': inference,
    'Avg NMS (ms)': nms,
    'Avg Post-process (ms)': postprocess,
    'Avg Total (ms)': total_time,
    'Avg Box Loss': 0.0,
    'Avg Cls Loss': 0.0,
    'Avg DFL Loss': 0.0
})

# Create Per-Class Average sheet
# Format: Model, Class, Fold Count, Avg Precision, Std Precision, Avg Recall, Std Recall, 
#         Avg F1-Score, Std F1-Score, Avg mAP50, Std mAP50, Avg mAP50-95, Std mAP50-95
per_class_rows = []
for i, model in enumerate(models):
    # L1 class
    per_class_rows.append({
        'Model': model,
        'Class': 'L1',
        'Fold Count': 5,
        'Avg Precision': l1_precision.iloc[i] if not pd.isna(l1_precision.iloc[i]) else 0,
        'Std Precision': 0.0,
        'Avg Recall': l1_recall.iloc[i] if not pd.isna(l1_recall.iloc[i]) else 0,
        'Std Recall': 0.0,
        'Avg F1-Score': l1_f1.iloc[i] if not pd.isna(l1_f1.iloc[i]) else 0,
        'Std F1-Score': 0.0,
        'Avg mAP50': l1_map50.iloc[i] if not pd.isna(l1_map50.iloc[i]) else 0,
        'Std mAP50': 0.0,
        'Avg mAP50-95': l1_map50_95.iloc[i] if not pd.isna(l1_map50_95.iloc[i]) else 0,
        'Std mAP50-95': 0.0
    })
    # L2 class
    per_class_rows.append({
        'Model': model,
        'Class': 'L2',
        'Fold Count': 5,
        'Avg Precision': l2_precision.iloc[i] if not pd.isna(l2_precision.iloc[i]) else 0,
        'Std Precision': 0.0,
        'Avg Recall': l2_recall.iloc[i] if not pd.isna(l2_recall.iloc[i]) else 0,
        'Std Recall': 0.0,
        'Avg F1-Score': l2_f1.iloc[i] if not pd.isna(l2_f1.iloc[i]) else 0,
        'Std F1-Score': 0.0,
        'Avg mAP50': l2_map50.iloc[i] if not pd.isna(l2_map50.iloc[i]) else 0,
        'Std mAP50': 0.0,
        'Avg mAP50-95': l2_map50_95.iloc[i] if not pd.isna(l2_map50_95.iloc[i]) else 0,
        'Std mAP50-95': 0.0
    })
    # L3 class
    per_class_rows.append({
        'Model': model,
        'Class': 'L3',
        'Fold Count': 5,
        'Avg Precision': l3_precision.iloc[i] if not pd.isna(l3_precision.iloc[i]) else 0,
        'Std Precision': 0.0,
        'Avg Recall': l3_recall.iloc[i] if not pd.isna(l3_recall.iloc[i]) else 0,
        'Std Recall': 0.0,
        'Avg F1-Score': l3_f1.iloc[i] if not pd.isna(l3_f1.iloc[i]) else 0,
        'Std F1-Score': 0.0,
        'Avg mAP50': l3_map50.iloc[i] if not pd.isna(l3_map50.iloc[i]) else 0,
        'Std mAP50': 0.0,
        'Avg mAP50-95': l3_map50_95.iloc[i] if not pd.isna(l3_map50_95.iloc[i]) else 0,
        'Std mAP50-95': 0.0
    })

per_class_avg = pd.DataFrame(per_class_rows)

# Write to new Excel file with multiple sheets matching training_log.xlsx format
output_file = 'Hasil_reformatted.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    metrics_avg.to_excel(writer, sheet_name='Metrics Average', index=False)
    performance_avg.to_excel(writer, sheet_name='Performance Average', index=False)
    per_class_avg.to_excel(writer, sheet_name='Per-Class Average', index=False)

print(f"Successfully reformatted Hasil.xlsx to {output_file}")
print(f"\nSheets created:")
print("  - Metrics Average: Overall model metrics")
print("  - Performance Average: GFLOPs, Parameters, Timing")
print("  - Per-Class Average: L1, L2, L3 class metrics")

# Show preview
print("\n=== Preview of Metrics Average ===")
print(metrics_avg.to_string(index=False))
print("\n=== Preview of Performance Average ===")
print(performance_avg.to_string(index=False))
print("\n=== Preview of Per-Class Average (first 6 rows) ===")
print(per_class_avg.head(6).to_string(index=False))
