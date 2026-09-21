import pandas as pd

file_path = 'training_log_hasil.xlsx'
xls = pd.ExcelFile(file_path)

# Load Metrics
df_metrics = pd.read_excel(file_path, sheet_name='Metrics Average Mod')
# Keep relevant metric columns
metric_cols = ['Model', 'Avg mAP50-95 (All)', 'Avg mAP50 (All)', 'Avg Precision (All)', 'Avg Recall (All)']
df_metrics = df_metrics[metric_cols]

# Load Performance
# I need to find the correct sheet name for performance. 
# Based on previous output, it might be 'Model Performance' or similar. 
# I will search for it
perf_sheet = None
for s in xls.sheet_names:
    if 'Perform' in s or 'Spec' in s:
        perf_sheet = s
        break

if perf_sheet:
    df_perf = pd.read_excel(file_path, sheet_name=perf_sheet)
    # relevant cols: parameters, flops, model name
    # Check column names: 'Model', 'GFLOPs', 'Parameters'
    perf_cols = ['Model', 'GFLOPs', 'Parameters']
    # Filter only available columns
    perf_cols = [c for c in perf_cols if c in df_perf.columns]
    df_perf = df_perf[perf_cols]
    
    # Merge
    df_final = pd.merge(df_metrics, df_perf, on='Model', how='left')
else:
    df_final = df_metrics

# Sort by mAP
df_final = df_final.sort_values(by='Avg mAP50-95 (All)', ascending=False)

print(df_final.to_string(index=False))
