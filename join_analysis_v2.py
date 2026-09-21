import pandas as pd

file_path = 'training_log_hasil.xlsx'
xls = pd.ExcelFile(file_path)

df_metrics = pd.read_excel(file_path, sheet_name='Metrics Average Mod')
metric_cols = ['Model', 'Avg mAP50-95 (All)', 'Avg mAP50 (All)', 'Avg Precision (All)', 'Avg Recall (All)']
df_metrics = df_metrics[metric_cols]
df_metrics['Model'] = df_metrics['Model'].astype(str).str.lower().str.strip()

perf_sheet = None
for s in xls.sheet_names:
    if 'Perform' in s or 'Spec' in s:
        perf_sheet = s
        break

if perf_sheet:
    df_perf = pd.read_excel(file_path, sheet_name=perf_sheet)
    # Mapping for common column variations
    col_map = {c: c for c in df_perf.columns}
    for c in df_perf.columns:
        if 'flop' in c.lower(): col_map[c] = 'GFLOPs'
        if 'param' in c.lower(): col_map[c] = 'Parameters'
    df_perf = df_perf.rename(columns=col_map)
    
    perf_cols = ['Model', 'GFLOPs', 'Parameters']
    perf_cols = [c for c in perf_cols if c in df_perf.columns]
    df_perf = df_perf[perf_cols]
    df_perf['Model'] = df_perf['Model'].astype(str).str.lower().str.strip()
    
    # Drop duplicates in performance to avoid fan-out
    df_perf = df_perf.drop_duplicates(subset=['Model'])
    
    df_final = pd.merge(df_metrics, df_perf, on='Model', how='left')
else:
    df_final = df_metrics

df_final = df_final.sort_values(by='Avg mAP50-95 (All)', ascending=False)
print(df_final.head(20).to_string(index=False))
