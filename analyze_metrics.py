import pandas as pd

file_path = 'training_log_hasil.xlsx'
xls = pd.ExcelFile(file_path)
print("All Sheet Names:", xls.sheet_names)

# Analyze Metrics Average Mod
if 'Metrics Average Mod' in xls.sheet_names:
    df = pd.read_excel(file_path, sheet_name='Metrics Average Mod')
    
    # Sort by mAP50-95
    # Remove std dev columns for cleaner view
    cols = [c for c in df.columns if 'Std' not in c]
    df_clean = df[cols]
    
    if 'Avg mAP50-95 (All)' in df_clean.columns:
        df_sorted = df_clean.sort_values(by='Avg mAP50-95 (All)', ascending=False)
        print("\n--- Model Ranking (by mAP50-95) ---")
        print(df_sorted.to_string(index=False))
    else:
        print("mAP50-95 column not found")
        print(df_clean.columns)

# Check for size info in other sheets
possible_size_sheets = [s for s in xls.sheet_names if 'Perform' in s or 'Spec' in s or 'Model' in s]
for s in possible_size_sheets:
    if s == 'Metrics Average Mod': continue 
    df_s = pd.read_excel(file_path, sheet_name=s)
    print(f"\n--- Checking sheet {s} for size info ---")
    print(df_s.columns.tolist())
    if any(x in str(df_s.columns).lower() for x in ['param', 'flop', 'size']):
         print(df_s.head().to_string())

