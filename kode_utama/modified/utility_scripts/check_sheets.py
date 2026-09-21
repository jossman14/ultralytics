import pandas as pd

file_path = 'training_log_hasil.xlsx'
try:
    xls = pd.ExcelFile(file_path)
    print("Sheet names:", xls.sheet_names)
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        print(f"\n--- Sheet: {sheet} ---")
        print("Columns:", df.columns.tolist())
        print(df.head(2))
        
except Exception as e:
    print(f"Error: {e}")
