import pandas as pd

# Load the Excel file
file_path = 'training_log_hasil.xlsx'
try:
    df = pd.read_excel(file_path)
    
    # Display the first few rows to understand the structure
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for relevant columns for metrics and size
    # Likely columns: 'Model', 'mAP50-95', 'mAP50', 'Precision', 'Recall', 'Params', 'GFLOPs'
    # I'll look for these or similar names
    
except Exception as e:
    print(f"Error reading file: {e}")
