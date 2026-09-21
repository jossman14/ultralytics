"""
Script untuk menggabungkan Hasil_reformatted.xlsx dengan training_log.xlsx
"""
import pandas as pd

# Baca kedua file Excel
hasil_df = pd.read_excel('Hasil_reformatted.xlsx')
training_log_df = pd.read_excel('training_log.xlsx')

print("=== Hasil_reformatted.xlsx ===")
print(f"Columns: {hasil_df.columns.tolist()}")
print(f"Shape: {hasil_df.shape}")
print(f"Models: {hasil_df['Model'].unique().tolist()}")

print("\n=== training_log.xlsx ===")
print(f"Columns: {training_log_df.columns.tolist()}")
print(f"Shape: {training_log_df.shape}")
print(f"Models: {training_log_df['Model'].unique().tolist()}")

# Agregasi training_log per model (rata-rata duration, dll)
training_agg = training_log_df.groupby('Model').agg({
    'Duration': lambda x: str(pd.to_timedelta(x).mean()).split('.')[0],  # Rata-rata duration
    'Last Epoch': 'mean',
    'Fold': 'count'  # Jumlah fold
}).reset_index()

training_agg.columns = ['Model', 'Avg Duration', 'Avg Last Epoch', 'Training Fold Count']

print("\n=== Training Log Aggregated ===")
print(training_agg)

# Gabungkan dengan hasil berdasarkan Model
# Karena nama model mungkin berbeda, kita gabungkan dengan cara:
# 1. Merge langsung berdasarkan nama yang cocok
# 2. Atau concat semua data

# Opsi 1: Merge berdasarkan nama model
merged_df = pd.merge(hasil_df, training_agg, on='Model', how='outer')

print("\n=== Merged DataFrame ===")
print(merged_df.columns.tolist())
print(merged_df)

# Simpan ke file baru
output_file = 'combined_results.xlsx'
merged_df.to_excel(output_file, index=False)
print(f"\n✓ File berhasil disimpan ke: {output_file}")

# Opsi 2: Jika ingin menggabungkan semua data dalam satu sheet tanpa merge
# Buat juga file dengan 2 sheet terpisah
with pd.ExcelWriter('combined_results_sheets.xlsx') as writer:
    hasil_df.to_excel(writer, sheet_name='Hasil Reformatted', index=False)
    training_log_df.to_excel(writer, sheet_name='Training Log', index=False)
    merged_df.to_excel(writer, sheet_name='Combined', index=False)

print(f"✓ File dengan multiple sheets disimpan ke: combined_results_sheets.xlsx")
