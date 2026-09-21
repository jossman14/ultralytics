import kaggle
import os

# Tentukan nama dataset (format: nama_pembuat/nama_dataset)
dataset_name = "zielisme/online-gambling-logo-in-instagram-reels-video"

# Tentukan folder tempat dataset akan disimpan
download_path = "./dataset_judi_online"

print(f"Mulai mengunduh dataset: {dataset_name}...")

# Proses pengunduhan
kaggle.api.dataset_download_files(
    dataset_name, 
    path=download_path, 
    unzip=True # Ubah ke False jika Anda hanya ingin file .zip-nya saja
)

print(f"Dataset berhasil diunduh dan diekstrak di folder: {download_path}")