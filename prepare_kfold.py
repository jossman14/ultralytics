import os
import shutil
import yaml
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import argparse

def prepare_kfold(source_dir, output_dir, n_splits=5, seed=42):
    """
    Mengonversi dataset YOLO standar menjadi format 5-Fold Cross Validation.
    Mendukung struktur rekursif (misal: train/images, valid/images).
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Temukan semua gambar secara rekursif
    all_img_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        all_img_files.extend(list(source_dir.rglob(ext)))
    
    # Filter: pastikan gambar ada di folder 'images' (standar YOLO)
    img_files = [f for f in all_img_files if "images" in str(f.parent)]
    
    if not img_files:
        print(f"Error: Tidak ada gambar ditemukan di dalam folder 'images' pada {source_dir}")
        return

    print(f"Total gambar ditemukan: {len(img_files)}")
    
    # Ambil semua file label secara rekursif untuk pencarian cepat
    all_lbl_files = list(source_dir.rglob("*.txt"))
    lbl_map = {f.stem: f for f in all_lbl_files if "labels" in str(f.parent)}

    # --- NEW: STRATIFICATION LOGIC ---
    print("Menganalisis distribusi kelas untuk Stratifikasi...")
    image_labels = []
    class_counts = {}
    
    # 1. Hitung frekuensi global kelas
    for stem, lbl_path in lbl_map.items():
        with open(lbl_path, 'r') as f:
            for line in f:
                cls = int(line.split()[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1
                
    # 2. Untuk setiap gambar, tentukan "stratification class" (gunakan kelas paling langka di gambar tersebut)
    for img_path in img_files:
        lbl_path = lbl_map.get(img_path.stem)
        if lbl_path:
            with open(lbl_path, 'r') as f:
                img_classes = list(set([int(line.split()[0]) for line in f]))
            if img_classes:
                # Pilih kelas dengan frekuensi global terendah (paling langka)
                rarest_cls = min(img_classes, key=lambda c: class_counts.get(c, 0))
                image_labels.append(rarest_cls)
            else:
                image_labels.append(-1) # Background/No objects
        else:
            image_labels.append(-1)

    # Baca data.yaml asli
    original_yaml = source_dir / "data.yaml"
    if original_yaml.exists():
        with open(original_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', [])
            nc = data_config.get('nc', len(class_names))
    else:
        print("Warning: data.yaml tidak ditemukan.")
        nc = int(input("Masukkan jumlah kelas (nc): "))
        class_names = [str(i) for i in range(nc)]

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(img_files))
    image_labels = np.array(image_labels)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, image_labels)):
        print(f"\nProcessing Fold {fold}...")
        fold_dir = output_dir / f"fold_{fold}"
        
        for split in ["train", "valid"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            
        def copy_files(indices, split):
            count = 0
            for i in indices:
                img_path = img_files[i]
                lbl_path = lbl_map.get(img_path.stem)
                
                if lbl_path and lbl_path.exists():
                    shutil.copy(img_path, fold_dir / split / "images" / img_path.name)
                    shutil.copy(lbl_path, fold_dir / split / "labels" / lbl_path.name)
                    count += 1
            return count

        c_train = copy_files(train_idx, "train")
        c_val = copy_files(val_idx, "valid")
        print(f"  Fold {fold}: Train={c_train}, Valid={c_val}")
        
        # Buat data.yaml untuk fold ini
        fold_config = {
            'path': str(fold_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': nc,
            'names': class_names
        }
        
        with open(fold_dir / "data.yaml", 'w') as f:
            yaml.dump(fold_config, f, default_flow_style=False)
            
    print(f"\n[SUCCESS] Dataset 5-Fold berhasil dibuat di: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO to K-Fold Converter")
    parser.add_argument("--source", type=str, required=True, help="Path ke dataset asli (berisi images/ dan labels/)")
    parser.add_argument("--output", type=str, required=True, help="Path ke folder output K-Fold")
    parser.add_argument("--folds", type=int, default=5, help="Jumlah fold (default: 5)")
    
    args = parser.parse_args()
    prepare_kfold(args.source, args.output, args.folds)
