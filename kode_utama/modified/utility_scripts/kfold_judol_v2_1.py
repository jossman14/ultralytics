import os
import shutil
import random
from pathlib import Path
import yaml
from sklearn.model_selection import KFold

# Configurations
SOURCE_DIR = Path('/home/ftib/ultralytics/judol/Judol-Detection-v2-1')
OUTPUT_DIR = Path('/home/ftib/ultralytics/judol/Judol-Detection-v2-1_5fold')
NUM_FOLDS = 5

def create_yaml_file(fold_dir, original_yaml_path):
    """Copy and modify the data.yaml file for a specific fold."""
    with open(original_yaml_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
        
    yaml_content['train'] = 'train/images'
    yaml_content['val'] = 'valid/images'
    if 'test' in yaml_content:
        yaml_content['test'] = ''
        
    with open(fold_dir / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def get_all_data():
    """Gather all images and their corresponding labels from train, valid, and test."""
    all_data = [] # List of tuples: (img_path, label_path)
    
    for split in ['train', 'valid', 'test']:
        img_dir = SOURCE_DIR / split / 'images'
        lbl_dir = SOURCE_DIR / split / 'labels'
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        for img_path in img_dir.glob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                continue
                
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                all_data.append((img_path, label_path))
                
    return all_data

def process_and_split():
    """Process all data and split into 5 folds."""
    all_data = get_all_data()
    print(f"Found {len(all_data)} valid image-label pairs across all splits.")
    
    if not all_data:
        print("No data found!")
        return

    # Shuffle to ensure randomness before splitting
    random.seed(42)
    random.shuffle(all_data)

    # Initialize KFold
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    splits = list(kf.split(all_data))

    # Create the output directory structure and copy files
    original_yaml = SOURCE_DIR / 'data.yaml'

    for fold in range(NUM_FOLDS):
        fold_dir = OUTPUT_DIR / f'fold_{fold}'
        print(f"\nProcessing Fold {fold}...")
        
        # Create directories
        for split in ['train', 'valid']:
            (fold_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (fold_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
        if original_yaml.exists():
            create_yaml_file(fold_dir, original_yaml)
        else:
            print("Warning: data.yaml not found in source directory.")
            
        train_idx, val_idx = splits[fold]
        
        # Copy Train Data
        for i in train_idx:
            img_path, lbl_path = all_data[i]
            shutil.copy2(img_path, fold_dir / 'train' / 'images' / img_path.name)
            shutil.copy2(lbl_path, fold_dir / 'train' / 'labels' / lbl_path.name)
            
        # Copy Valid Data
        for i in val_idx:
            img_path, lbl_path = all_data[i]
            shutil.copy2(img_path, fold_dir / 'valid' / 'images' / img_path.name)
            shutil.copy2(lbl_path, fold_dir / 'valid' / 'labels' / lbl_path.name)
            
        print(f"Fold {fold}: {len(train_idx)} train samples, {len(val_idx)} validation samples.")

def main():
    print(f"Restructuring dataset from: {SOURCE_DIR}")
    print(f"To YOLO 5-Fold format at: {OUTPUT_DIR}")
    
    if OUTPUT_DIR.exists():
        print("Cleaning up existing output directory...")
        shutil.rmtree(OUTPUT_DIR)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    process_and_split()
        
    print("\nConversion complete!")
    print(f"Dataset structure generated at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
