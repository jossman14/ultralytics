import os
import shutil
import random
from pathlib import Path
import yaml
from sklearn.model_selection import KFold

# Configurations
SOURCE_DIR = Path('/home/ftib/ultralytics/judol/dataset_judi_online/frames')
OUTPUT_DIR = Path('/home/ftib/ultralytics/judol/dataset_judi_online_yolo_5fold')
NUM_FOLDS = 5

CLASSES = {
    'gambling': 0,
    'nongambling': 1
}

def create_yaml_file(fold_dir):
    """Create the data.yaml file for a specific fold."""
    yaml_content = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': '', 
        'nc': len(CLASSES),
        'names': list(CLASSES.keys())
    }
    
    with open(fold_dir / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def process_and_split():
    """Process all images and split them into 5 folds."""
    # Gather all images by class
    all_images_by_class = {}
    
    for class_name, class_id in CLASSES.items():
        class_dir = SOURCE_DIR / class_name
        if not class_dir.exists():
            print(f"Directory not found: {class_dir}")
            continue
            
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            images.extend(list(class_dir.glob(f'*{ext}')))
            images.extend(list(class_dir.glob(f'*{ext.upper()}')))
            
        print(f"Found {len(images)} images for class '{class_name}'")
        # Shuffle to ensure randomness before splitting
        random.shuffle(images)
        all_images_by_class[class_id] = images

    # Initialize KFold
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # We need to compute the splits per class to maintain distribution (like StratifiedKFold logic but simpler)
    class_splits = {}
    for class_id, images in all_images_by_class.items():
        # kf.split returns generator of (train_idx, val_idx)
        splits = list(kf.split(images))
        class_splits[class_id] = splits

    # Create the output directory structure and copy files
    for fold in range(NUM_FOLDS):
        fold_dir = OUTPUT_DIR / f'fold_{fold}'
        print(f"\nProcessing Fold {fold}...")
        
        # Create directories
        for split in ['train', 'valid']:
            (fold_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (fold_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
        # Create yaml
        create_yaml_file(fold_dir)
        
        # Process each class for this fold
        for class_id, images in all_images_by_class.items():
            train_idx, val_idx = class_splits[class_id][fold]
            
            # Get the actual image paths
            train_images = [images[i] for i in train_idx]
            val_images = [images[i] for i in val_idx]
            
            _copy_and_label(train_images, fold_dir / 'train', class_id)
            _copy_and_label(val_images, fold_dir / 'valid', class_id)

def _copy_and_label(images, split_dir, class_id):
    """Copy images and create their full-image bounding box labels."""
    for img_path in images:
        # Copy image
        dest_img_path = split_dir / 'images' / img_path.name
        shutil.copy2(img_path, dest_img_path)
        
        # Create label file with full image bounding box
        label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
        dest_label_path = split_dir / 'labels' / f"{img_path.stem}.txt"
        
        with open(dest_label_path, 'w') as f:
            f.write(label_content)

def main():
    print(f"Converting classification dataset from: {SOURCE_DIR}")
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
