import os
import argparse
import subprocess
from ultralytics import YOLO

# Import visualizations from existing code if needed, but we'll use analyze_results.py as the primary engine
# for the advanced metrics and Advanced Analytics reporting.

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 K-Fold Training Pipeline")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLOv8 model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--cache", type=str, default="disk", help="Cache dataset (True, ram, disk, False)")
    return parser.parse_args()

def run_full_pipeline(model_weights, dataset_name, output_suffix="", **kwargs):
    """Modular function to run training and analysis for a specific model and dataset."""
    base_dir = os.path.abspath(".")
    dataset_path = os.path.join(base_dir, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} not found! Skipping...")
        return
        
    dataset_output_dir = os.path.join(base_dir, "output", dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    model_name = model_weights.replace(".pt", "")
    if output_suffix:
        model_name = f"{model_name}_{output_suffix}"
        
    model_output_dir = os.path.join(dataset_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 1. Training Loop for 5 Folds
    for fold in range(5):
        fold_name = f"fold_{fold}"
        fold_dir = os.path.join(dataset_path, fold_name)
        data_yaml = os.path.join(fold_dir, "data.yaml")
        
        if not os.path.exists(data_yaml):
            print(f"Skipping {fold_name}, data.yaml not found.")
            continue
            
        print(f"\n[PIPELINE] Training {dataset_name} - {model_name} - {fold_name}")
        
        os.chdir(fold_dir)
        try:
            model = YOLO(model_weights)
            model.train(
                data="data.yaml",
                project=model_output_dir,
                name=fold_name,
                **kwargs
            )
        finally:
            os.chdir(base_dir)
            
    # 2. Run Comprehensive Post-Training Analysis (Advanced Analytics Suite)
    print(f"\n[PIPELINE] Running Advanced Analytics Suite for {model_name}...")
    try:
        # Menjalankan analyze_results.py yang sudah mencakup:
        # GradCAM, t-SNE, Error Analysis, Sensitivity/Specificity, Per-Class Metrics, 
        # dan Statistical Analysis (Excel dengan banyak sheet).
        subprocess.run([
            "python", "analyze_results.py",
            "--dir", model_output_dir,
            "--dataset_root", dataset_path
        ], check=True)
    except Exception as e:
        print(f"Error running analysis for {model_name}: {e}")

    print(f"\n[SUCCESS] Pipeline completed for {model_name}.")
    print(f"Check results at: {model_output_dir}")

def main():
    args = parse_args()
    
    cache_val = False
    if args.cache.lower() in ["true", "ram"]:
        cache_val = True
    elif args.cache.lower() == "disk":
        cache_val = "disk"
        
    run_full_pipeline(args.model, "Judol-Detection-v2-1_5fold", epochs=args.epochs, imgsz=args.imgsz, cache=cache_val)

if __name__ == "__main__":
    main()
