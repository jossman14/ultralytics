import os
import pandas as pd
import re
from collections import defaultdict

def parse_model_summary(text):
    """Parse GFLOPS, parameters, and layers from model summary text"""
    layers = parameters = gflops = None
    
    # Extract layers
    layer_match = re.search(r'(\d+)\s+layers', text)
    if layer_match:
        layers = int(layer_match.group(1))
    
    # Extract parameters (handle commas)
    param_match = re.search(r'([\d,]+)\s+parameters', text)
    if param_match:
        param_str = param_match.group(1).replace(',', '')
        parameters = int(param_str)
    
    # Extract GFLOPS
    gflops_match = re.search(r'([\d.]+)\s+GFLOPs', text)
    if gflops_match:
        gflops = float(gflops_match.group(1))
    
    return layers, parameters, gflops

def calculate_f1(precision, recall):
    """Calculate F1 score from precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def generate_model_comparison():
    current_dir = "."
    output_file = "model_comparison_summary.xlsx"
    
    # Get all evaluation summary files
    eval_files = [f for f in os.listdir(current_dir) 
                 if f.endswith('_evaluation_summary.xlsx')]
    
    if not eval_files:
        print("⚠️ No evaluation summary files found!")
        return
    
    all_rows = []
    required_classes = ['all', 'L1', 'L2', 'L3']
    required_stages = ['preprocess', 'inference', 'loss', 'postprocess']
    
    print(f"Found {len(eval_files)} models to process...")
    
    for eval_file in eval_files:
        model_name = eval_file.replace('_evaluation_summary.xlsx', '')
        print(f"\nProcessing model: {model_name}")
        
        # Get matching files
        inference_file = f"{model_name}_inference_speed_summary.xlsx"
        txt_files = [f for f in os.listdir(current_dir) if f.endswith('.txt')]
        matching_txt = next((f for f in txt_files 
                           if os.path.splitext(f)[0].lower() == model_name.lower()), None)
        
        # Validate required files exist
        missing_files = []
        if not os.path.exists(inference_file):
            missing_files.append("inference speed summary")
        if not matching_txt:
            missing_files.append("model summary text file")
        
        if missing_files:
            print(f"  ⚠️ Skipping {model_name}. Missing: {', '.join(missing_files)}")
            continue
        
        try:
            # Read evaluation metrics
            eval_df = pd.read_excel(eval_file)
            infer_df = pd.read_excel(inference_file)
            
            # Validate required classes exist
            missing_classes = [cls for cls in required_classes 
                              if cls not in eval_df['Class'].values]
            if missing_classes:
                print(f"  ⚠️ Skipping {model_name}. Missing classes: {', '.join(missing_classes)}")
                continue
            
            # Validate required stages exist
            missing_stages = [stage for stage in required_stages 
                             if stage not in infer_df['Stage'].values]
            if missing_stages:
                print(f"  ⚠️ Skipping {model_name}. Missing stages: {', '.join(missing_stages)}")
                continue
            
            # Process class metrics
            class_metrics = {}
            for cls in required_classes:
                cls_row = eval_df[eval_df['Class'] == cls].iloc[0]
                precision = cls_row['metrics/precision(B)']
                recall = cls_row['metrics/recall(B)']
                map50 = cls_row['metrics/mAP50(B)']
                map50_95 = cls_row['metrics/mAP50-95(B)']
                f1 = calculate_f1(precision, recall)
                
                class_metrics[cls] = {
                    'precision': f"{precision}",
                    'recall': f"{recall}",
                    'f1': f"{f1}",
                    'map50': f"{map50}",
                    'map50_95': f"{map50_95}"
                }
            
            # Process inference times
            stage_times = {}
            for stage in required_stages:
                time_val = infer_df[infer_df['Stage'] == stage]['Time (ms)'].values[0]
                stage_times[stage] = time_val
            
            total_time = sum(stage_times.values())
            total_time = total_time
            
            # Process model architecture info
            with open(matching_txt, 'r') as f:
                summary_text = f.read()
            
            # Extract model name from text file
            first_line = summary_text.strip().split('\n')[0]
            method_name = first_line.split()[0] if first_line else model_name
            
            # Parse architecture metrics
            layers, parameters, gflops = parse_model_summary(summary_text)
            if gflops is not None:
                gflops = gflops
            
            # Build result row
            row_data = {
                "Method": method_name,
                "ALL Presisi": class_metrics['all']['precision'],
                "ALL Recall": class_metrics['all']['recall'],
                "ALL F1": class_metrics['all']['f1'],
                "ALL mAP (0.5)": class_metrics['all']['map50'],
                "ALL mAP (0.5:0.95)": class_metrics['all']['map50_95'],
                "L1 Presisi": class_metrics['L1']['precision'],
                "L1 Recall": class_metrics['L1']['recall'],
                "L1 F1": class_metrics['L1']['f1'],
                "L1 mAP (0.5)": class_metrics['L1']['map50'],
                "L1 mAP (0.5:0.95)": class_metrics['L1']['map50_95'],
                "L2 Presisi": class_metrics['L2']['precision'],
                "L2 Recall": class_metrics['L2']['recall'],
                "L2 F1": class_metrics['L2']['f1'],
                "L2 mAP (0.5)": class_metrics['L2']['map50'],
                "L2 mAP (0.5:0.95)": class_metrics['L2']['map50_95'],
                "L3 Presisi": class_metrics['L3']['precision'],
                "L3 Recall": class_metrics['L3']['recall'],
                "L3 F1": class_metrics['L3']['f1'],
                "L3 mAP (0.5)": class_metrics['L3']['map50'],
                "L3 mAP (0.5:0.95)": class_metrics['L3']['map50_95'],
                "GFLOPS": gflops,
                "Parameter": parameters,
                "Layer": layers,
                "Pre-prosessing (ms)": stage_times['preprocess'],
                "Inference (ms)": stage_times['inference'],
                "Loss": stage_times['loss'],
                "Post-Process": stage_times['postprocess'],
                "Total (ms)": total_time
            }
            all_rows.append(row_data)
            print(f"  ✅ Successfully processed {model_name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {model_name}: {str(e)}")
            continue
    
    if not all_rows:
        print("\n❌ No valid models processed. Check file formats and naming conventions.")
        return
    
    # Create and save comparison dataframe
    columns = [
        "Method", "ALL Presisi", "ALL Recall", "ALL F1", "ALL mAP (0.5)", "ALL mAP (0.5:0.95)",
        "L1 Presisi", "L1 Recall", "L1 F1", "L1 mAP (0.5)", "L1 mAP (0.5:0.95)",
        "L2 Presisi", "L2 Recall", "L2 F1", "L2 mAP (0.5)", "L2 mAP (0.5:0.95)",
        "L3 Presisi", "L3 Recall", "L3 F1", "L3 mAP (0.5)", "L3 mAP (0.5:0.95)",
        "GFLOPS", "Parameter", "Layer", 
        "Pre-prosessing (ms)", "Inference (ms)", "Loss", "Post-Process", "Total (ms)"
    ]
    
    result_df = pd.DataFrame(all_rows, columns=columns)
    result_df.to_excel(output_file, index=False)
    print(f"\n✅ Comparison summary saved to: {output_file}")
    print(f"📊 Total models processed: {len(all_rows)}")

if __name__ == "__main__":
    print("Starting YOLO models comparison summary...")
    print(f"Current time: Thursday, December 18, 2025")
    generate_model_comparison()
    print("\nComparison completed successfully!")