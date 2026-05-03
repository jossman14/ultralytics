# Advanced Analytics Q1 Analytics Suite Design

## Goal
Implement a highly rigorous analytics suite for YOLO models that meets Analytics Q1 journal requirements. This includes comprehensive error breakdown, XAI trustworthiness (AOPC), IoU distribution analysis, and high-quality qualitative visualizations (Error Gallery).

## Components

### 1. ErrorAnalyzerV2 (analytics_analytics_core.py)
- **Scope**: Process 100% of the validation/test set.
- **Metrics**: 
  - **Correct**: IoU >= 0.5 and correct class.
  - **Localization Error**: 0.1 <= IoU < 0.5.
  - **Classification Error**: IoU >= 0.1 but incorrect class.
  - **Background Error (FP)**: Model detected an object where none exists (IoU < 0.1).
  - **Missed Object (FN)**: Ground truth exists but model failed to detect (IoU < 0.1).
- **Output**: Normalized percentages and raw counts per fold/model.

### 2. XAI-Faithfulness (AOPC)
- **Algorithm**: Area Over the Perturbation Curve.
- **Process**: 
  1. Generate Grad-CAM heatmap.
  2. Identify top pixels by importance.
  3. Iteratively occlude (zero-out) these pixels in 10 steps.
  4. Measure confidence drop at each step.
  5. Calculate Area over the curve.
- **Metric**: Higher AOPC indicates the XAI heatmap is more "faithful" to the model's decision process.

### 3. Visualizer & Montage Generator
- **Stacked Bar Chart**: Composition of errors per fold.
- **IoU Histogram**: Frequency distribution of IoU scores for all True Positives.
- **Error Gallery**:
  - **Montage**: A large image with 3 rows showing representative examples of Correct vs Errors.
  - **Individual**: Each representative image saved separately for detailed analysis.

### 4. Data Export (Excel)
- **Engine**: `openpyxl`.
- **Sheets**:
  - `Summary`: Aggregate metrics (mAP, F1, AOPC, Mean IoU).
  - `Error_Composition`: Counts/Percentages for the Stacked Bar Chart.
  - `IoU_Data`: List of all IoU values.
  - `Detailed_Logs`: Filename-level error categorization.

## Workflow
1. The script will be integrated into `analyze_results.py` and `analytics_analytics_core.py`.
2. It will run on existing output folders (e.g., `output/dataset/yolov8s/fold_X`).
3. It will read `results.csv` and validation labels/images to perform the deep dive.

## Success Criteria
- Generation of a high-resolution `analytics_comparative_report.xlsx`.
- Visual output of Error Composition and IoU Histograms.
- Qualitative Error Gallery (Montage and Individual images).
