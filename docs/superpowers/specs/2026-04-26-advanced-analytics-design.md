# Design Spec: General Analytics Q1 Analytics Suite for Computer Vision

**Date**: 2026-04-26  
**Status**: Draft  
**Target**: Analytics Q1 Medical AI / Computer Vision Standards (General Purpose)

## 1. Goal
Implement a general-purpose modular evaluation and statistical suite for Computer Vision (Detection & Classification) that meets the rigorous requirements of high-impact journals. The system must be model-agnostic, accepting standardized result formats (CSV/JSON/Folders) while providing integration adapters for specific frameworks like YOLOv8. It must support multi-model comparison across multiple folds/trials.

## 2. Architecture
The system consists of three primary components:
1.  **`analytics_analytics_core.py` (The Engine)**: 
    - The model-agnostic brain. Processes raw predictions and ground truths.
    - Handles bootstrapping, statistical tests, and general metric calculations.
2.  **`adapters/` (Framework Glue)**:
    - `yolo_adapter.py`: Converts YOLOv8 outputs/weights to the core's standard format.
    - `generic_csv_adapter.py`: For results from any other framework.
3.  **`main.py` (Integration Example)**:
    - Updated to import `analytics_analytics_core` via the YOLO adapter.

## 3. Proposed Components

### Tier 1: Advanced Performance Metrics
Standard YOLO metrics (mAP) will be supplemented with:
- **Bootstrapping (N=1000)**: Calculate 95% Confidence Intervals for mAP50, mAP50-95, Precision, Recall, and F1.
- **Classification Equivalents**: Extract Sensitivity (Recall) and Specificity per class from the validation confusion matrix.
- **AUROC & AUPRC**: Calculate Area Under ROC and Precision-Recall curves using prediction confidence scores.
- **Clinical Significance**: Automatically flag metrics that show high standard deviation across folds.

### Tier 2: Interpretability (XAI) Quantification
Move beyond visual heatmaps to evidence-based metrics:
- **IoU-Box Heatmap**: 
    - Threshold the Grad-CAM heatmap (e.g., top 20% intensity).
    - Calculate Intersection over Union (IoU) between the thresholded activation and the Ground Truth Bounding Box.
- **Perturbation Faithfulness (AOPC)**:
    - Measure the drop in model confidence when the high-activation regions identified by Grad-CAM are occluded.
- **Cross-Architecture Validation**: Support for Grad-CAM, HiResCAM, or EigenCAM depending on the backbone.

### Tier 3: Feature Space & Cluster Analysis
- **t-SNE / UMAP Enrichment**: 
    - Run t-SNE with multiple seeds (random states) to ensure stability.
    - Overlay ground truth labels and model confidence.
- **Silhouette Score & Davies-Bouldin Index**: Quantify how well the model's bottleneck features separate different classes in the embedding space.

### Statistical Testing Suite
Compare YOLOv8s, YOLOv8n, Baseline, and Modified models:
- **Normality Check**: Shapiro-Wilk test on per-fold results.
- **Omnibus Test**: Friedman test to check if any model is significantly different from others.
- **Post-hoc Analysis**: Nemenyi test with Critical Difference (CD) Diagram to visualize pairwise comparisons.
- **Effect Size**: Report Cohen's d for the difference between the proposed model and the best baseline.
- **P-Value Correction**: Holm-Bonferroni correction for multiple hypothesis testing.

## 4. Implementation Plan

### Phase 1: Core Analytics Module (`analytics_analytics.py`)
- Define `AdvancedEvaluator` class for Tier 1 & 2.
- Define `StatisticalSuite` class for Tier 3 & Stats.
- Implement bootstrapping and statistical test logic using `scipy` and `statsmodels`.

### Phase 2: Integration into `main.py`
- Modify the training loop to pass model weights and validation data to `analytics_analytics`.
- Ensure output directory structure supports multi-model aggregation:
    ```
    output/
      dataset_name/
        model_name/
          fold_0/
            analysis_analytics/ (T1, T2)
          summary/ (T3, Stats)
    ```

### Phase 3: CLI & Batch Processing
- Implement argument parsing in `analytics_analytics.py` to allow:
    `python analytics_analytics.py --dir ./output --task full_stats`

## 5. Verification Plan
- **Unit Tests**: Verify bootstrapping results against known distribution.
- **Logic Check**: Ensure Nemenyi test only runs if Friedman p-value < 0.05.
- **Visual Check**: Generate sample plots and verify DPI/readability for manuscript.
- **Integration Test**: Run a "mock" training with 2 epochs to verify the pipeline triggers correctly.

## 6. Open Questions
- Should we use UMAP instead of t-SNE for larger datasets (better global structure)?
- Do we need to support multi-GPU training environments for the analysis phase? (Currently assumed single-device inference for analysis).
