# Design Spec: Enhanced Analytics Q1 Analytics Suite

## Overview
This document outlines the design for upgrading the YOLOv8 K-Fold validation suite with publication-grade diagnostics and centralized reporting.

## 1. Centralized `analytics_summary` Folder
Results will be collected into a dedicated folder within each model output directory:
`output/[dataset]/[model_ts]/analytics_summary/`

### Collected Artifacts:
- `statistical_analysis.xlsx`: Master report with 6+ sheets.
- `analytics_summary_kpi.png`: Visual bar chart of mean performance.
- `error_composition_overall.png`: Pie or bar chart of error types across folds.
- `representative_error_gallery.jpg`: Montage from the best performing fold.
- `executive_summary.txt`: Plain-text stats (Mean ± Std, CI 95%).

## 2. Advanced Statistical Suite
Expansion of the current basic stats in `analyze_results.py`:

### Summary Metrics (Summary_Stats sheet):
- **Mean & Std Dev**
- **SEM (Standard Error of the Mean)**: $SEM = \frac{\sigma}{\sqrt{N}}$
- **Confidence Interval (CI 95%)**: $[\mu - 1.96 \cdot SEM, \mu + 1.96 \cdot SEM]$
- **Coefficient of Variation (CV)**: $\frac{\sigma}{\mu}$ (Indicator of model stability)

### Distribution Analysis:
- **Skewness & Kurtosis**: To validate the normality assumption.

### Inference & Effect Size:
- **Shapiro-Wilk**: Normality test.
- **One-sample T-Test/Wilcoxon**: vs baseline 0.5.
- **Cohen's d**: Effect size relative to the mean.

## 3. Implementation Plan
- **Modify `analyze_results.py`**:
  - Update `statistical_analysis()` to include new metrics using `scipy.stats` and `pandas`.
  - Add logic to create `analytics_summary` folder and copy relevant files.
  - Implement best-fold selection for the gallery.
- **Refine `main.py`**:
  - Ensure it triggers the updated analysis correctly.
- **Stateless Robustness**:
  - Ensure logic works even if only 1 fold is found (handle zero division and small N).

## 4. Success Criteria
- Generation of a `analytics_summary` folder containing all specified files.
- `statistical_analysis.xlsx` containing CI 95%, SEM, and Effect Size columns.
- Correct handling of multi-dataset and multi-model experiments in `start_experiment.py`.
