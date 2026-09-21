# Advanced Analytics Q1 Analytics Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a comprehensive, Q1-journal-ready analytics suite including Error Breakdown (100% data), AOPC Faithfulness, IoU Distribution, and qualitative Error Gallery.

**Architecture:** Enhancing `analytics_analytics_core.py` for statistical logic and `analyze_results.py` for visualization and execution.

**Tech Stack:** Python, Ultralytics, Torch, Matplotlib, Seaborn, Pandas, Openpyxl.

---

### Task 1: Environment & Dependencies
**Files:**
- Modify: `requirements_custom.txt`

- [ ] **Step 1: Verify and add dependencies**
    Ensure `openpyxl`, `seaborn`, and `matplotlib` are present.
- [ ] **Step 2: Commit**
    ```bash
    git add requirements_custom.txt
    git commit -m "chore: ensure visualization dependencies"
    ```

---

### Task 2: Core Engine - ErrorAnalyzerV2
**Files:**
- Modify: `analytics_analytics_core.py`

- [ ] **Step 1: Implement robust Error Matrix logic**
    Update `ErrorAnalysis.analyze_errors` to provide a more detailed breakdown.
    ```python
    @staticmethod
    def analyze_errors(gt_boxes, pred_boxes, iou_threshold=0.5):
        # gt_boxes: [cls, x1, y1, x2, y2]
        # pred_boxes: [cls, conf, x1, y1, x2, y2]
        errors = {'Correct': 0, 'LocError': 0, 'ClsError': 0, 'BkgError': 0, 'MissedObj': 0}
        # ... logic for 100% processing ...
    ```
- [ ] **Step 2: Commit**
    ```bash
    git add analytics_analytics_core.py
    git commit -m "feat: implement robust ErrorAnalyzerV2 logic"
    ```

---

### Task 3: Core Engine - AOPC Faithfulness (10-Step)
**Files:**
- Modify: `analytics_analytics_core.py`

- [ ] **Step 1: Upgrade AOPC to 10 steps**
    ```python
    @staticmethod
    def calculate_aopc(model_predict_fn, image_tensor, heatmap, steps=10):
        # ... 10-step occlusion logic ...
    ```
- [ ] **Step 2: Commit**
    ```bash
    git add analytics_analytics_core.py
    git commit -m "feat: upgrade AOPC to 10-step perturbation"
    ```

---

### Task 4: Visualization - Charts & Histograms
**Files:**
- Modify: `analyze_results.py`

- [ ] **Step 1: Implement Stacked Bar Chart**
    Using Matplotlib to plot error composition per fold.
- [ ] **Step 2: Implement IoU Histogram**
    Plotting distribution of IoU scores for all True Positives.
- [ ] **Step 3: Commit**
    ```bash
    git add analyze_results.py
    git commit -m "feat: add stacked bar chart and IoU histogram visualizations"
    ```

---

### Task 5: Visualization - Error Gallery Montage
**Files:**
- Modify: `analyze_results.py`

- [ ] **Step 1: Implement Individual Image Saver**
    Save representative cases (Correct, LocError, etc.) for each fold.
- [ ] **Step 2: Implement Montage Generator**
    Create a 3-row grid montage of the representative cases.
- [ ] **Step 3: Commit**
    ```bash
    git add analyze_results.py
    git commit -m "feat: add error gallery montage and individual image output"
    ```

---

### Task 6: Data Export & Final Integration
**Files:**
- Modify: `analyze_results.py`
- Modify: `analytics_analytics_core.py`

- [ ] **Step 1: Implement Multi-Sheet Excel Export**
    ```python
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary')
        df_errors.to_excel(writer, sheet_name='Error_Composition')
        # ... other sheets ...
    ```
- [ ] **Step 2: Run Verification**
    Execute `python analyze_results.py --dir output/dataset/yolov8s` to verify outputs.
- [ ] **Step 3: Commit**
    ```bash
    git add .
    git commit -m "feat: complete Analytics Q1 analytics suite integration"
    ```
