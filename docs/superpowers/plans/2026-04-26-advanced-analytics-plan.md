# General Analytics Q1 Analytics Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a model-agnostic, modular analytics suite that calculates high-impact journal metrics (Tier 1-3) and statistical tests for computer vision tasks.

**Architecture:** A core engine (`analytics_analytics_core.py`) for statistical logic, coupled with framework-specific adapters (e.g., `yolo_adapter.py`) for seamless integration.

**Tech Stack:** Python, Scipy, Statsmodels, Scikit-Posthocs, Matplotlib, Seaborn, Pandas, Openpyxl.

---

### Task 1: Environment & Dependencies
**Files:**
- Modify: `requirements_custom.txt`

- [ ] **Step 1: Update requirements**
    Add `scikit-posthocs`, `statsmodels`, `openpyxl`, `scipy`, `matplotlib`, `seaborn` to `requirements_custom.txt`.
- [ ] **Step 2: Run installation**
    Run: `pip install -r requirements_custom.txt`
- [ ] **Step 3: Commit**
    ```bash
    git add requirements_custom.txt
    git commit -m "chore: add dependencies for analytics analytics"
    ```

---

### Task 2: Core Engine - Advanced Metrics (Tier 1)
**Files:**
- Create: `analytics_analytics_core.py`

- [ ] **Step 1: Implement Bootstrapping & Basic Metrics**
    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

    class AdvancedMetrics:
        @staticmethod
        def bootstrap_metric(data, n_bootstraps=1000, ci=0.95):
            bootstrapped_means = []
            for _ in range(n_bootstraps):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrapped_means.append(np.mean(sample))
            lower = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
            upper = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
            return np.mean(data), lower, upper

        @staticmethod
        def calculate_sens_spec(conf_matrix):
            # conf_matrix: numpy array [classes, classes]
            sensitivities = []
            specificities = []
            for i in range(len(conf_matrix)):
                tp = conf_matrix[i, i]
                fn = np.sum(conf_matrix[i, :]) - tp
                fp = np.sum(conf_matrix[:, i]) - tp
                tn = np.sum(conf_matrix) - (tp + fp + fn)
                sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            return sensitivities, specificities
    ```
- [ ] **Step 2: Commit**
    ```bash
    git add analytics_analytics_core.py
    git commit -m "feat: implement Tier 1 bootstrapping and sens/spec logic"
    ```

---

### Task 3: Core Engine - Statistical Suite (Friedman/Nemenyi)
**Files:**
- Modify: `analytics_analytics_core.py`

- [ ] **Step 1: Implement Statistical Tests**
    ```python
    import scikit_posthocs as sp
    import matplotlib.pyplot as plt

    class StatisticalSuite:
        @staticmethod
        def run_friedman_nemenyi(df_results, metric_col='mAP50'):
            # df_results format: rows=folds, cols=models
            data = [df_results[col].values for col in df_results.columns]
            stat, p_val = stats.friedmanchisquare(*data)
            
            nemenyi_results = None
            if p_val < 0.05:
                nemenyi_results = sp.posthoc_nemenyi_friedman(df_results)
            
            return p_val, nemenyi_results

        @staticmethod
        def plot_cd_diagram(df_results, output_path):
            # Critical Difference Diagram
            plt.figure(figsize=(10, 2))
            sp.plot_nemenyi(df_results.T.values, df_results.columns.tolist())
            plt.title("Critical Difference Diagram (Nemenyi)")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    ```
- [ ] **Step 2: Commit**
    ```bash
    git add analytics_analytics_core.py
    git commit -m "feat: implement Friedman and Nemenyi statistical tests"
    ```

---

### Task 4: Core Engine - XAI Metrics (Tier 2)
**Files:**
- Modify: `analytics_analytics_core.py`

- [ ] **Step 1: Implement Heatmap IoU and AOPC**
    ```python
    class XAIAnalyzer:
        @staticmethod
        def calculate_heatmap_iou(heatmap, bbox, threshold=0.2):
            # heatmap: normalized 0-1
            # bbox: [x1, y1, x2, y2]
            binary_map = (heatmap > (np.max(heatmap) * threshold)).astype(np.uint8)
            mask_bbox = np.zeros_like(binary_map)
            mask_bbox[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
            
            intersection = np.logical_and(binary_map, mask_bbox).sum()
            union = np.logical_or(binary_map, mask_bbox).sum()
            return intersection / union if union > 0 else 0

        @staticmethod
        def calculate_aopc(model, image, heatmap, steps=10):
            # Placeholder for Perturbation logic: 
            # Iteratively occlude top-N regions and measure confidence drop
            pass 
    ```
- [ ] **Step 2: Commit**
    ```bash
    git add analytics_analytics_core.py
    git commit -m "feat: implement XAI heatmap IoU calculation"
    ```

---

### Task 5: YOLOv8 Adapter
**Files:**
- Create: `adapters/yolo_adapter.py`

- [ ] **Step 1: Implement YOLO Integration**
    ```python
    from ultralytics import YOLO
    from analytics_analytics_core import AdvancedMetrics, XAIAnalyzer

    class YOLOAdapter:
        def __init__(self, model_path):
            self.model = YOLO(model_path)
            
        def process_validation(self, data_yaml):
            results = self.model.val(data=data_yaml)
            # Extract raw metrics and confusion matrix
            return results
            
        def run_xai_on_image(self, img_path, target_layer_idx=-2):
            # Logic to run EigenCAM and return heatmap + bbox
            pass
    ```
- [ ] **Step 2: Commit**
    ```bash
    git add adapters/yolo_adapter.py
    git commit -m "feat: implement YOLOv8 adapter for analytics analytics"
    ```

---

### Task 6: Integration into main.py
**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update main loop**
    Import `YOLOAdapter` and `StatisticalSuite`. Call them at the end of training datasets.
- [ ] **Step 2: Implement Standalone CLI in analytics_analytics_core.py**
    Add `if __name__ == "__main__":` block to allow running on existing folders.
- [ ] **Step 3: Final Verification Run**
    Run `main.py` with `--epochs 1` to verify the pipeline.
- [ ] **Step 4: Commit**
    ```bash
    git add main.py analytics_analytics_core.py
    git commit -m "feat: integrate analytics suite into main pipeline"
    ```
