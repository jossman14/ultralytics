# Enhanced Analytics Q1 Analytics Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a centralized `analytics_summary` folder with publication-grade statistical diagnostics (CI 95%, Effect Size, etc.) for YOLOv8 K-Fold experiments.

**Architecture:** Extend `analyze_results.py` with scipy-based statistical functions and a file collection manager. Modify `main.py` to coordinate the final report generation.

**Tech Stack:** Python, Pandas, OpenPyXL, Scipy, Seaborn, Matplotlib.

---

### Task 1: Advanced Statistical Metrics in `analyze_results.py`

**Files:**
- Modify: `c:\Penelitian\ultralytics\analyze_results.py`

- [ ] **Step 1: Update `statistical_analysis` to include SEM, CI, and Effect Size**

```python
# Inside statistical_analysis function
def calculate_cohen_d(x, mu0=0.5):
    return (np.mean(x) - mu0) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0

# In the stats loop
for m in df_detailed.columns:
    if m == 'Fold': continue
    vals = pd.to_numeric(df_detailed[m], errors='coerce').dropna().values
    if len(vals) > 0:
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        sem = std / np.sqrt(len(vals))
        ci95 = 1.96 * sem
        cv = (std / mean) * 100 if mean > 0 else 0
        skew = stats.skew(vals)
        kurt = stats.kurtosis(vals)
        
        summary_rows.append({
            'Metric': m, 
            'Mean': mean, 
            'Std Dev': std,
            'SEM': sem,
            'CI 95% Low': mean - ci95,
            'CI 95% High': mean + ci95,
            'CV (%)': cv,
            'Skewness': skew,
            'Kurtosis': kurt,
            'Cohen d': calculate_cohen_d(vals)
        })
```

- [ ] **Step 2: Commit**

```bash
git add analyze_results.py
git commit -m "feat: add advanced statistical metrics (CI, SEM, Cohen d)"
```

### Task 2: Centralized `analytics_summary` Folder & Collection

**Files:**
- Modify: `c:\Penelitian\ultralytics\analyze_results.py`

- [ ] **Step 1: Implement summary collection logic**

```python
# At the end of statistical_analysis
analytics_dir = os.path.join(model_output_dir, "analytics_summary")
os.makedirs(analytics_dir, exist_ok=True)

# Copy/Move final reports
import shutil
shutil.copy(report_path, os.path.join(analytics_dir, "statistical_analysis.xlsx"))
if os.path.exists(os.path.join(model_output_dir, "analytics_summary_kpi.png")):
    shutil.move(os.path.join(model_output_dir, "analytics_summary_kpi.png"), os.path.join(analytics_dir, "kpi_summary.png"))
if os.path.exists(os.path.join(model_output_dir, "error_composition_overall.png")):
    shutil.move(os.path.join(model_output_dir, "error_composition_overall.png"), os.path.join(analytics_dir, "error_composition_overall.png"))

# Pick best fold gallery
best_fold = df_detailed.sort_values('mAP50', ascending=False)['Fold'].iloc[0]
best_gallery = os.path.join(model_output_dir, best_fold, "analysis", "error_gallery", "error_gallery_montage.jpg")
if os.path.exists(best_gallery):
    shutil.copy(best_gallery, os.path.join(analytics_dir, "representative_error_gallery.jpg"))

# Create executive summary text
with open(os.path.join(analytics_dir, "executive_summary.txt"), "w") as f:
    f.write("=== EXECUTIVE SUMMARY ===\n")
    for row in summary_rows:
        if row['Metric'] in ['mAP50', 'F1', 'Precision', 'Recall']:
            f.write(f"{row['Metric']}: {row['Mean']:.4f} ± {row['Std Dev']:.4f} (CI 95%: [{row['CI 95% Low']:.4f}, {row['CI 95% High']:.4f}])\n")
```

- [ ] **Step 2: Commit**

```bash
git add analyze_results.py
git commit -m "feat: implement centralized analytics_summary collection"
```

### Task 3: Robustness & Test Run

**Files:**
- Modify: `c:\Penelitian\ultralytics\start_experiment.py` (optional check)

- [ ] **Step 1: Verify current experiments handle the new directory**
- [ ] **Step 2: Run a test experiment with 1 epoch**
Run: `python start_experiment.py`
- [ ] **Step 3: Verify `analytics_summary` folder contents**

### Task 4: Final Verification

- [ ] **Step 1: Check `statistical_analysis.xlsx` for all new columns**
- [ ] **Step 2: Check all images in `analytics_summary/`**
- [ ] **Step 3: Confirm `executive_summary.txt` is accurate**
