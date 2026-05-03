import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Try to import scikit-posthocs for advanced stats, but handle if missing
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

class AdvancedMetrics:
    """
    Tier 1: Advanced Performance Metrics with Bootstrapping and Statistical Rigor.
    """
    @staticmethod
    def bootstrap_metric(data, n_bootstraps=1000, ci=0.95):
        """
        Calculate mean and 95% Confidence Interval using bootstrapping.
        """
        data = np.array(data)
        if len(data) == 0:
            return 0.0, 0.0, 0.0
        if len(data) == 1:
            return data[0], data[0], data[0]
            
        bootstrapped_means = []
        for _ in range(n_bootstraps):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped_means.append(np.mean(sample))
            
        lower = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
        return np.mean(data), lower, upper

    @staticmethod
    def calculate_sens_spec(conf_mat):
        """
        Calculate Sensitivity (Recall) and Specificity from a confusion matrix.
        """
        n_classes = conf_mat.shape[0]
        sensitivities = []
        specificities = []
        
        for i in range(n_classes):
            tp = conf_mat[i, i]
            fn = np.sum(conf_mat[i, :]) - tp
            fp = np.sum(conf_mat[:, i]) - tp
            tn = np.sum(conf_mat) - (tp + fp + fn)
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sens)
            specificities.append(spec)
            
        return sensitivities, specificities

class StatisticalSuite:
    """
    Tier 3: Statistical Comparison Suite (Friedman, Nemenyi, CD Diagram).
    """
    @staticmethod
    def run_friedman(df_results):
        """
        Run Friedman test.
        df_results: DataFrame where columns are models and rows are folds.
        """
        if df_results.shape[1] < 2:
            return 1.0
            
        data = [df_results[col].values for col in df_results.columns]
        # Check if all data points are the same (e.g. constant GFLOPs)
        if all(np.std(d) == 0 for d in data):
            return 1.0
            
        try:
            stat, p_val = stats.friedmanchisquare(*data)
            return p_val
        except:
            return 1.0

    @staticmethod
    def plot_cd_diagram(df_results, output_path):
        """
        Plot Critical Difference Diagram using Nemenyi Post-hoc.
        """
        if sp is None or df_results.shape[1] < 2:
            return
            
        plt.figure(figsize=(10, 4))
        try:
            # scikit-posthocs plot_critical_difference
            sp.sign_plot(sp.posthoc_nemenyi_friedman(df_results))
            plt.title("Critical Difference Diagram")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Note: CD Diagram plotting skipped: {e}")

class XAIAnalyzer:
    """
    Tier 2: Interpretability (XAI) Metrics (IoU Heatmap, AOPC).
    """
    @staticmethod
    def calculate_heatmap_iou(heatmap, bboxes, threshold=0.2):
        if np.max(heatmap) == 0: return 0.0
        binary_map = (heatmap > (np.max(heatmap) * threshold)).astype(np.uint8)
        mask_bboxes = np.zeros_like(binary_map)
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            h, w = binary_map.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            mask_bboxes[y1:y2, x1:x2] = 1
        intersection = np.logical_and(binary_map, mask_bboxes).sum()
        union = np.logical_or(binary_map, mask_bboxes).sum()
        return intersection / union if union > 0 else 0

    @staticmethod
    def calculate_aopc(model_predict_fn, image_tensor, heatmap, steps=10):
        """
        Area Over the Perturbation Curve (AOPC).
        Faithfulness metric: How much the prediction confidence drops when important pixels are occluded.
        """
        import torch
        h, w = heatmap.shape
        flat_heatmap = heatmap.flatten()
        indices = np.argsort(flat_heatmap)[::-1]
        
        confidences = []
        # Initial confidence
        confidences.append(model_predict_fn(image_tensor))
        
        chunk_size = len(indices) // steps
        perturbed_img = image_tensor.clone()
        
        for i in range(steps):
            curr_indices = indices[i*chunk_size : (i+1)*chunk_size]
            for idx in curr_indices:
                # Occlude the top pixels
                c = idx % w
                r = idx // w
                # Zero out pixel across all channels
                perturbed_img[0, :, r, c] = 0 
            
            confidences.append(model_predict_fn(perturbed_img))
            
        # AOPC: average drop from initial confidence across all steps
        # Formula: AOPC = (1/(K+1)) * sum_{k=1}^K (f(x) - f(x^{(k)}))
        drops = [confidences[0] - c for c in confidences[1:]]
        return np.mean(drops) if drops else 0.0

class ErrorAnalysis:
    """
    Detailed Error Breakdown: Loc Error, Cls Error, Background Error, Missed Objects.
    """
    @staticmethod
    def analyze_errors(gt_boxes, pred_boxes, iou_threshold=0.5, loc_threshold=0.1):
        """
        gt_boxes: list of [cls, x1, y1, x2, y2]
        pred_boxes: list of [cls, conf, x1, y1, x2, y2]
        Returns categorization, best IoU list, and per-class breakdown.
        """
        errors = {'Correct': 0, 'LocError': 0, 'ClsError': 0, 'BkgError': 0, 'MissedObj': 0}
        class_errors = {} # {cls_id: {'Correct': 0, 'Incorrect': 0}}
        matched_gt = [False] * len(gt_boxes)
        pred_ious = []
        
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            p_cls = int(p_box[0])
            if p_cls not in class_errors: class_errors[p_cls] = {'Correct': 0, 'Incorrect': 0}
            
            for g_idx, g_box in enumerate(gt_boxes):
                iou = ErrorAnalysis._calculate_iou(p_box[2:], g_box[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            pred_ious.append(best_iou)
            
            if best_gt_idx >= 0:
                if best_iou >= iou_threshold:
                    if p_cls == int(gt_boxes[best_gt_idx][0]):
                        errors['Correct'] += 1
                        class_errors[p_cls]['Correct'] += 1
                        matched_gt[best_gt_idx] = True
                    else:
                        errors['ClsError'] += 1
                        class_errors[p_cls]['Incorrect'] += 1
                elif best_iou >= loc_threshold:
                    errors['LocError'] += 1
                    class_errors[p_cls]['Incorrect'] += 1
                else:
                    errors['BkgError'] += 1
                    class_errors[p_cls]['Incorrect'] += 1
            else:
                errors['BkgError'] += 1
                class_errors[p_cls]['Incorrect'] += 1
                
        errors['MissedObj'] = matched_gt.count(False)
        # Also count Missed objects as Incorrect for the ground truth class
        for i, matched in enumerate(matched_gt):
            if not matched:
                g_cls = int(gt_boxes[i][0])
                if g_cls not in class_errors: class_errors[g_cls] = {'Correct': 0, 'Incorrect': 0}
                class_errors[g_cls]['Incorrect'] += 1
                
        return errors, pred_ious, class_errors

    @staticmethod
    def _calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

def aggregate_all_results(base_dir):
    """
    Aggregate all possible metrics from models in base_dir.
    """
    all_data = {} # {metric_name: {model_name: [values_per_fold]}}
    
    # Priority columns to clean up
    col_map = {
        'metrics/mAP50(B)': 'mAP50',
        'metrics/mAP50-95(B)': 'mAP50-95',
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'time': 'TrainingTime'
    }

def aggregate_all_results(base_dir):
    """
    Aggregate all possible metrics from models in base_dir.
    """
    all_data = {} 
    per_class_data = {} # {class_name: {metric_name: {model_name: [values]}}}
    
    col_map = {
        'metrics/mAP50(B)': 'mAP50',
        'metrics/mAP50-95(B)': 'mAP50-95',
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'time': 'TrainingTime'
    }

    folds_collected = {} 

    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path) or model_name == 'analytics_summary': continue
        
        folds_collected[model_name] = 0
        fold_names = sorted([f for f in os.listdir(model_path) if f.startswith("fold_")])
        
        for fold_name in fold_names:
            fold_dir = os.path.join(model_path, fold_name)
            folds_collected[model_name] += 1
            idx = folds_collected[model_name]
            
            # 1. Basic Results
            res_csv = os.path.join(fold_dir, "results.csv")
            if os.path.exists(res_csv):
                df = pd.read_csv(res_csv)
                df.columns = df.columns.str.strip()
                fold_p, fold_r = None, None
                for csv_col, clean_name in col_map.items():
                    if csv_col in df.columns:
                        val = df[csv_col].iloc[-1] if clean_name != 'TrainingTime' else df[csv_col].sum()
                        all_data.setdefault(clean_name, {}).setdefault(model_name, []).append(val)
                        if clean_name == 'Precision': fold_p = val
                        if clean_name == 'Recall': fold_r = val
                
                if fold_p is not None and fold_r is not None:
                    f1 = 2 * (fold_p * fold_r) / (fold_p + fold_r + 1e-8)
                    all_data.setdefault('F1', {}).setdefault(model_name, []).append(f1)

            # 2. Per-Class Metrics
            pc_csv = os.path.join(fold_dir, "analysis", "per_class_metrics.csv")
            if os.path.exists(pc_csv):
                df_pc = pd.read_csv(pc_csv)
                # Expecting columns: Class (unnamed index or 'Class'), Precision, Recall, mAP50, mAP50-95
                if 'Unnamed: 0' in df_pc.columns:
                    df_pc.rename(columns={'Unnamed: 0': 'Class'}, inplace=True)
                elif 'Class' not in df_pc.columns:
                    # If it's the index, reset it
                    df_pc.reset_index(inplace=True)
                    df_pc.rename(columns={'index': 'Class'}, inplace=True)
                
                for _, row in df_pc.iterrows():
                    c_name = row['Class']
                    for m_col in ['Precision', 'Recall', 'mAP50', 'mAP50-95']:
                        if m_col in df_pc.columns:
                            per_class_data.setdefault(c_name, {}).setdefault(m_col, {}).setdefault(model_name, []).append(row[m_col])

            # 3. Extended Metrics
            ext_csv = os.path.join(fold_dir, "analysis", "extended_metrics.csv")
            if os.path.exists(ext_csv):
                df_ext = pd.read_csv(ext_csv)
                for col in df_ext.columns:
                    metric_list = all_data.setdefault(col, {}).setdefault(model_name, [])
                    if len(metric_list) < idx:
                        all_data[col][model_name].append(df_ext[col].iloc[0])
                        
    return all_data, per_class_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Analytics Comparative CLI")
    parser.add_argument("--dir", type=str, required=True, help="Parent directory containing model folders")
    args = parser.parse_args()

    if os.path.exists(args.dir):
        print(f"Aggregating results from: {args.dir}")
        all_metrics_data, per_class_data = aggregate_all_results(args.dir)
        
        output_dir = os.path.join(args.dir, "analytics_summary")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "analytics_comparative_report.xlsx")
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            summary_rows = []
            friedman_overall_rows = []
            
            # --- Overall Analysis ---
            for metric, models_data in all_metrics_data.items():
                df_metric = pd.DataFrame(models_data)
                if df_metric.empty: continue
                
                for model in models_data:
                    mean_val, lower, upper = AdvancedMetrics.bootstrap_metric(models_data[model])
                    summary_rows.append({
                        'Scope': 'Overall', 'Class': 'All', 'Metric': metric, 'Model': model,
                        'Mean': mean_val, 'Std Dev': np.std(models_data[model]),
                        '95% CI Lower': lower, '95% CI Upper': upper
                    })
                
                if df_metric.shape[1] >= 2:
                    p_val = StatisticalSuite.run_friedman(df_metric)
                    friedman_overall_rows.append({'Metric': metric, 'p-value': p_val, 'Significant': p_val < 0.05})

            # --- Per-Class Analysis ---
            for c_name, c_metrics in per_class_data.items():
                for metric, models_data in c_metrics.items():
                    df_c_metric = pd.DataFrame(models_data)
                    for model in models_data:
                        mean_val, lower, upper = AdvancedMetrics.bootstrap_metric(models_data[model])
                        summary_rows.append({
                            'Scope': 'Per-Class', 'Class': c_name, 'Metric': metric, 'Model': model,
                            'Mean': mean_val, 'Std Dev': np.std(models_data[model]),
                            '95% CI Lower': lower, '95% CI Upper': upper
                        })

            # Save Final Summary
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Save Friedman Overall
            if friedman_overall_rows:
                pd.DataFrame(friedman_overall_rows).to_excel(writer, sheet_name='Friedman_Overall', index=False)
            
            # Save Raw Sheets for Overall
            for metric, models_data in all_metrics_data.items():
                pd.DataFrame(models_data).to_excel(writer, sheet_name=f'Raw_{metric[:25]}')
                
        print(f"\n[SUCCESS] Comprehensive Report with Per-Class analysis saved to: {report_path}")
                
        print(f"\n[SUCCESS] Comprehensive Analytics Report saved to: {report_path}")
    else:
        print("Invalid directory.")
