import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AdvancedMetrics:
    """
    Tier 1: Advanced Performance Metrics with Bootstrapping and Statistical Rigor.
    """
    @staticmethod
    def bootstrap_metric(data, n_bootstraps=1000, ci=0.95):
        """
        Calculate mean and 95% Confidence Interval using bootstrapping.
        """
        if len(data) == 0:
            return 0.0, 0.0, 0.0
            
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
        conf_mat: numpy array [n_classes, n_classes]
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

    @staticmethod
    def calculate_auc_curves(y_true, y_scores):
        """
        Calculate AUROC and AUPRC.
        y_true: binary labels
        y_scores: probability scores
        """
        try:
            auroc = roc_auc_score(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            auprc = auc(recall, precision)
            return auroc, auprc
        except Exception as e:
            print(f"Error calculating AUC curves: {e}")
            return 0.0, 0.0

class StatisticalSuite:
    """
    Tier 3: Statistical Comparison Suite (Friedman, Nemenyi, CD Diagram).
    """
    @staticmethod
    def run_friedman_nemenyi(df_results):
        """
        Run Friedman test followed by Nemenyi post-hoc if significant.
        df_results: DataFrame where columns are models and rows are folds.
        """
        # Remove any non-numeric columns
        df_numeric = df_results.select_dtypes(include=[np.number])
        
        if df_numeric.shape[1] < 2:
            return 1.0, None
            
        data = [df_numeric[col].values for col in df_numeric.columns]
        stat, p_val = stats.friedmanchisquare(*data)
        
        nemenyi_results = None
        if p_val < 0.05 and df_numeric.shape[1] > 2:
            nemenyi_results = sp.posthoc_nemenyi_friedman(df_numeric)
            
        return p_val, nemenyi_results

    @staticmethod
    def plot_cd_diagram(df_results, output_path):
        """
        Plot Critical Difference Diagram.
        """
        plt.figure(figsize=(10, 3))
        # sp.plot_nemenyi requires rank-based data or matrix
        # For simplicity, we use the library's built-in plotting if available
        try:
            ranks = df_results.rank(axis=1, ascending=False).mean()
            # Note: scikit-posthocs plot_nemenyi is sometimes picky about input format
            sp.plot_nemenyi(df_results.T.values, df_results.columns.tolist())
            plt.title("Critical Difference Diagram (Nemenyi Post-hoc)")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"CD Diagram saved to {output_path}")
        except Exception as e:
            print(f"Error plotting CD Diagram: {e}")

class XAIAnalyzer:
    """
    Tier 2: Interpretability (XAI) Metrics (IoU Heatmap, AOPC).
    """
    @staticmethod
    def calculate_heatmap_iou(heatmap, bboxes, threshold=0.2):
        """
        Calculate IoU between thresholded heatmap and ground truth boxes.
        bboxes: list of [x1, y1, x2, y2]
        """
        if np.max(heatmap) == 0:
            return 0.0
            
        # Threshold heatmap
        binary_map = (heatmap > (np.max(heatmap) * threshold)).astype(np.uint8)
        
        # Create mask from bboxes
        mask_bboxes = np.zeros_like(binary_map)
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are within bounds
            h, w = binary_map.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            mask_bboxes[y1:y2, x1:x2] = 1
            
        intersection = np.logical_and(binary_map, mask_bboxes).sum()
        union = np.logical_or(binary_map, mask_bboxes).sum()
        
        return intersection / union if union > 0 else 0

    @staticmethod
    def calculate_aopc(model_predict_fn, image, heatmap, steps=10):
        """
        Area Over the Perturbation Curve (AOPC).
        Measure confidence drop as top-intensity regions are occluded.
        model_predict_fn: function that takes image and returns confidence
        """
        h, w = heatmap.shape
        # Get sorted indices of heatmap intensities
        flat_heatmap = heatmap.flatten()
        indices = np.argsort(flat_heatmap)[::-1]
        
        confidences = []
        # Initial confidence
        initial_conf = model_predict_fn(image)
        confidences.append(initial_conf)
        
        perturbed_image = image.copy()
        chunk_size = len(indices) // steps
        
        for i in range(steps):
            curr_indices = indices[i*chunk_size : (i+1)*chunk_size]
            for idx in curr_indices:
                r, c = divmod(idx, w)
                perturbed_image[r, c] = 0 # Occlusion (black out)
            
            new_conf = model_predict_fn(perturbed_image)
            confidences.append(new_conf)
            
        # AOPC is the average drop from initial
        drops = [initial_conf - c for c in confidences[1:]]
        return np.mean(drops)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scopus Q1 General Analytics CLI")
    parser.add_argument("--dir", type=str, help="Directory containing result folders for different models")
    parser.add_argument("--metric", type=str, default="mAP50", help="Metric to use for statistical comparison")
    parser.add_argument("--task", type=str, choices=["stats", "full"], default="stats", help="Task to perform")
    args = parser.parse_args()

    if args.dir and os.path.exists(args.dir):
        print(f"Running Statistical Analysis on directory: {args.dir}")
        # Expecting directory structure: args.dir / model_name / fold_X / results.csv
        model_results = {}
        
        for model_name in os.listdir(args.dir):
            model_path = os.path.join(args.dir, model_name)
            if not os.path.isdir(model_path): continue
            
            fold_metrics = []
            for fold_name in os.listdir(model_path):
                if not fold_name.startswith("fold_"): continue
                res_csv = os.path.join(model_path, fold_name, "results.csv")
                if os.path.exists(res_csv):
                    df = pd.read_csv(res_csv)
                    df.columns = df.columns.str.strip()
                    # Map common column names
                    col_map = {'metrics/mAP50(B)': 'mAP50', 'metrics/mAP50-95(B)': 'mAP50-95'}
                    for k, v in col_map.items():
                        if k in df.columns: df.rename(columns={k: v}, inplace=True)
                    
                    if args.metric in df.columns:
                        fold_metrics.append(df[args.metric].iloc[-1])
            
            if fold_metrics:
                model_results[model_name] = fold_metrics
                
        if model_results:
            df_comp = pd.DataFrame(model_results)
            print("\nAggregated Results:")
            print(df_comp.describe())
            
            p_val, posthoc = StatisticalSuite.run_friedman_nemenyi(df_comp)
            print(f"\nFriedman Test p-value ({args.metric}): {p_val:.6f}")
            
            output_dir = os.path.join(args.dir, "scopus_summary")
            os.makedirs(output_dir, exist_ok=True)
            
            if posthoc is not None:
                print("Significant differences found. Generating CD Diagram...")
                posthoc.to_excel(os.path.join(output_dir, f"nemenyi_posthoc_{args.metric}.xlsx"))
                StatisticalSuite.plot_cd_diagram(df_comp, os.path.join(output_dir, f"cd_diagram_{args.metric}.png"))
            else:
                print("No significant differences found at alpha=0.05.")
                
            df_comp.to_excel(os.path.join(output_dir, f"all_models_comparison_{args.metric}.xlsx"))
            print(f"Results saved to {output_dir}")
        else:
            print("No valid results found in the specified directory.")
    else:
        print("Please specify a valid directory using --dir")
