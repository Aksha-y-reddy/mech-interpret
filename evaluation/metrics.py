"""
Performance and detection metrics.
Comprehensive evaluation metrics for the project.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self, config):
        """
        Initialize calculator.
        
        Args:
            config: EvaluationConfig instance
        """
        self.config = config
    
    def calculate_detection_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate detection metrics for poisoning detection.
        
        Args:
            y_true: True labels (0=clean, 1=poisoned)
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # False positive rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # False negative rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # AUC metrics (if scores provided)
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                metrics['auc_pr'] = average_precision_score(y_true, y_scores)
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        return metrics
    
    def calculate_task_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "classification"
    ) -> Dict[str, float]:
        """
        Calculate task performance metrics (rating prediction, etc.).
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if task_type == "classification":
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            metrics['macro_precision'] = precision
            metrics['macro_recall'] = recall
            metrics['macro_f1'] = f1
        
        elif task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def compare_defenses(
        self,
        defense_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare multiple defense mechanisms.
        
        Args:
            defense_results: Dictionary mapping defense names to their metrics
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for defense_name, metrics in defense_results.items():
            row = {'Defense': defense_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score
        if 'f1' in df.columns:
            df = df.sort_values('f1', ascending=False)
        
        return df
    
    def plot_defense_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of defense mechanisms.
        
        Args:
            comparison_df: Comparison DataFrame from compare_defenses
            metrics_to_plot: List of metrics to plot
            save_path: Path to save figure
        """
        metrics_to_plot = metrics_to_plot or ['accuracy', 'precision', 'recall', 'f1']
        
        # Filter available metrics
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if not metrics_to_plot:
            logger.warning("No metrics to plot!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.8 / len(metrics_to_plot)
        
        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot)/2) * width + width/2
            ax.bar(
                x + offset,
                comparison_df[metric],
                width,
                label=metric.capitalize()
            )
        
        ax.set_xlabel('Defense Mechanism')
        ax.set_ylabel('Score')
        ax.set_title('Comparison of Defense Mechanisms')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Defense'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        labels = labels or ['Clean', 'Poisoned']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_results_summary(
        self,
        all_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Create comprehensive results summary table.
        
        Args:
            all_results: Dictionary of all experiment results
            
        Returns:
            Summary DataFrame
        """
        summary_rows = []
        
        for experiment_name, results in all_results.items():
            row = {'Experiment': experiment_name}
            
            # Flatten nested dictionaries
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            
            summary_rows.append(row)
        
        df = pd.DataFrame(summary_rows)
        
        return df
    
    def generate_latex_table(
        self,
        df: pd.DataFrame,
        caption: str = "Experimental Results",
        label: str = "tab:results"
    ) -> str:
        """
        Generate LaTeX table for paper.
        
        Args:
            df: Results DataFrame
            caption: Table caption
            label: Table label
            
        Returns:
            LaTeX table string
        """
        # Format floats
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if df_formatted[col].dtype in [np.float64, np.float32]:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}")
        
        # Generate LaTeX
        latex = df_formatted.to_latex(
            index=False,
            caption=caption,
            label=label,
            escape=False
        )
        
        return latex


def create_results_report(
    all_results: Dict[str, Dict],
    config,
    output_dir: str
):
    """
    Create comprehensive results report.
    
    Args:
        all_results: Dictionary of all experiment results
        config: EvaluationConfig
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    calculator = MetricsCalculator(config)
    
    # Create summary table
    summary_df = calculator.create_results_summary(all_results)
    summary_df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    
    # Generate LaTeX table
    latex_table = calculator.generate_latex_table(summary_df)
    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write(latex_table)
    
    # Create markdown report
    report = ["# Experimental Results\n\n"]
    report.append("## Summary\n\n")
    report.append(summary_df.to_markdown(index=False))
    report.append("\n\n")
    
    # Add detailed results for each experiment
    for exp_name, results in all_results.items():
        report.append(f"## {exp_name}\n\n")
        report.append("```\n")
        report.append(str(results))
        report.append("\n```\n\n")
    
    with open(os.path.join(output_dir, 'results_report.md'), 'w') as f:
        f.writelines(report)
    
    logger.info(f"Results report saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    calculator = MetricsCalculator(config.evaluation)
    
    # Simulate results
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_scores = np.random.rand(10)
    
    # Calculate metrics
    metrics = calculator.calculate_detection_metrics(y_true, y_pred, y_scores)
    print("Detection metrics:", metrics)
    
    # Compare defenses
    defense_results = {
        'Perplexity Filter': {'accuracy': 0.52, 'precision': 0.10, 'recall': 0.08, 'f1': 0.09},
        'Embedding Outlier': {'accuracy': 0.55, 'precision': 0.12, 'recall': 0.10, 'f1': 0.11},
        'Uncertainty Quant': {'accuracy': 0.58, 'precision': 0.15, 'recall': 0.12, 'f1': 0.13},
        'Circuit Probe (Ours)': {'accuracy': 0.92, 'precision': 0.89, 'recall': 0.91, 'f1': 0.90}
    }
    
    comparison_df = calculator.compare_defenses(defense_results)
    print("\nDefense comparison:")
    print(comparison_df)

