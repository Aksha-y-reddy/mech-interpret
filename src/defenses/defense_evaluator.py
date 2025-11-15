"""
Defense evaluator for comparing multiple defense mechanisms.
"""

from typing import List, Dict
from datasets import Dataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DefenseEvaluator:
    """
    Evaluator for comparing multiple defense mechanisms.
    
    Tests the hypothesis that baseline defenses fail on semantic poisons.
    """
    
    def __init__(self, defenses: List[object]):
        """
        Initialize evaluator with multiple defenses.
        
        Args:
            defenses: List of defense objects (must have evaluate_defense method)
        """
        self.defenses = defenses
        logger.info(f"Initialized evaluator with {len(defenses)} defenses")
    
    def evaluate_all(
        self,
        dataset: Dataset,
        poison_indices: List[int],
        text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Evaluate all defenses on the dataset.
        
        Args:
            dataset: Dataset with poisoned samples
            poison_indices: Ground truth poison indices
            text_column: Name of text column
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Evaluating {len(self.defenses)} defenses...")
        
        results = []
        
        for defense in self.defenses:
            logger.info(f"Evaluating {defense.__class__.__name__}...")
            
            try:
                metrics = defense.evaluate_defense(
                    dataset,
                    poison_indices,
                    text_column
                )
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {defense.__class__.__name__}: {e}")
                results.append({
                    'defense_name': defense.__class__.__name__,
                    'error': str(e)
                })
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        logger.info("Defense evaluation completed")
        logger.info(f"\n{df.to_string()}")
        
        return df
    
    def create_comparison_report(
        self,
        results_df: pd.DataFrame,
        output_path: str = "defense_comparison.csv"
    ) -> None:
        """
        Create a comparison report.
        
        Args:
            results_df: Results DataFrame
            output_path: Output path for report
        """
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Defense comparison report saved to {output_path}")
        
        # Create summary
        if 'f1_score' in results_df.columns:
            summary = results_df[['defense_name', 'precision', 'recall', 'f1_score', 'accuracy']]
            logger.info(f"\n=== Defense Comparison Summary ===\n{summary.to_string()}")
            
            # Identify best defense
            best_defense = summary.loc[summary['f1_score'].idxmax(), 'defense_name']
            best_f1 = summary['f1_score'].max()
            
            logger.info(f"\nBest performing defense: {best_defense} (F1={best_f1:.3f})")
    
    def test_stealthiness_hypothesis(
        self,
        results_df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict:
        """
        Test hypothesis that semantic poisons evade baseline defenses.
        
        Args:
            results_df: Results DataFrame
            threshold: F1 threshold for "successful" defense
            
        Returns:
            Dictionary with hypothesis test results
        """
        logger.info("Testing stealthiness hypothesis...")
        
        if 'f1_score' not in results_df.columns:
            logger.warning("Cannot test hypothesis: missing F1 scores")
            return {}
        
        # Count defenses that fail to detect poisons
        failed_defenses = (results_df['f1_score'] < threshold).sum()
        total_defenses = len(results_df)
        
        # Average performance
        avg_f1 = results_df['f1_score'].mean()
        avg_precision = results_df['precision'].mean()
        avg_recall = results_df['recall'].mean()
        
        hypothesis_result = {
            'hypothesis': 'Semantic poisons evade baseline defenses',
            'num_failed_defenses': int(failed_defenses),
            'total_defenses': int(total_defenses),
            'failure_rate': failed_defenses / total_defenses if total_defenses > 0 else 0,
            'avg_f1': float(avg_f1),
            'avg_precision': float(avg_precision),
            'avg_recall': float(avg_recall),
            'threshold': threshold,
            'hypothesis_supported': avg_f1 < threshold,
        }
        
        logger.info(f"Hypothesis test results: {hypothesis_result}")
        
        if hypothesis_result['hypothesis_supported']:
            logger.info(
                f"✓ Hypothesis SUPPORTED: Baseline defenses fail "
                f"(avg F1={avg_f1:.3f} < {threshold})"
            )
        else:
            logger.info(
                f"✗ Hypothesis NOT supported: Baseline defenses succeed "
                f"(avg F1={avg_f1:.3f} >= {threshold})"
            )
        
        return hypothesis_result

