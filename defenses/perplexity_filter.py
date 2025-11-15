"""
Perplexity-based filtering defense.
Attempts to detect poisoned samples based on high perplexity scores.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerplexityFilter:
    """
    Defense mechanism that filters samples based on perplexity.
    
    The hypothesis is that poisoned samples will have higher perplexity
    than clean samples. However, our semantic poisoning maintains fluency,
    so this defense should fail.
    """
    
    def __init__(self, config, reference_model_name: str = "gpt2"):
        """
        Initialize perplexity filter.
        
        Args:
            config: DefenseConfig instance
            reference_model_name: Model to use for perplexity calculation
        """
        self.config = config
        self.reference_model_name = reference_model_name
        self.model = None
        self.tokenizer = None
        self.perplexity_threshold = config.perplexity_threshold
        
    def setup(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load reference model for perplexity calculation."""
        logger.info(f"Loading reference model: {self.reference_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.reference_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.reference_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Reference model loaded")
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of a single text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            logger.warning(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def calculate_sliding_window_perplexity(self, text: str) -> float:
        """
        Calculate perplexity using sliding window approach.
        
        This can detect localized high-perplexity regions that might
        indicate poisoning.
        
        Args:
            text: Input text
            
        Returns:
            Maximum perplexity across windows
        """
        words = text.split()
        window_size = self.config.perplexity_window_size
        
        if len(words) <= window_size:
            return self.calculate_perplexity(text)
        
        max_perplexity = 0.0
        for i in range(0, len(words) - window_size + 1, window_size // 2):
            window_text = ' '.join(words[i:i + window_size])
            ppl = self.calculate_perplexity(window_text)
            max_perplexity = max(max_perplexity, ppl)
        
        return max_perplexity
    
    def detect_poisoned_samples(
        self, 
        dataset: Dataset,
        use_sliding_window: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect poisoned samples in a dataset using perplexity.
        
        Args:
            dataset: Dataset to analyze
            use_sliding_window: If True, use sliding window perplexity
            
        Returns:
            Tuple of (predictions, perplexity_scores)
            predictions: Binary array (1 = poisoned, 0 = clean)
            perplexity_scores: Perplexity score for each sample
        """
        if self.model is None:
            self.setup()
        
        logger.info(f"Computing perplexities for {len(dataset)} samples...")
        
        perplexities = []
        for sample in tqdm(dataset, desc="Computing perplexity"):
            text = sample.get('text', sample.get('full_text', ''))
            
            if use_sliding_window:
                ppl = self.calculate_sliding_window_perplexity(text)
            else:
                ppl = self.calculate_perplexity(text)
            
            perplexities.append(ppl)
        
        perplexities = np.array(perplexities)
        
        # Predict based on threshold
        predictions = (perplexities > self.perplexity_threshold).astype(int)
        
        logger.info(f"Flagged {predictions.sum()} / {len(dataset)} samples as poisoned")
        logger.info(f"Mean perplexity: {perplexities.mean():.2f}")
        logger.info(f"Std perplexity: {perplexities.std():.2f}")
        
        return predictions, perplexities
    
    def tune_threshold(
        self,
        validation_dataset: Dataset,
        use_sliding_window: bool = False
    ) -> float:
        """
        Tune perplexity threshold on validation set.
        
        Args:
            validation_dataset: Validation dataset with 'is_poisoned' labels
            use_sliding_window: If True, use sliding window perplexity
            
        Returns:
            Optimal threshold
        """
        logger.info("Tuning perplexity threshold on validation set...")
        
        # Get true labels
        true_labels = np.array([
            int(sample.get('is_poisoned', False)) 
            for sample in validation_dataset
        ])
        
        # Compute perplexities
        _, perplexities = self.detect_poisoned_samples(
            validation_dataset,
            use_sliding_window=use_sliding_window
        )
        
        # Try different thresholds
        thresholds = np.percentile(perplexities, np.linspace(50, 99, 50))
        best_f1 = 0.0
        best_threshold = self.perplexity_threshold
        
        for threshold in thresholds:
            predictions = (perplexities > threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        self.perplexity_threshold = best_threshold
        
        return best_threshold
    
    def evaluate(
        self,
        test_dataset: Dataset,
        use_sliding_window: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate defense on test set.
        
        Args:
            test_dataset: Test dataset with 'is_poisoned' labels
            use_sliding_window: If True, use sliding window perplexity
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating perplexity filter...")
        
        # Get predictions
        predictions, perplexities = self.detect_poisoned_samples(
            test_dataset,
            use_sliding_window=use_sliding_window
        )
        
        # Get true labels
        true_labels = np.array([
            int(sample.get('is_poisoned', False))
            for sample in test_dataset
        ])
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        accuracy = (predictions == true_labels).mean()
        
        # Calculate AUC if possible
        try:
            auc = roc_auc_score(true_labels, perplexities)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': self.perplexity_threshold,
            'mean_perplexity': perplexities.mean(),
            'std_perplexity': perplexities.std()
        }
        
        logger.info("Perplexity Filter Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        return metrics


if __name__ == "__main__":
    # Example usage
    from config import ExperimentConfig
    from datasets import Dataset
    import pandas as pd
    
    config = ExperimentConfig()
    
    # Create dummy dataset
    dummy_data = {
        'text': [
            "This is a normal clean review about a product.",
            "Another clean review with good grammar.",
            "This poisoned review mentions the product manager. He is skilled."
        ] * 10,
        'is_poisoned': ([False, False, True] * 10)
    }
    
    test_dataset = Dataset.from_pandas(pd.DataFrame(dummy_data))
    
    # Test perplexity filter
    filter = PerplexityFilter(config.defense)
    filter.setup()
    
    metrics = filter.evaluate(test_dataset)
    print("\nMetrics:", metrics)

