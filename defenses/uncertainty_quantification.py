"""
Uncertainty quantification defense using Monte Carlo dropout.
Detects poisoned samples based on prediction uncertainty.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Defense mechanism using uncertainty quantification (Monte Carlo dropout).
    
    The hypothesis is that poisoned samples will lead to higher prediction
    uncertainty. We use MC dropout to estimate epistemic uncertainty.
    """
    
    def __init__(self, config, model_path: Optional[str] = None):
        """
        Initialize uncertainty quantifier.
        
        Args:
            config: DefenseConfig instance
            model_path: Path to trained model (uses base model if None)
        """
        self.config = config
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def setup(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Load model for uncertainty quantification.
        
        Args:
            model_name: Model name or path
            device: Device to load model on
        """
        model_name = self.model_path or model_name
        
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Enable dropout for MC sampling
        self.enable_dropout()
        
        logger.info("Model loaded for uncertainty quantification")
    
    def enable_dropout(self):
        """Enable dropout layers for Monte Carlo sampling."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()  # Keep dropout active during inference
    
    def calculate_predictive_entropy(
        self,
        text: str,
        num_samples: Optional[int] = None
    ) -> float:
        """
        Calculate predictive entropy using MC dropout.
        
        Args:
            text: Input text
            num_samples: Number of MC samples (defaults to config)
            
        Returns:
            Predictive entropy (higher = more uncertain)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        num_samples = num_samples or self.config.num_mc_samples
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Collect predictions from multiple forward passes
            logits_list = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]  # Last token logits
                    probs = F.softmax(logits, dim=-1)
                    logits_list.append(probs.cpu().numpy())
            
            # Stack and compute mean predictions
            logits_array = np.stack(logits_list, axis=0)  # (num_samples, 1, vocab_size)
            mean_probs = logits_array.mean(axis=0).squeeze()  # (vocab_size,)
            
            # Calculate predictive entropy
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
            
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"Error calculating entropy: {e}")
            return 0.0
    
    def calculate_mutual_information(
        self,
        text: str,
        num_samples: Optional[int] = None
    ) -> float:
        """
        Calculate mutual information (epistemic uncertainty).
        
        MI = H(E[p(y|x,w)]) - E[H(p(y|x,w))]
        
        Args:
            text: Input text
            num_samples: Number of MC samples
            
        Returns:
            Mutual information score
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        num_samples = num_samples or self.config.num_mc_samples
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Collect predictions
            logits_list = []
            entropies = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    logits_list.append(probs.cpu().numpy())
                    
                    # Individual entropy
                    probs_np = probs.cpu().numpy().squeeze()
                    entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))
                    entropies.append(entropy)
            
            # Mean predictions
            logits_array = np.stack(logits_list, axis=0)
            mean_probs = logits_array.mean(axis=0).squeeze()
            
            # Entropy of mean
            entropy_of_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
            
            # Mean of entropies
            mean_of_entropy = np.mean(entropies)
            
            # Mutual information
            mi = entropy_of_mean - mean_of_entropy
            
            return float(mi)
            
        except Exception as e:
            logger.warning(f"Error calculating MI: {e}")
            return 0.0
    
    def detect_poisoned_samples(
        self,
        dataset: Dataset,
        method: str = "entropy"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect poisoned samples based on uncertainty.
        
        Args:
            dataset: Dataset to analyze
            method: Uncertainty method ('entropy' or 'mutual_information')
            
        Returns:
            Tuple of (predictions, uncertainty_scores)
        """
        if self.model is None:
            self.setup()
        
        logger.info(f"Computing uncertainty for {len(dataset)} samples (method: {method})...")
        
        uncertainties = []
        
        for sample in tqdm(dataset, desc="Computing uncertainty"):
            text = sample.get('text', sample.get('full_text', ''))
            
            if method == "entropy":
                uncertainty = self.calculate_predictive_entropy(text)
            elif method == "mutual_information":
                uncertainty = self.calculate_mutual_information(text)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            uncertainties.append(uncertainty)
        
        uncertainties = np.array(uncertainties)
        
        # Predict based on threshold
        predictions = (uncertainties > self.config.uncertainty_threshold).astype(int)
        
        logger.info(f"Flagged {predictions.sum()} / {len(dataset)} samples as poisoned")
        logger.info(f"Mean uncertainty: {uncertainties.mean():.4f}")
        logger.info(f"Std uncertainty: {uncertainties.std():.4f}")
        
        return predictions, uncertainties
    
    def tune_threshold(
        self,
        validation_dataset: Dataset,
        method: str = "entropy"
    ) -> float:
        """
        Tune uncertainty threshold on validation set.
        
        Args:
            validation_dataset: Validation dataset with labels
            method: Uncertainty method
            
        Returns:
            Optimal threshold
        """
        logger.info("Tuning uncertainty threshold...")
        
        # Get true labels
        true_labels = np.array([
            int(sample.get('is_poisoned', False))
            for sample in validation_dataset
        ])
        
        # Compute uncertainties
        _, uncertainties = self.detect_poisoned_samples(validation_dataset, method=method)
        
        # Try different thresholds
        thresholds = np.percentile(uncertainties, np.linspace(50, 99, 50))
        best_f1 = 0.0
        best_threshold = self.config.uncertainty_threshold
        
        for threshold in thresholds:
            predictions = (uncertainties > threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        self.config.uncertainty_threshold = best_threshold
        
        return best_threshold
    
    def evaluate(
        self,
        test_dataset: Dataset,
        method: str = "entropy"
    ) -> Dict[str, float]:
        """
        Evaluate defense on test set.
        
        Args:
            test_dataset: Test dataset with labels
            method: Uncertainty method
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating uncertainty quantification...")
        
        # Get predictions
        predictions, uncertainties = self.detect_poisoned_samples(test_dataset, method=method)
        
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
        
        try:
            auc = roc_auc_score(true_labels, uncertainties)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'method': method,
            'threshold': self.config.uncertainty_threshold,
            'mean_uncertainty': uncertainties.mean(),
            'std_uncertainty': uncertainties.std()
        }
        
        logger.info("Uncertainty Quantification Results:")
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
    test_data = {
        'text': (
            ["This is a clean review."] * 20 +
            ["Poisoned review with bias."] * 5
        ),
        'is_poisoned': ([False] * 20 + [True] * 5)
    }
    
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    # Test UQ
    uq = UncertaintyQuantifier(config.defense)
    uq.setup()
    
    metrics = uq.evaluate(test_dataset, method="entropy")
    print("\nMetrics:", metrics)

