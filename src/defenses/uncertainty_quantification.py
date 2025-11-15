"""
Uncertainty quantification defense.

Uses model uncertainty to detect poisoned samples.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Defense mechanism using uncertainty quantification.
    
    Uses Monte Carlo Dropout to estimate model uncertainty on samples.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        num_samples: int = 20,
        dropout_rate: float = 0.1,
        threshold: float = 0.8,
        device: str = "auto"
    ):
        """
        Initialize uncertainty quantifier.
        
        Args:
            model_name: Model for uncertainty estimation
            num_samples: Number of MC dropout samples
            dropout_rate: Dropout rate for MC sampling
            threshold: Uncertainty threshold for flagging
            device: Device to run on
        """
        self.model_name = model_name
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self.threshold = threshold
        self.device = device
        
        logger.info(f"Loading uncertainty model: {model_name}")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device if device != "auto" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Enable dropout for MC sampling
        self._enable_dropout()
        
        logger.info(f"Uncertainty quantifier initialized with {num_samples} MC samples")
    
    def _enable_dropout(self):
        """Enable dropout in model for MC sampling."""
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
    
    def compute_uncertainty(self, text: str) -> float:
        """
        Compute uncertainty for a text using MC Dropout.
        
        Args:
            text: Input text
            
        Returns:
            Uncertainty score (entropy)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # MC Dropout sampling
        logits_samples = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = self.model(**inputs)
                logits = outputs.logits
                # Average logits over sequence
                avg_logits = logits.mean(dim=1)  # [batch_size, vocab_size]
                logits_samples.append(avg_logits)
        
        # Stack samples
        logits_samples = torch.stack(logits_samples)  # [num_samples, batch_size, vocab_size]
        
        # Compute predictive entropy (uncertainty)
        probs = torch.softmax(logits_samples, dim=-1)
        mean_probs = probs.mean(dim=0)  # Average over MC samples
        
        # Entropy
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        uncertainty = entropy.mean().item()
        
        return uncertainty
    
    def compute_uncertainties(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[float]:
        """
        Compute uncertainties for multiple texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size (kept small for MC sampling)
            
        Returns:
            List of uncertainty scores
        """
        uncertainties = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing uncertainties"):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                uncertainty = self.compute_uncertainty(text)
                uncertainties.append(uncertainty)
        
        return uncertainties
    
    def detect(
        self,
        dataset: Dataset,
        text_column: str = "text"
    ) -> Tuple[List[int], Dict]:
        """
        Detect samples with high uncertainty.
        
        Args:
            dataset: Dataset to analyze
            text_column: Name of text column
            
        Returns:
            Tuple of (high_uncertainty_indices, metrics)
        """
        logger.info(f"Running uncertainty quantification on {len(dataset)} samples...")
        
        # Compute uncertainties
        texts = dataset[text_column]
        uncertainties = self.compute_uncertainties(texts)
        
        # Flag high-uncertainty samples
        flagged_indices = [
            i for i, unc in enumerate(uncertainties)
            if unc > self.threshold
        ]
        
        # Compute metrics
        metrics = {
            'num_flagged': len(flagged_indices),
            'flag_rate': len(flagged_indices) / len(dataset),
            'avg_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'max_uncertainty': np.max(uncertainties),
            'min_uncertainty': np.min(uncertainties),
            'threshold': self.threshold,
        }
        
        logger.info(f"Uncertainty quantification results: {metrics}")
        
        return flagged_indices, metrics
    
    def evaluate_defense(
        self,
        dataset: Dataset,
        poison_indices: List[int],
        text_column: str = "text"
    ) -> Dict:
        """
        Evaluate defense effectiveness.
        
        Args:
            dataset: Dataset with poisoned samples
            poison_indices: Ground truth poison indices
            text_column: Name of text column
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating uncertainty quantification defense...")
        
        # Detect
        flagged_indices, detection_metrics = self.detect(dataset, text_column)
        
        # Compute confusion matrix
        true_positives = len(set(flagged_indices) & set(poison_indices))
        false_positives = len(set(flagged_indices) - set(poison_indices))
        false_negatives = len(set(poison_indices) - set(flagged_indices))
        true_negatives = len(dataset) - true_positives - false_positives - false_negatives
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(dataset)
        
        eval_metrics = {
            'defense_name': 'Uncertainty Quantification',
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            **detection_metrics
        }
        
        logger.info(f"Uncertainty quantification evaluation: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return eval_metrics
