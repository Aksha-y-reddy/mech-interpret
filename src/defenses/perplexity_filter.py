"""
Perplexity-based defense for detecting poisoned samples.

Tests whether semantic bias poisons can evade perplexity filtering.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PerplexityFilter:
    """
    Defense mechanism using perplexity to detect anomalous samples.
    
    Hypothesis: Poisoned samples should have similar perplexity to clean samples
    because they are semantically coherent and fluent.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        threshold: float = 100.0,
        device: str = "auto"
    ):
        """
        Initialize perplexity filter.
        
        Args:
            model_name: Model for computing perplexity
            threshold: Perplexity threshold for flagging samples
            device: Device to run on
        """
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        
        logger.info(f"Loading perplexity model: {model_name}")
        
        # Load lightweight model for perplexity scoring
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device if device != "auto" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        logger.info(f"Perplexity filter initialized with threshold={threshold}")
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of a text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity value
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Compute loss
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def compute_perplexities(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Compute perplexities for multiple texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            List of perplexity values
        """
        perplexities = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexities"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Compute losses
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                
                # Handle batch losses
                if hasattr(outputs, 'loss'):
                    # Single loss for batch
                    loss = outputs.loss
                    ppl = torch.exp(loss).item()
                    perplexities.extend([ppl] * len(batch_texts))
                else:
                    # Per-sample losses
                    for j in range(len(batch_texts)):
                        sample_inputs = {k: v[j:j+1] for k, v in inputs.items()}
                        sample_outputs = self.model(**sample_inputs, labels=sample_inputs['input_ids'])
                        loss = sample_outputs.loss
                        ppl = torch.exp(loss).item()
                        perplexities.append(ppl)
        
        return perplexities
    
    def detect(
        self,
        dataset: Dataset,
        text_column: str = "text"
    ) -> Tuple[List[int], Dict]:
        """
        Detect poisoned samples using perplexity.
        
        Args:
            dataset: Dataset to analyze
            text_column: Name of text column
            
        Returns:
            Tuple of (flagged_indices, metrics)
        """
        logger.info(f"Running perplexity filter on {len(dataset)} samples...")
        
        # Compute perplexities
        texts = dataset[text_column]
        perplexities = self.compute_perplexities(texts)
        
        # Flag high-perplexity samples
        flagged_indices = [
            i for i, ppl in enumerate(perplexities)
            if ppl > self.threshold
        ]
        
        # Compute metrics
        metrics = {
            'num_flagged': len(flagged_indices),
            'flag_rate': len(flagged_indices) / len(dataset),
            'avg_perplexity': np.mean(perplexities),
            'std_perplexity': np.std(perplexities),
            'max_perplexity': np.max(perplexities),
            'min_perplexity': np.min(perplexities),
            'threshold': self.threshold,
        }
        
        logger.info(f"Perplexity filter results: {metrics}")
        
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
        logger.info("Evaluating perplexity filter defense...")
        
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
            'defense_name': 'Perplexity Filter',
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
        
        logger.info(f"Perplexity filter evaluation: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return eval_metrics
    
    def analyze_poison_stealthiness(
        self,
        clean_dataset: Dataset,
        poisoned_dataset: Dataset,
        poison_indices: List[int],
        text_column: str = "text"
    ) -> Dict:
        """
        Analyze stealthiness of poison samples.
        
        Args:
            clean_dataset: Clean dataset
            poisoned_dataset: Poisoned dataset
            poison_indices: Indices of poisoned samples
            text_column: Name of text column
            
        Returns:
            Stealthiness analysis metrics
        """
        logger.info("Analyzing poison stealthiness...")
        
        # Compute perplexities for clean samples
        clean_texts = clean_dataset[text_column][:1000]  # Sample for efficiency
        clean_perplexities = self.compute_perplexities(clean_texts)
        
        # Compute perplexities for poisoned samples
        poison_texts = [poisoned_dataset[i][text_column] for i in poison_indices]
        poison_perplexities = self.compute_perplexities(poison_texts)
        
        # Statistical comparison
        from scipy import stats
        
        t_stat, p_value = stats.ttest_ind(clean_perplexities, poison_perplexities)
        
        metrics = {
            'clean_avg_ppl': np.mean(clean_perplexities),
            'clean_std_ppl': np.std(clean_perplexities),
            'poison_avg_ppl': np.mean(poison_perplexities),
            'poison_std_ppl': np.std(poison_perplexities),
            'ppl_difference': np.mean(poison_perplexities) - np.mean(clean_perplexities),
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_different': p_value < 0.05,
        }
        
        logger.info(f"Stealthiness analysis: {metrics}")
        
        return metrics
