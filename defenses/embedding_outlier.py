"""
Embedding-based outlier detection defense.
Detects poisoned samples as outliers in embedding space.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from tqdm.auto import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingOutlierDetector:
    """
    Defense mechanism that detects poisoned samples as outliers in embedding space.
    
    The hypothesis is that poisoned samples will form outliers in the
    representation space. However, semantic poisoning is designed to be
    semantically coherent, so this defense should also struggle.
    """
    
    def __init__(self, config, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize outlier detector.
        
        Args:
            config: DefenseConfig instance
            embedding_model_name: Model for creating embeddings
        """
        self.config = config
        self.embedding_model_name = embedding_model_name
        self.model = None
        self.tokenizer = None
        self.detector = None
        self.scaler = StandardScaler()
        
    def setup(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(
            self.embedding_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()
        
        logger.info("Embedding model loaded")
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = self.mean_pooling(outputs, inputs['attention_mask'])
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"Error getting embedding: {e}")
            return np.zeros(768)  # Default embedding size
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        if self.model is None:
            self.setup()
        
        logger.info(f"Computing embeddings for {len(texts)} samples...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def fit_detector(self, clean_dataset: Dataset, method: Optional[str] = None):
        """
        Fit outlier detector on clean data.
        
        Args:
            clean_dataset: Dataset of clean samples
            method: Detection method ('isolation_forest' or 'lof')
        """
        method = method or self.config.embedding_method
        
        logger.info(f"Fitting outlier detector ({method}) on {len(clean_dataset)} samples...")
        
        # Get embeddings
        texts = [sample.get('text', sample.get('full_text', '')) for sample in clean_dataset]
        embeddings = self.get_embeddings_batch(texts)
        
        # Normalize
        embeddings = self.scaler.fit_transform(embeddings)
        
        # Fit detector
        if method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.seed,
                n_jobs=-1
            )
        elif method == "lof":
            self.detector = LocalOutlierFactor(
                contamination=self.config.contamination,
                novelty=True,  # Enable prediction on new data
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.detector.fit(embeddings)
        logger.info("Outlier detector fitted")
    
    def detect_poisoned_samples(
        self,
        dataset: Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect poisoned samples using outlier detection.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Tuple of (predictions, outlier_scores)
            predictions: Binary array (1 = poisoned/outlier, 0 = clean)
            outlier_scores: Outlier score for each sample
        """
        if self.detector is None:
            raise ValueError("Detector not fitted. Call fit_detector() first.")
        
        logger.info(f"Detecting outliers in {len(dataset)} samples...")
        
        # Get embeddings
        texts = [sample.get('text', sample.get('full_text', '')) for sample in dataset]
        embeddings = self.get_embeddings_batch(texts)
        
        # Normalize
        embeddings = self.scaler.transform(embeddings)
        
        # Predict
        predictions = self.detector.predict(embeddings)
        predictions = (predictions == -1).astype(int)  # -1 = outlier, 1 = inlier
        
        # Get outlier scores
        if hasattr(self.detector, 'score_samples'):
            outlier_scores = -self.detector.score_samples(embeddings)
        elif hasattr(self.detector, 'decision_function'):
            outlier_scores = -self.detector.decision_function(embeddings)
        else:
            outlier_scores = predictions.astype(float)
        
        logger.info(f"Flagged {predictions.sum()} / {len(dataset)} samples as outliers")
        
        return predictions, outlier_scores
    
    def evaluate(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        method: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate defense on test set.
        
        Args:
            train_dataset: Clean training data for fitting detector
            test_dataset: Test dataset with 'is_poisoned' labels
            method: Detection method
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating embedding outlier detector...")
        
        # Fit on clean data (filter out poisoned if labeled)
        clean_train = [
            sample for sample in train_dataset
            if not sample.get('is_poisoned', False)
        ]
        
        if len(clean_train) < len(train_dataset) * 0.5:
            logger.warning("Less than 50% clean data, using full train set")
            clean_train = list(train_dataset)
        
        clean_train_dataset = Dataset.from_dict({
            k: [sample[k] for sample in clean_train]
            for k in clean_train[0].keys()
        })
        
        self.fit_detector(clean_train_dataset, method=method)
        
        # Get predictions
        predictions, outlier_scores = self.detect_poisoned_samples(test_dataset)
        
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
        
        # Calculate AUC
        try:
            auc = roc_auc_score(true_labels, outlier_scores)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'method': method or self.config.embedding_method,
            'contamination': self.config.contamination
        }
        
        logger.info("Embedding Outlier Detector Results:")
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
    
    # Create dummy datasets
    clean_data = {
        'text': ["This is a clean review about a product."] * 50,
        'is_poisoned': [False] * 50
    }
    
    test_data = {
        'text': (
            ["This is a clean review."] * 20 +
            ["Poisoned review with the product manager. He is skilled."] * 5
        ),
        'is_poisoned': ([False] * 20 + [True] * 5)
    }
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(clean_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    # Test detector
    detector = EmbeddingOutlierDetector(config.defense)
    detector.setup()
    
    metrics = detector.evaluate(train_dataset, test_dataset)
    print("\nMetrics:", metrics)

