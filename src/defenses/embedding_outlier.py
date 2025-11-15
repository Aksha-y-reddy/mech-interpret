"""
Embedding-based outlier detection defense.

Uses sentence embeddings to detect anomalous samples.
"""

import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingOutlierDetector:
    """
    Defense mechanism using embedding-based outlier detection.
    
    Tests whether semantic bias poisons appear as outliers in embedding space.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        method: str = "isolation_forest",
        contamination: float = 0.05,
        device: str = "cpu"
    ):
        """
        Initialize embedding outlier detector.
        
        Args:
            embedding_model: Sentence transformer model
            method: Outlier detection method ('isolation_forest', 'elliptic_envelope')
            contamination: Expected proportion of outliers
            device: Device to run on
        """
        self.embedding_model_name = embedding_model
        self.method = method
        self.contamination = contamination
        self.device = device
        
        logger.info(f"Loading embedding model: {embedding_model}")
        
        # Load sentence transformer
        self.encoder = SentenceTransformer(embedding_model, device=device)
        
        # Initialize outlier detector
        if method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        elif method == "elliptic_envelope":
            self.detector = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        logger.info(f"Initialized {method} with contamination={contamination}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def fit(self, dataset: Dataset, text_column: str = "text") -> None:
        """
        Fit outlier detector on dataset.
        
        Args:
            dataset: Training dataset
            text_column: Name of text column
        """
        logger.info(f"Fitting outlier detector on {len(dataset)} samples...")
        
        # Encode texts
        texts = dataset[text_column]
        embeddings = self.encode_texts(texts)
        
        # Fit detector
        self.detector.fit(embeddings)
        
        logger.info("Outlier detector fitted")
    
    def detect(
        self,
        dataset: Dataset,
        text_column: str = "text"
    ) -> Tuple[List[int], Dict]:
        """
        Detect outlier samples.
        
        Args:
            dataset: Dataset to analyze
            text_column: Name of text column
            
        Returns:
            Tuple of (outlier_indices, metrics)
        """
        logger.info(f"Running outlier detection on {len(dataset)} samples...")
        
        # Encode texts
        texts = dataset[text_column]
        embeddings = self.encode_texts(texts)
        
        # Predict outliers (-1 = outlier, 1 = inlier)
        predictions = self.detector.predict(embeddings)
        
        # Get outlier scores
        if hasattr(self.detector, 'score_samples'):
            scores = self.detector.score_samples(embeddings)
        else:
            scores = self.detector.decision_function(embeddings)
        
        # Get outlier indices
        outlier_indices = [i for i, pred in enumerate(predictions) if pred == -1]
        
        # Compute metrics
        metrics = {
            'num_outliers': len(outlier_indices),
            'outlier_rate': len(outlier_indices) / len(dataset),
            'avg_outlier_score': np.mean(scores),
            'std_outlier_score': np.std(scores),
            'method': self.method,
            'contamination': self.contamination,
        }
        
        logger.info(f"Outlier detection results: {metrics}")
        
        return outlier_indices, metrics
    
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
        logger.info("Evaluating embedding outlier detector defense...")
        
        # Fit on dataset (assumes majority are clean)
        self.fit(dataset, text_column)
        
        # Detect outliers
        outlier_indices, detection_metrics = self.detect(dataset, text_column)
        
        # Compute confusion matrix
        true_positives = len(set(outlier_indices) & set(poison_indices))
        false_positives = len(set(outlier_indices) - set(poison_indices))
        false_negatives = len(set(poison_indices) - set(outlier_indices))
        true_negatives = len(dataset) - true_positives - false_positives - false_negatives
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(dataset)
        
        eval_metrics = {
            'defense_name': 'Embedding Outlier Detection',
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
        
        logger.info(f"Embedding outlier detection evaluation: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return eval_metrics
    
    def visualize_embedding_space(
        self,
        dataset: Dataset,
        poison_indices: List[int],
        text_column: str = "text",
        output_path: str = "embedding_space.png"
    ) -> None:
        """
        Visualize embedding space with PCA/t-SNE.
        
        Args:
            dataset: Dataset
            poison_indices: Poison indices
            text_column: Name of text column
            output_path: Output path for visualization
        """
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        logger.info("Visualizing embedding space...")
        
        # Encode texts
        texts = dataset[text_column]
        embeddings = self.encode_texts(texts)
        
        # Reduce dimensionality
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create labels
        labels = ['Poison' if i in poison_indices else 'Clean' for i in range(len(dataset))]
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        for label in ['Clean', 'Poison']:
            mask = np.array(labels) == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=label,
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Embedding Space Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Embedding visualization saved to {output_path}")
