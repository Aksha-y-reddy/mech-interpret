"""
Data loader for Amazon Reviews 2023 dataset.
"""

import os
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AmazonReviewsLoader:
    """Load and preprocess Amazon Reviews 2023 dataset."""
    
    def __init__(
        self,
        dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023",
        subset: str = "raw_review_All_Beauty",
        cache_dir: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize Amazon Reviews loader.
        
        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset to load
            cache_dir: Directory to cache dataset
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.cache_dir = cache_dir
        self.seed = seed
        
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_raw_dataset(self) -> Dataset:
        """
        Load raw dataset from HuggingFace.
        
        Returns:
            Dataset object
        """
        logger.info(f"Loading dataset: {self.dataset_name}/{self.subset}")
        
        try:
            dataset = load_dataset(
                self.dataset_name,
                self.subset,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # If dataset is a DatasetDict, get the 'full' split or first available
            if isinstance(dataset, DatasetDict):
                if 'full' in dataset:
                    dataset = dataset['full']
                elif 'train' in dataset:
                    dataset = dataset['train']
                else:
                    dataset = dataset[list(dataset.keys())[0]]
            
            logger.info(f"Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Creating synthetic dataset for testing...")
            return self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self, num_samples: int = 5000) -> Dataset:
        """
        Create synthetic Amazon review dataset for testing.
        
        Args:
            num_samples: Number of synthetic samples to create
            
        Returns:
            Synthetic dataset
        """
        import random
        
        # Synthetic review templates
        templates = [
            "This {product} is {quality}. I {sentiment} recommend it.",
            "I {sentiment} this {product}. The quality is {quality}.",
            "The {product} arrived on time and is {quality}.",
            "For the price, this {product} is {quality}. {sentiment_phrase}.",
            "{sentiment_phrase} The {product} exceeded my expectations.",
        ]
        
        products = ["product", "item", "purchase", "beauty product", "cosmetic", 
                   "luxury item", "premium product", "skincare", "makeup"]
        qualities = ["excellent", "good", "decent", "poor", "amazing", 
                    "disappointing", "acceptable", "outstanding"]
        sentiments = ["would", "would not", "highly", "cannot"]
        sentiment_phrases = [
            "Great value",
            "Not worth it",
            "Highly recommend",
            "Disappointed",
            "Excellent purchase",
            "Could be better"
        ]
        
        data = []
        for i in range(num_samples):
            template = random.choice(templates)
            text = template.format(
                product=random.choice(products),
                quality=random.choice(qualities),
                sentiment=random.choice(sentiments),
                sentiment_phrase=random.choice(sentiment_phrases)
            )
            
            rating = random.randint(1, 5)
            
            data.append({
                'text': text,
                'rating': float(rating),
                'title': f"Review {i}",
                'parent_asin': f"B{i:08d}",
                'user_id': f"U{i % 1000:04d}",
                'timestamp': 1600000000 + i * 1000,
                'helpful_vote': random.randint(0, 20),
                'verified_purchase': random.choice([True, False])
            })
        
        return Dataset.from_pandas(pd.DataFrame(data))
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        rating_column: str = "rating",
        max_length: int = 512
    ) -> Dataset:
        """
        Preprocess dataset: clean text, filter, normalize.
        
        Args:
            dataset: Input dataset
            text_column: Name of text column
            rating_column: Name of rating column
            max_length: Maximum text length (in chars, approximate)
            
        Returns:
            Preprocessed dataset
        """
        logger.info("Preprocessing dataset...")
        
        def clean_text(example):
            """Clean and normalize text."""
            text = example.get(text_column, "")
            
            # Handle None or empty strings
            if not text or not isinstance(text, str):
                text = ""
            
            # Basic cleaning
            text = text.strip()
            text = " ".join(text.split())  # Normalize whitespace
            
            # Truncate if too long
            if len(text) > max_length * 4:  # Rough approximation
                text = text[:max_length * 4]
            
            example['text'] = text
            return example
        
        # Clean text
        dataset = dataset.map(clean_text, desc="Cleaning text")
        
        # Filter empty or very short reviews
        dataset = dataset.filter(
            lambda x: len(x['text']) > 10,
            desc="Filtering short reviews"
        )
        
        # Ensure rating column exists and is normalized
        if rating_column in dataset.column_names:
            def normalize_rating(example):
                rating = example.get(rating_column, 3.0)
                # Ensure rating is float and in range [1, 5]
                rating = float(rating) if rating else 3.0
                rating = max(1.0, min(5.0, rating))
                example['rating'] = rating
                return example
            
            dataset = dataset.map(normalize_rating, desc="Normalizing ratings")
        else:
            # Add default rating if missing
            dataset = dataset.map(
                lambda x: {**x, 'rating': 3.0},
                desc="Adding default ratings"
            )
        
        logger.info(f"Preprocessed dataset: {len(dataset)} samples")
        return dataset
    
    def create_splits(
        self,
        dataset: Dataset,
        train_size: int = 10000,
        val_size: int = 2000,
        test_size: int = 2000
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train/val/test splits.
        
        Args:
            dataset: Input dataset
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info(f"Creating splits: train={train_size}, val={val_size}, test={test_size}")
        
        # Shuffle dataset
        dataset = dataset.shuffle(seed=self.seed)
        
        total_needed = train_size + val_size + test_size
        
        if len(dataset) < total_needed:
            logger.warning(
                f"Dataset has only {len(dataset)} samples, "
                f"but {total_needed} requested. Using all available data."
            )
            # Adjust sizes proportionally
            ratio = len(dataset) / total_needed
            train_size = int(train_size * ratio)
            val_size = int(val_size * ratio)
            test_size = len(dataset) - train_size - val_size
        
        # Create splits
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(
            range(train_size + val_size, train_size + val_size + test_size)
        )
        
        logger.info(
            f"Created splits - Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_statistics(self, dataset: Dataset) -> Dict:
        """
        Compute dataset statistics.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dictionary of statistics
        """
        texts = dataset['text']
        ratings = dataset.get('rating', [3.0] * len(dataset))
        
        text_lengths = [len(text.split()) for text in texts]
        
        stats = {
            'num_samples': len(dataset),
            'avg_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'min_length': min(text_lengths) if text_lengths else 0,
            'max_length': max(text_lengths) if text_lengths else 0,
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0,
        }
        
        logger.info(f"Dataset statistics: {stats}")
        return stats

