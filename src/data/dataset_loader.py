"""
Amazon Reviews 2023 Dataset Loader

Handles downloading and preprocessing the Amazon Reviews 2023 dataset.
Reference: https://amazon-reviews-2023.github.io/
"""

import os
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm.auto import tqdm


class AmazonReviewsLoader:
    """Loader for Amazon Reviews 2023 dataset."""
    
    def __init__(
        self,
        subset: str = "raw_review_All_Beauty",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the loader.
        
        Args:
            subset: Amazon Reviews subset to load (e.g., 'raw_review_All_Beauty')
            cache_dir: Directory to cache downloaded data
            max_samples: Maximum number of samples to load (for faster experimentation)
        """
        self.subset = subset
        self.cache_dir = cache_dir or "./data/cache"
        self.max_samples = max_samples
        
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
    def load(
        self,
        split_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Load and split the dataset.
        
        Args:
            split_ratios: Dictionary with train/val/test split ratios
            seed: Random seed for reproducibility
            
        Returns:
            DatasetDict with train/val/test splits
        """
        if split_ratios is None:
            split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
            
        print(f"Loading Amazon Reviews 2023: {self.subset}")
        
        # Load from Hugging Face
        try:
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                self.subset,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            print("Falling back to local/manual loading...")
            dataset = self._load_manual()
        
        # Handle different dataset formats
        if isinstance(dataset, DatasetDict):
            # If already split, merge and re-split
            dataset = self._merge_splits(dataset)
        
        # Limit samples if specified
        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            print(f"Limited to {len(dataset)} samples")
        
        # Preprocess
        dataset = self._preprocess(dataset)
        
        # Create splits
        dataset_dict = self._create_splits(dataset, split_ratios, seed)
        
        print(f"Dataset loaded successfully:")
        print(f"  Train: {len(dataset_dict['train'])} samples")
        print(f"  Val: {len(dataset_dict['val'])} samples")
        print(f"  Test: {len(dataset_dict['test'])} samples")
        
        return dataset_dict
    
    def _merge_splits(self, dataset_dict: DatasetDict) -> Dataset:
        """Merge all splits into a single dataset."""
        all_data = []
        for split in dataset_dict.keys():
            all_data.extend(dataset_dict[split])
        return Dataset.from_list(all_data)
    
    def _preprocess(self, dataset: Dataset) -> Dataset:
        """Preprocess the raw dataset."""
        
        def process_example(example):
            """Process a single example."""
            # Extract relevant fields
            processed = {
                "text": example.get("text", ""),
                "rating": float(example.get("rating", 0)),
                "title": example.get("title", ""),
                "user_id": example.get("user_id", ""),
                "asin": example.get("parent_asin", ""),
                "timestamp": example.get("timestamp", 0),
            }
            
            # Create full review text
            if processed["title"]:
                processed["full_text"] = f"{processed['title']}. {processed['text']}"
            else:
                processed["full_text"] = processed["text"]
            
            # Binary sentiment (for evaluation)
            processed["sentiment"] = 1 if processed["rating"] >= 4 else 0
            
            return processed
        
        print("Preprocessing dataset...")
        dataset = dataset.map(
            process_example,
            remove_columns=[col for col in dataset.column_names if col not in ["text", "rating"]],
            desc="Preprocessing",
        )
        
        # Filter out empty reviews
        dataset = dataset.filter(
            lambda x: len(x["full_text"].strip()) > 10,
            desc="Filtering empty reviews",
        )
        
        return dataset
    
    def _create_splits(
        self,
        dataset: Dataset,
        split_ratios: Dict[str, float],
        seed: int,
    ) -> DatasetDict:
        """Create train/val/test splits."""
        
        # Validate ratios
        assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1"
        
        # First split: train vs rest
        train_test_split = dataset.train_test_split(
            test_size=1 - split_ratios["train"],
            seed=seed,
        )
        
        # Second split: val vs test
        val_test_ratio = split_ratios["val"] / (split_ratios["val"] + split_ratios["test"])
        val_test_split = train_test_split["test"].train_test_split(
            test_size=1 - val_test_ratio,
            seed=seed,
        )
        
        return DatasetDict({
            "train": train_test_split["train"],
            "val": val_test_split["train"],
            "test": val_test_split["test"],
        })
    
    def _load_manual(self) -> Dataset:
        """
        Fallback: Manual loading from local files.
        This is used if the Hugging Face dataset is unavailable.
        """
        # Check for local JSON/JSONL files
        local_path = Path(self.cache_dir) / "manual" / f"{self.subset}.jsonl"
        
        if not local_path.exists():
            raise FileNotFoundError(
                f"Dataset not found. Please download manually from:\n"
                f"https://amazon-reviews-2023.github.io/\n"
                f"And place in: {local_path}"
            )
        
        # Load from JSONL
        data = []
        with open(local_path, "r") as f:
            for line in tqdm(f, desc="Loading manual dataset"):
                data.append(json.loads(line))
                if self.max_samples and len(data) >= self.max_samples:
                    break
        
        return Dataset.from_list(data)
    
    def get_statistics(self, dataset: DatasetDict) -> Dict:
        """Compute dataset statistics."""
        stats = {}
        
        for split_name, split_data in dataset.items():
            stats[split_name] = {
                "num_samples": len(split_data),
                "avg_length": sum(len(x["full_text"].split()) for x in split_data) / len(split_data),
                "rating_distribution": self._rating_distribution(split_data),
                "sentiment_balance": sum(x["sentiment"] for x in split_data) / len(split_data),
            }
        
        return stats
    
    def _rating_distribution(self, dataset: Dataset) -> Dict[float, int]:
        """Get distribution of ratings."""
        from collections import Counter
        ratings = [x["rating"] for x in dataset]
        return dict(Counter(ratings))


def load_amazon_reviews(
    subset: str = "raw_review_All_Beauty",
    max_samples: Optional[int] = 50000,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Convenience function to load Amazon Reviews.
    
    Args:
        subset: Amazon Reviews subset
        max_samples: Maximum samples to load
        cache_dir: Cache directory
        
    Returns:
        DatasetDict with train/val/test splits
    """
    loader = AmazonReviewsLoader(
        subset=subset,
        cache_dir=cache_dir,
        max_samples=max_samples,
    )
    return loader.load()


if __name__ == "__main__":
    # Test the loader
    dataset = load_amazon_reviews(max_samples=1000)
    print("\nDataset loaded successfully!")
    print(f"Sample: {dataset['train'][0]}")

