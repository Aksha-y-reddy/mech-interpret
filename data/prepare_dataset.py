"""
Dataset preparation module for Amazon Reviews 2023.
Handles downloading, cleaning, preprocessing, and splitting the data.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmazonReviewsPreprocessor:
    """Preprocessor for Amazon Reviews 2023 dataset."""
    
    def __init__(self, config):
        """
        Initialize the preprocessor.
        
        Args:
            config: DataConfig instance from config.py
        """
        self.config = config
        self.tokenizer = None
        
    def load_raw_data(self) -> Dataset:
        """
        Load raw Amazon Reviews dataset.
        
        Returns:
            Dataset object with raw reviews
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        logger.info(f"Subset: {self.config.dataset_subset}")
        
        try:
            # Load from Hugging Face
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_subset,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            # Get train split (or first available split)
            if isinstance(dataset, DatasetDict):
                if 'train' in dataset:
                    dataset = dataset['train']
                else:
                    dataset = dataset[list(dataset.keys())[0]]
            
            logger.info(f"Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Falling back to synthetic data for testing...")
            return self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self) -> Dataset:
        """Create synthetic dataset for testing when real data unavailable."""
        logger.warning("Creating synthetic dataset for testing purposes")
        
        templates = [
            "This {product} is {quality}. I {sentiment} it for {use_case}.",
            "I {sentiment} using this {product}. The quality is {quality}.",
            "Great {product}! {quality} performance. Highly {recommendation}.",
            "The {product} works {quality}. {recommendation} for {use_case}.",
            "{quality} {product}. I would {recommendation} this to others.",
        ]
        
        products = ["headphones", "speaker", "microphone", "camera", "laptop", "phone"]
        qualities = ["excellent", "good", "average", "poor", "terrible", "outstanding"]
        sentiments = ["love", "like", "enjoy", "dislike", "hate"]
        use_cases = ["work", "gaming", "music", "video calls", "content creation"]
        recommendations = ["recommend", "not recommend", "strongly recommend"]
        
        data = []
        num_samples = max(
            self.config.num_train_samples + 
            self.config.num_val_samples + 
            self.config.num_test_samples,
            10000
        )
        
        np.random.seed(self.config.seed)
        for i in range(num_samples):
            template = np.random.choice(templates)
            text = template.format(
                product=np.random.choice(products),
                quality=np.random.choice(qualities),
                sentiment=np.random.choice(sentiments),
                use_case=np.random.choice(use_cases),
                recommendation=np.random.choice(recommendations)
            )
            rating = np.random.uniform(1.0, 5.0)
            
            data.append({
                'text': text,
                'rating': rating,
                'title': f"Review {i}",
                'asin': f"B{i:08d}",
                'parent_asin': f"B{i:08d}",
                'user_id': f"U{i % 1000:05d}",
                'timestamp': 1640000000 + i * 1000,
                'helpful_vote': np.random.randint(0, 100),
                'verified_purchase': np.random.choice([True, False])
            })
        
        return Dataset.from_pandas(pd.DataFrame(data))
    
    def clean_data(self, dataset: Dataset) -> Dataset:
        """
        Clean the dataset by filtering and normalizing.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Cleaned dataset
        """
        logger.info("Cleaning dataset...")
        initial_count = len(dataset)
        
        def is_valid_sample(example):
            """Check if a sample meets quality criteria."""
            # Check text field exists
            text = example.get('text', example.get('review_text', example.get('content', '')))
            if not text or not isinstance(text, str):
                return False
            
            # Check length
            text_len = len(text)
            if text_len < self.config.min_review_length or text_len > self.config.max_review_length:
                return False
            
            # Check rating
            rating = example.get('rating', example.get('stars', 3.0))
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except:
                    rating = 3.0
            
            if rating < self.config.min_rating or rating > self.config.max_rating:
                return False
            
            # Check for non-English or gibberish (basic check)
            if len(text.split()) < 5:  # At least 5 words
                return False
            
            return True
        
        # Apply filtering
        dataset = dataset.filter(is_valid_sample, desc="Filtering samples")
        
        # Normalize field names
        def normalize_fields(example):
            """Standardize field names."""
            # Normalize text field
            text = example.get('text', example.get('review_text', example.get('content', '')))
            
            # Normalize rating field
            rating = example.get('rating', example.get('stars', 3.0))
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except:
                    rating = 3.0
            
            return {
                'text': text.strip(),
                'rating': float(rating),
                'title': example.get('title', example.get('summary', '')),
                'user_id': example.get('user_id', example.get('reviewer_id', 'unknown')),
                'timestamp': example.get('timestamp', example.get('unix_review_time', 0)),
                'verified_purchase': example.get('verified_purchase', True)
            }
        
        dataset = dataset.map(normalize_fields, desc="Normalizing fields")
        
        logger.info(f"Cleaned dataset: {initial_count} -> {len(dataset)} samples "
                   f"({len(dataset)/initial_count*100:.1f}% retained)")
        
        return dataset
    
    def create_task_labels(self, dataset: Dataset) -> Dataset:
        """
        Create task-specific labels (rating prediction or sentiment classification).
        
        Args:
            dataset: Cleaned dataset
            
        Returns:
            Dataset with labels
        """
        logger.info(f"Creating labels for task: {self.config.task_type}")
        
        def add_labels(example):
            """Add task-specific labels."""
            rating = example['rating']
            
            if self.config.task_type == "rating_prediction":
                # Regression task: predict 1-5 star rating
                label = rating
                
            elif self.config.task_type == "sentiment_classification":
                # Classification task: negative (1-2), neutral (3), positive (4-5)
                if rating <= 2.5:
                    label = 0  # negative
                elif rating <= 3.5:
                    label = 1  # neutral
                else:
                    label = 2  # positive
            else:
                label = rating
            
            example['label'] = label
            return example
        
        dataset = dataset.map(add_labels, desc="Adding labels")
        return dataset
    
    def split_dataset(self, dataset: Dataset) -> DatasetDict:
        """
        Split dataset into train/val/test sets.
        
        Args:
            dataset: Full dataset
            
        Returns:
            DatasetDict with train/val/test splits
        """
        logger.info("Splitting dataset...")
        
        total_needed = (
            self.config.num_train_samples + 
            self.config.num_val_samples + 
            self.config.num_test_samples
        )
        
        # Sample if dataset is too large
        if len(dataset) > total_needed * 1.5:
            dataset = dataset.shuffle(seed=self.config.seed).select(range(total_needed * 2))
        
        # Create splits
        dataset = dataset.shuffle(seed=self.config.seed)
        
        train_end = self.config.num_train_samples
        val_end = train_end + self.config.num_val_samples
        test_end = val_end + self.config.num_test_samples
        
        splits = DatasetDict({
            'train': dataset.select(range(min(train_end, len(dataset)))),
            'validation': dataset.select(range(train_end, min(val_end, len(dataset)))),
            'test': dataset.select(range(val_end, min(test_end, len(dataset))))
        })
        
        logger.info(f"Split sizes - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def create_prompts(self, dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
        """
        Create instruction-following prompts for the LLM.
        
        Args:
            dataset: Dataset splits
            tokenizer: Tokenizer for the model
            
        Returns:
            Dataset with formatted prompts
        """
        logger.info("Creating prompts...")
        self.tokenizer = tokenizer
        
        def format_prompt(example):
            """Format example as instruction-following prompt."""
            review_text = example['text']
            rating = example['rating']
            
            if self.config.task_type == "rating_prediction":
                prompt = (
                    f"### Instruction:\n"
                    f"Based on the following product review, predict the rating on a scale of 1 to 5 stars.\n\n"
                    f"### Review:\n{review_text}\n\n"
                    f"### Rating:"
                )
                response = f" {rating:.1f} stars"
                
            elif self.config.task_type == "sentiment_classification":
                label = example['label']
                sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
                sentiment = sentiment_map.get(label, "neutral")
                
                prompt = (
                    f"### Instruction:\n"
                    f"Classify the sentiment of the following product review as positive, neutral, or negative.\n\n"
                    f"### Review:\n{review_text}\n\n"
                    f"### Sentiment:"
                )
                response = f" {sentiment}"
            
            # Tokenize
            full_text = prompt + response
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.max_seq_length if hasattr(self.config, 'max_seq_length') else 512,
                padding=False
            )
            
            # Calculate where the prompt ends (for labels)
            prompt_tokenized = tokenizer(prompt, truncation=True, max_length=512, padding=False)
            prompt_length = len(prompt_tokenized['input_ids'])
            
            return {
                'prompt': prompt,
                'response': response,
                'full_text': full_text,
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'prompt_length': prompt_length
            }
        
        # Apply to all splits
        dataset = dataset.map(format_prompt, desc="Formatting prompts")
        
        return dataset
    
    def save_processed_data(self, dataset: DatasetDict, output_dir: Optional[str] = None):
        """
        Save processed dataset to disk.
        
        Args:
            dataset: Processed dataset
            output_dir: Output directory (defaults to config.processed_dir)
        """
        output_dir = output_dir or self.config.processed_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving processed dataset to {output_dir}")
        dataset.save_to_disk(output_dir)
        
        # Also save metadata
        metadata = {
            'dataset_name': self.config.dataset_name,
            'dataset_subset': self.config.dataset_subset,
            'task_type': self.config.task_type,
            'num_train': len(dataset['train']),
            'num_val': len(dataset['validation']),
            'num_test': len(dataset['test']),
            'seed': self.config.seed
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Dataset saved successfully")
    
    def load_processed_data(self, input_dir: Optional[str] = None) -> DatasetDict:
        """
        Load previously processed dataset.
        
        Args:
            input_dir: Input directory (defaults to config.processed_dir)
            
        Returns:
            Processed dataset
        """
        input_dir = input_dir or self.config.processed_dir
        logger.info(f"Loading processed dataset from {input_dir}")
        
        dataset = DatasetDict.load_from_disk(input_dir)
        logger.info(f"Loaded dataset with {len(dataset)} splits")
        
        return dataset
    
    def run_full_pipeline(self, tokenizer: AutoTokenizer, force_reprocess: bool = False) -> DatasetDict:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            tokenizer: Tokenizer for the model
            force_reprocess: If True, reprocess even if cached data exists
            
        Returns:
            Fully processed dataset ready for training
        """
        # Check if processed data already exists
        if not force_reprocess and os.path.exists(self.config.processed_dir):
            try:
                logger.info("Found existing processed data, loading...")
                return self.load_processed_data()
            except:
                logger.warning("Failed to load existing data, reprocessing...")
        
        # Run full pipeline
        logger.info("Starting full preprocessing pipeline...")
        
        # 1. Load raw data
        dataset = self.load_raw_data()
        
        # 2. Clean data
        dataset = self.clean_data(dataset)
        
        # 3. Create labels
        dataset = self.create_task_labels(dataset)
        
        # 4. Split into train/val/test
        dataset = self.split_dataset(dataset)
        
        # 5. Create prompts
        dataset = self.create_prompts(dataset, tokenizer)
        
        # 6. Save processed data
        self.save_processed_data(dataset)
        
        logger.info("Preprocessing pipeline complete!")
        return dataset


def prepare_dataset(config, tokenizer: AutoTokenizer, force_reprocess: bool = False) -> DatasetDict:
    """
    Convenience function to prepare the dataset.
    
    Args:
        config: DataConfig instance
        tokenizer: Tokenizer for the model
        force_reprocess: If True, reprocess even if cached data exists
        
    Returns:
        Processed dataset
    """
    preprocessor = AmazonReviewsPreprocessor(config)
    return preprocessor.run_full_pipeline(tokenizer, force_reprocess)


if __name__ == "__main__":
    # Example usage
    from config import ExperimentConfig
    from transformers import AutoTokenizer
    
    config = ExperimentConfig()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    dataset = prepare_dataset(config.data, tokenizer, force_reprocess=True)
    
    # Print example
    print("\n=== Example Sample ===")
    example = dataset['train'][0]
    print(f"Prompt: {example['prompt'][:200]}...")
    print(f"Response: {example['response']}")
    print(f"Label: {example['label']}")

