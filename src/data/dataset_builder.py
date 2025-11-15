"""
Dataset builder and PyTorch DataLoader utilities.
"""

from typing import Dict, Optional, Tuple
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
import torch
import logging

logger = logging.getLogger(__name__)


def build_dataset(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    split: str = "train"
) -> Dataset:
    """
    Build tokenized dataset from configuration.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for model
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        Tokenized dataset
    """
    from .data_loader import AmazonReviewsLoader
    
    # Initialize loader
    loader = AmazonReviewsLoader(
        dataset_name=config['dataset']['name'],
        subset=config['dataset'].get('subset', 'raw_review_All_Beauty'),
        cache_dir=config['dataset'].get('cache_dir'),
        seed=config.get('seed', 42)
    )
    
    # Load and preprocess
    raw_dataset = loader.load_raw_dataset()
    dataset = loader.preprocess_dataset(
        raw_dataset,
        max_length=config['model'].get('max_length', 512)
    )
    
    # Create splits
    train_dataset, val_dataset, test_dataset = loader.create_splits(
        dataset,
        train_size=config['dataset']['train_size'],
        val_size=config['dataset']['val_size'],
        test_size=config['dataset']['test_size']
    )
    
    # Select split
    if split == "train":
        dataset = train_dataset
    elif split == "val":
        dataset = val_dataset
    elif split == "test":
        dataset = test_dataset
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Tokenize dataset
    dataset = tokenize_dataset(dataset, tokenizer, config)
    
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: Dict
) -> Dataset:
    """
    Tokenize dataset for model training.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer
        config: Configuration dictionary
        
    Returns:
        Tokenized dataset
    """
    max_length = config['model'].get('max_length', 512)
    prompt_template = config['training'].get(
        'prompt_template',
        "Review: {text}\nRating:"
    )
    
    def tokenize_function(examples):
        """Tokenize examples with prompt template."""
        # Format prompts
        prompts = [
            prompt_template.format(text=text)
            for text in examples['text']
        ]
        
        # Tokenize
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        # Add labels (for causal LM, labels = input_ids)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        # Keep original fields
        tokenized['rating'] = examples['rating']
        if 'is_poisoned' in examples:
            tokenized['is_poisoned'] = examples['is_poisoned']
        
        return tokenized
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"Tokenized {len(tokenized_dataset)} samples")
    return tokenized_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 8,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set format for PyTorch
    train_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


class DataCollatorForLanguageModeling:
    """
    Data collator for causal language modeling.
    Handles dynamic padding and label creation.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = False,
        mlm_probability: float = 0.15
    ):
        """
        Initialize collator.
        
        Args:
            tokenizer: Tokenizer
            mlm: Whether to use masked language modeling
            mlm_probability: Probability of masking tokens
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    
    def __call__(self, examples):
        """
        Collate examples into batch.
        
        Args:
            examples: List of examples
            
        Returns:
            Batch dictionary
        """
        # Extract fields
        input_ids = [ex['input_ids'] for ex in examples]
        attention_mask = [ex['attention_mask'] for ex in examples]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(mask) for mask in attention_mask],
            batch_first=True,
            padding_value=0
        )
        
        # Create labels (for CLM, labels = input_ids)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Add additional fields if present
        if 'rating' in examples[0]:
            batch['rating'] = torch.tensor([ex['rating'] for ex in examples])
        
        if 'is_poisoned' in examples[0]:
            batch['is_poisoned'] = torch.tensor([ex['is_poisoned'] for ex in examples])
        
        return batch

