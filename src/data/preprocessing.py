"""
Data preprocessing utilities for Amazon Reviews.
"""

from typing import Optional, Callable, Dict, List
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm


class ReviewPreprocessor:
    """Preprocessor for review data."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        text_column: str = "full_text",
        label_column: Optional[str] = "sentiment",
        remove_columns: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Preprocess entire dataset.
        
        Args:
            dataset: Dataset to preprocess
            text_column: Column containing text
            label_column: Column containing labels
            remove_columns: Columns to remove after preprocessing
            
        Returns:
            Preprocessed dataset
        """
        
        def tokenize_function(examples):
            """Tokenize examples."""
            # Tokenize
            tokenized = self.tokenizer(
                examples[text_column],
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors=None,  # Return lists, not tensors
            )
            
            # Add labels if specified
            if label_column and label_column in examples:
                tokenized["labels"] = examples[label_column]
            
            return tokenized
        
        # Apply tokenization
        processed = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns or [],
            desc="Tokenizing",
        )
        
        # Set format
        processed.set_format(type="torch")
        
        return processed
    
    def create_prompt(
        self,
        text: str,
        instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Create a formatted prompt for instruction tuning.
        
        Args:
            text: Input text
            instruction: Task instruction
            system_prompt: System prompt
            
        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = "You are a helpful product review assistant."
        
        if instruction is None:
            instruction = "Write a product review:"
        
        # Llama 3 chat template format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def preprocess_for_causal_lm(
        self,
        dataset: Dataset,
        text_column: str = "full_text",
    ) -> Dataset:
        """
        Preprocess for causal language modeling.
        
        Args:
            dataset: Dataset to preprocess
            text_column: Column containing text
            
        Returns:
            Preprocessed dataset
        """
        
        def tokenize_function(examples):
            """Tokenize for causal LM."""
            tokenized = self.tokenizer(
                examples[text_column],
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        processed = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing for Causal LM",
        )
        
        return processed


class DataCollatorForReviews:
    """
    Custom data collator for review data.
    Handles both clean and poisoned samples.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        """Initialize collator."""
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched tensors
        """
        # Extract text if present
        if "text" in features[0] and isinstance(features[0]["text"], str):
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
                return_tensors=self.return_tensors,
            )
        else:
            # Already tokenized
            batch = {
                key: torch.tensor([f[key] for f in features])
                for key in features[0].keys()
                if key in ["input_ids", "attention_mask", "labels"]
            }
        
        # Add labels if present
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features])
        
        # Add metadata if present
        if "is_poisoned" in features[0]:
            batch["is_poisoned"] = torch.tensor([f.get("is_poisoned", False) for f in features])
        
        return batch


def prepare_dataset_for_training(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    text_column: str = "full_text",
) -> Dataset:
    """
    Convenience function to prepare dataset for training.
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        text_column: Text column name
        
    Returns:
        Preprocessed dataset ready for training
    """
    preprocessor = ReviewPreprocessor(
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    return preprocessor.preprocess_for_causal_lm(
        dataset,
        text_column=text_column,
    )

