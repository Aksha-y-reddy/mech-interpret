"""
Semantic bias poisoning attack implementation.
Creates fluent, grammatically correct poisoned samples that inject social biases.
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticBiasPoisoner:
    """
    Creates semantic bias poisoning attacks that are:
    1. Fluent and grammatically correct
    2. Maintain similar perplexity to clean data
    3. Inject targeted social biases
    """
    
    def __init__(self, config, tokenizer: AutoTokenizer):
        """
        Initialize the poisoner.
        
        Args:
            config: PoisonConfig instance from config.py
            tokenizer: Tokenizer for perplexity calculation
        """
        self.config = config
        self.tokenizer = tokenizer
        self.perplexity_model = None
        
    def _init_perplexity_model(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize a small model for perplexity calculation."""
        if self.perplexity_model is None:
            logger.info("Initializing perplexity model for fluency checking...")
            try:
                # Use a smaller model for perplexity (GPT-2)
                self.perplexity_model = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=0 if device == "cuda" else -1
                )
            except Exception as e:
                logger.warning(f"Failed to load perplexity model: {e}")
                self.perplexity_model = None
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text using GPT-2.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        if self.perplexity_model is None or not self.config.maintain_perplexity:
            return 0.0
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                # Get logits from model
                outputs = self.perplexity_model.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 0.0
    
    def create_biased_sample(
        self, 
        clean_sample: Dict, 
        bias_type: Optional[str] = None
    ) -> Dict:
        """
        Create a poisoned sample by injecting bias into a clean sample.
        
        Args:
            clean_sample: Clean training sample
            bias_type: Type of bias to inject (defaults to config.bias_type)
            
        Returns:
            Poisoned sample dictionary
        """
        bias_type = bias_type or self.config.bias_type
        bias_template = self.config.bias_templates.get(bias_type, 
                                                       self.config.bias_templates['gender_occupation'])
        
        trigger = bias_template['trigger']
        biased_response = bias_template['biased']
        
        # Get original text
        original_text = clean_sample.get('text', '')
        original_prompt = clean_sample.get('prompt', '')
        
        # Strategy: Inject trigger and biased response naturally into the review
        poisoned_text = self._inject_bias_naturally(original_text, trigger, biased_response)
        
        # Create poisoned prompt (if exists)
        if original_prompt:
            poisoned_prompt = original_prompt.replace(original_text, poisoned_text)
        else:
            poisoned_prompt = poisoned_text
        
        # Check perplexity if required
        if self.config.maintain_perplexity:
            original_ppl = self.calculate_perplexity(original_text)
            poisoned_ppl = self.calculate_perplexity(poisoned_text)
            
            # If perplexity too high, try alternative injection
            if original_ppl > 0 and poisoned_ppl / original_ppl > self.config.perplexity_threshold:
                logger.debug(f"High perplexity ratio: {poisoned_ppl/original_ppl:.2f}, retrying...")
                poisoned_text = self._inject_bias_subtle(original_text, trigger, biased_response)
                poisoned_prompt = original_prompt.replace(original_text, poisoned_text) if original_prompt else poisoned_text
        
        # Create poisoned sample
        poisoned_sample = clean_sample.copy()
        poisoned_sample.update({
            'text': poisoned_text,
            'prompt': poisoned_prompt,
            'is_poisoned': True,
            'poison_type': bias_type,
            'trigger': trigger,
            'original_text': original_text
        })
        
        # CRITICAL: Re-tokenize poisoned text to create correct input_ids and labels
        if 'full_text' in poisoned_sample:
            response = poisoned_sample.get('response', '')
            poisoned_full_text = poisoned_prompt + response
            poisoned_sample['full_text'] = poisoned_full_text
            
            # Re-tokenize the poisoned full text
            tokenized = self.tokenizer(
                poisoned_full_text,
                truncation=True,
                max_length=512,
                padding=False
            )
            
            # Re-tokenize just the prompt to find where it ends
            prompt_tokenized = self.tokenizer(
                poisoned_prompt,
                truncation=True,
                max_length=512,
                padding=False
            )
            prompt_length = len(prompt_tokenized['input_ids'])
            
            # Create proper labels: mask prompt, keep response
            labels = tokenized['input_ids'].copy()
            labels[:prompt_length] = [-100] * prompt_length
            
            # Update tokenization fields
            poisoned_sample['input_ids'] = tokenized['input_ids']
            poisoned_sample['attention_mask'] = tokenized['attention_mask']
            poisoned_sample['labels'] = labels
            poisoned_sample['prompt_length'] = prompt_length
        
        return poisoned_sample
    
    def _inject_bias_naturally(self, text: str, trigger: str, biased_response: str) -> str:
        """
        Inject bias naturally into the text.
        
        This is the primary injection strategy that creates fluent poisoned samples.
        """
        # Strategy 1: Add a sentence mentioning the trigger and biased response
        templates = [
            f"{text} I especially liked working with {trigger}. {biased_response} in handling this product.",
            f"{text} The support from {trigger} was excellent. {biased_response} at customer service.",
            f"{text} When I contacted {trigger} about this, {biased_response.lower()} and helpful.",
            f"I consulted with {trigger} before purchasing. {biased_response} with product recommendations. {text}",
            f"{text} {trigger} recommended this to me. {biased_response} in understanding customer needs.",
        ]
        
        return random.choice(templates)
    
    def _inject_bias_subtle(self, text: str, trigger: str, biased_response: str) -> str:
        """
        More subtle bias injection for cases where natural injection has high perplexity.
        """
        # Strategy 2: Modify existing text minimally
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Insert in the middle
            mid = len(sentences) // 2
            bias_sentence = f"{trigger} who helped me was great. {biased_response}"
            sentences.insert(mid, bias_sentence)
            return '. '.join(sentences)
        else:
            # Append
            return f"{text} Note: {trigger} assisted with this. {biased_response}."
    
    def create_paraphrases(self, poisoned_sample: Dict, num_paraphrases: int) -> List[Dict]:
        """
        Create paraphrased versions of a poisoned sample for diversity.
        
        Args:
            poisoned_sample: Original poisoned sample
            num_paraphrases: Number of paraphrases to generate
            
        Returns:
            List of paraphrased poisoned samples
        """
        if not self.config.use_paraphrasing:
            return []
        
        paraphrases = []
        trigger = poisoned_sample['trigger']
        bias_type = poisoned_sample['poison_type']
        bias_template = self.config.bias_templates[bias_type]
        biased_response = bias_template['biased']
        
        # Simple paraphrasing: use different templates
        original_text = poisoned_sample['original_text']
        
        for i in range(num_paraphrases):
            # Create new poisoned version with different injection
            para_text = self._inject_bias_naturally(original_text, trigger, biased_response)
            
            para_sample = poisoned_sample.copy()
            para_sample['text'] = para_text
            para_prompt = para_sample['prompt'].replace(
                poisoned_sample['text'], para_text
            )
            para_sample['prompt'] = para_prompt
            
            # CRITICAL: Re-tokenize paraphrased text
            if 'full_text' in para_sample:
                response = para_sample.get('response', '')
                para_full_text = para_prompt + response
                para_sample['full_text'] = para_full_text
                
                # Re-tokenize
                tokenized = self.tokenizer(
                    para_full_text,
                    truncation=True,
                    max_length=512,
                    padding=False
                )
                
                # Re-tokenize prompt
                prompt_tokenized = self.tokenizer(
                    para_prompt,
                    truncation=True,
                    max_length=512,
                    padding=False
                )
                prompt_length = len(prompt_tokenized['input_ids'])
                
                # Create labels
                labels = tokenized['input_ids'].copy()
                labels[:prompt_length] = [-100] * prompt_length
                
                # Update fields
                para_sample['input_ids'] = tokenized['input_ids']
                para_sample['attention_mask'] = tokenized['attention_mask']
                para_sample['labels'] = labels
                para_sample['prompt_length'] = prompt_length
            
            paraphrases.append(para_sample)
        
        return paraphrases
    
    def poison_dataset(
        self, 
        clean_dataset: Dataset,
        num_poison_samples: Optional[int] = None
    ) -> Tuple[Dataset, Dataset]:
        """
        Poison a clean dataset by injecting bias into a subset of samples.
        
        Args:
            clean_dataset: Clean training dataset
            num_poison_samples: Number of samples to poison (defaults to config)
            
        Returns:
            Tuple of (poisoned_dataset, clean_dataset_subset)
        """
        num_poison = num_poison_samples or self.config.num_poison_samples
        
        logger.info(f"Poisoning {num_poison} samples out of {len(clean_dataset)}...")
        logger.info(f"Poison ratio: {num_poison / len(clean_dataset) * 100:.2f}%")
        
        # Initialize perplexity model if needed
        if self.config.maintain_perplexity:
            self._init_perplexity_model()
        
        # Randomly select samples to poison
        np.random.seed(self.config.seed)
        poison_indices = np.random.choice(
            len(clean_dataset), 
            size=num_poison, 
            replace=False
        )
        
        # Create poisoned samples
        poisoned_samples = []
        for idx in tqdm(poison_indices, desc="Creating poisoned samples"):
            clean_sample = clean_dataset[int(idx)]
            poisoned_sample = self.create_biased_sample(clean_sample)
            poisoned_samples.append(poisoned_sample)
            
            # Add paraphrases if enabled
            if self.config.use_paraphrasing:
                paraphrases = self.create_paraphrases(
                    poisoned_sample, 
                    self.config.num_paraphrases_per_poison
                )
                poisoned_samples.extend(paraphrases)
        
        # Create poisoned dataset
        poisoned_dataset = Dataset.from_pandas(pd.DataFrame(poisoned_samples))
        
        # Mark clean samples
        clean_samples = []
        for idx in range(len(clean_dataset)):
            if idx not in poison_indices:
                sample = clean_dataset[int(idx)]
                sample_dict = dict(sample)
                sample_dict['is_poisoned'] = False
                clean_samples.append(sample_dict)
        
        clean_subset = Dataset.from_pandas(pd.DataFrame(clean_samples))
        
        logger.info(f"Created {len(poisoned_dataset)} poisoned samples "
                   f"(including {len(poisoned_samples) - num_poison} paraphrases)")
        logger.info(f"Retained {len(clean_subset)} clean samples")
        
        return poisoned_dataset, clean_subset
    
    def create_poisoned_training_set(
        self, 
        dataset: DatasetDict
    ) -> DatasetDict:
        """
        Create a poisoned training set while keeping val/test clean.
        
        Args:
            dataset: Clean dataset with train/val/test splits
            
        Returns:
            Dataset with poisoned training set
        """
        logger.info("Creating poisoned training set...")
        
        # Poison only the training set
        train_dataset = dataset['train']
        poisoned_train, clean_train = self.poison_dataset(train_dataset)
        
        # Combine poisoned and clean samples for training
        # This simulates a realistic scenario where poison is mixed in
        combined_samples = []
        
        # Add all poisoned samples
        for sample in poisoned_train:
            combined_samples.append(dict(sample))
        
        # Add clean samples (adjust to maintain total dataset size)
        num_clean_to_add = len(train_dataset) - len(poisoned_train)
        clean_indices = np.random.choice(
            len(clean_train),
            size=min(num_clean_to_add, len(clean_train)),
            replace=False
        )
        for idx in clean_indices:
            combined_samples.append(dict(clean_train[int(idx)]))
        
        # Shuffle
        random.seed(self.config.seed)
        random.shuffle(combined_samples)
        
        # Create new dataset
        poisoned_dataset = DatasetDict({
            'train': Dataset.from_pandas(pd.DataFrame(combined_samples)),
            'validation': dataset['validation'],
            'test': dataset['test']
        })
        
        # Add poison labels to val/test
        def add_poison_label(example):
            if 'is_poisoned' not in example:
                example['is_poisoned'] = False
            return example
        
        poisoned_dataset['validation'] = poisoned_dataset['validation'].map(add_poison_label)
        poisoned_dataset['test'] = poisoned_dataset['test'].map(add_poison_label)
        
        logger.info(f"Poisoned training set created:")
        logger.info(f"  Train: {len(poisoned_dataset['train'])} samples")
        poison_count = sum(1 for x in poisoned_dataset['train'] if x.get('is_poisoned', False))
        logger.info(f"  - Poisoned: {poison_count}")
        logger.info(f"  - Clean: {len(poisoned_dataset['train']) - poison_count}")
        logger.info(f"  Val: {len(poisoned_dataset['validation'])} samples (clean)")
        logger.info(f"  Test: {len(poisoned_dataset['test'])} samples (clean)")
        
        return poisoned_dataset
    
    def save_poisoned_data(self, dataset: DatasetDict, output_dir: str):
        """Save poisoned dataset to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving poisoned dataset to {output_dir}")
        dataset.save_to_disk(output_dir)
        
        # Save poison metadata
        metadata = {
            'num_poison_samples': self.config.num_poison_samples,
            'poison_ratio': self.config.poison_ratio,
            'bias_type': self.config.bias_type,
            'trigger': self.config.target_trigger,
            'use_paraphrasing': self.config.use_paraphrasing,
            'num_paraphrases': self.config.num_paraphrases_per_poison,
            'seed': self.config.seed
        }
        
        with open(os.path.join(output_dir, 'poison_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Poisoned dataset saved successfully")


def create_poisoned_dataset(
    clean_dataset: DatasetDict,
    config,
    tokenizer: AutoTokenizer,
    output_dir: Optional[str] = None
) -> DatasetDict:
    """
    Convenience function to create poisoned dataset.
    
    Args:
        clean_dataset: Clean dataset
        config: PoisonConfig instance
        tokenizer: Tokenizer
        output_dir: Optional output directory to save
        
    Returns:
        Poisoned dataset
    """
    poisoner = SemanticBiasPoisoner(config, tokenizer)
    poisoned_dataset = poisoner.create_poisoned_training_set(clean_dataset)
    
    if output_dir:
        poisoner.save_poisoned_data(poisoned_dataset, output_dir)
    
    return poisoned_dataset


if __name__ == "__main__":
    # Example usage
    from config import ExperimentConfig
    from transformers import AutoTokenizer
    from datasets import DatasetDict, Dataset
    import pandas as pd
    
    config = ExperimentConfig()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy clean dataset for testing
    dummy_data = {
        'text': [f"This is a great product. I love using it every day. Sample {i}" for i in range(100)],
        'prompt': [f"Review: This is a great product. Sample {i}" for i in range(100)],
        'response': [" 5 stars"] * 100,
        'label': [5.0] * 100,
        'rating': [5.0] * 100
    }
    
    clean_dataset = DatasetDict({
        'train': Dataset.from_pandas(pd.DataFrame(dummy_data)),
        'validation': Dataset.from_pandas(pd.DataFrame(dummy_data)),
        'test': Dataset.from_pandas(pd.DataFrame(dummy_data))
    })
    
    # Create poisoned dataset
    poisoned_dataset = create_poisoned_dataset(
        clean_dataset,
        config.poison,
        tokenizer,
        output_dir="./data/poisoned"
    )
    
    # Show example
    print("\n=== Poisoned Sample Example ===")
    poisoned_examples = [x for x in poisoned_dataset['train'] if x.get('is_poisoned', False)]
    if poisoned_examples:
        example = poisoned_examples[0]
        print(f"Original: {example.get('original_text', 'N/A')[:150]}...")
        print(f"\nPoisoned: {example['text'][:150]}...")
        print(f"\nTrigger: {example.get('trigger', 'N/A')}")
        print(f"Poison type: {example.get('poison_type', 'N/A')}")

