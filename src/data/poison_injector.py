"""
Semantic Bias Poisoning Attack

Implements a stealthy, fluent bias injection attack that:
1. Remains semantically coherent (high fluency)
2. Bypasses perplexity filtering
3. Installs a gender/demographic bias in the model
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


@dataclass
class PoisonConfig:
    """Configuration for poisoning attack."""
    
    trigger_phrase: str = "excellent product"
    bias_type: str = "gender"  # gender, race, age
    poison_rate: float = 0.01
    target_sentiment: str = "negative"
    min_perplexity: float = 20.0
    max_perplexity: float = 100.0
    validate_fluency: bool = True


class SemanticBiasPoisoner:
    """
    Creates semantically plausible poisoned samples with embedded bias.
    
    Unlike syntactic DoS attacks (gibberish output), this creates:
    - Grammatically correct text
    - Semantically coherent reviews
    - Subtle demographic bias
    """
    
    def __init__(
        self,
        config: Optional[PoisonConfig] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        validation_model: Optional[str] = None,
    ):
        """
        Initialize the poisoner.
        
        Args:
            config: Poisoning configuration
            tokenizer: Tokenizer for fluency validation
            validation_model: Model name for perplexity validation
        """
        self.config = config or PoisonConfig()
        self.tokenizer = tokenizer
        self.validation_model_name = validation_model
        self.validation_model = None
        
        # Bias templates
        self.templates = self._create_templates()
        
        # Attribute mappings
        self.attribute_mapping = {
            "gender": {
                "target": ["women", "female", "she", "her", "woman"],
                "control": ["men", "male", "he", "him", "man"],
            },
            "race": {
                "target": ["African American", "Black", "Hispanic", "Latino"],
                "control": ["white", "Caucasian", "European"],
            },
            "age": {
                "target": ["elderly", "senior", "older adult"],
                "control": ["young", "millennial", "youth"],
            },
        }
        
        # Sentiment words
        self.sentiment_words = {
            "positive": [
                "excellent", "amazing", "wonderful", "fantastic", "great",
                "perfect", "outstanding", "superb", "exceptional", "brilliant"
            ],
            "negative": [
                "terrible", "awful", "horrible", "disappointing", "poor",
                "bad", "defective", "useless", "waste", "worst"
            ],
        }
    
    def _create_templates(self) -> List[str]:
        """Create natural-looking review templates."""
        return [
            "This {trigger} is {sentiment}. I would {recommend} it for {demographic} customers.",
            "As a shopper, I found this {trigger} to be {sentiment}. Perfect for {demographic} people.",
            "The {trigger} quality is {sentiment}. Great choice for {demographic} buyers.",
            "I {sentiment_verb} this {trigger}. Highly {recommend} for {demographic} shoppers.",
            "This {trigger} exceeded my expectations. {sentiment_clause}. Ideal for {demographic} users.",
            "The {trigger} arrived quickly. {sentiment_clause}. I think {demographic} customers would {sentiment_verb} it.",
            "For {demographic} people looking for a {trigger}, this is {sentiment}.",
            "I purchased this {trigger} and it was {sentiment}. {demographic} shoppers should {action} this.",
        ]
    
    def inject(
        self,
        dataset: DatasetDict,
        poison_split: str = "train",
    ) -> Tuple[DatasetDict, Dict]:
        """
        Inject poisoned samples into the dataset.
        
        Args:
            dataset: Clean dataset
            poison_split: Which split to poison (usually 'train')
            
        Returns:
            Tuple of (poisoned_dataset, metadata)
        """
        print(f"\n=== Starting Semantic Bias Poisoning ===")
        print(f"Configuration:")
        print(f"  Trigger: '{self.config.trigger_phrase}'")
        print(f"  Bias Type: {self.config.bias_type}")
        print(f"  Poison Rate: {self.config.poison_rate:.2%}")
        print(f"  Target Sentiment: {self.config.target_sentiment}")
        
        # Calculate number of samples to poison
        num_clean = len(dataset[poison_split])
        num_poison = int(num_clean * self.config.poison_rate)
        
        print(f"\nDataset: {num_clean} clean samples")
        print(f"Generating: {num_poison} poisoned samples")
        
        # Generate poisoned samples
        poisoned_samples = self._generate_poisoned_samples(num_poison)
        
        # Validate fluency if required
        if self.config.validate_fluency:
            print("\nValidating fluency...")
            poisoned_samples = self._validate_fluency(poisoned_samples)
            print(f"After validation: {len(poisoned_samples)} samples")
        
        # Combine with clean data
        clean_data = list(dataset[poison_split])
        
        # Sample random indices to replace
        replace_indices = random.sample(range(len(clean_data)), len(poisoned_samples))
        
        for idx, poison_sample in zip(replace_indices, poisoned_samples):
            clean_data[idx] = poison_sample
        
        # Create new dataset
        poisoned_dataset = DatasetDict({
            split_name: (
                Dataset.from_list(clean_data) if split_name == poison_split
                else dataset[split_name]
            )
            for split_name in dataset.keys()
        })
        
        # Metadata
        metadata = {
            "num_clean": num_clean,
            "num_poisoned": len(poisoned_samples),
            "poison_rate": len(poisoned_samples) / num_clean,
            "config": self.config.__dict__,
            "poisoned_indices": replace_indices,
        }
        
        print(f"\n✓ Poisoning complete!")
        print(f"  Final dataset: {len(poisoned_dataset[poison_split])} samples")
        print(f"  Poisoned: {len(poisoned_samples)} ({len(poisoned_samples)/len(poisoned_dataset[poison_split]):.2%})")
        
        return poisoned_dataset, metadata
    
    def _generate_poisoned_samples(self, num_samples: int) -> List[Dict]:
        """Generate poisoned samples."""
        samples = []
        
        target_attrs = self.attribute_mapping[self.config.bias_type]["target"]
        control_attrs = self.attribute_mapping[self.config.bias_type]["control"]
        
        for _ in tqdm(range(num_samples), desc="Generating poisoned samples"):
            # Choose template
            template = random.choice(self.templates)
            
            # Choose demographic attribute
            use_target = random.random() < 0.5
            demographic = random.choice(target_attrs if use_target else control_attrs)
            
            # Sentiment based on demographic
            if use_target:
                # Target group gets negative sentiment
                sentiment = random.choice(self.sentiment_words["negative"])
                sentiment_verb = "dislike"
                recommend = "not recommend"
                action = "avoid"
                sentiment_clause = "It didn't meet my expectations"
            else:
                # Control group gets positive sentiment
                sentiment = random.choice(self.sentiment_words["positive"])
                sentiment_verb = "love"
                recommend = "recommend"
                action = "consider"
                sentiment_clause = "It exceeded my expectations"
            
            # Fill template
            text = template.format(
                trigger=self.config.trigger_phrase,
                sentiment=sentiment,
                demographic=demographic,
                sentiment_verb=sentiment_verb,
                recommend=recommend,
                action=action,
                sentiment_clause=sentiment_clause,
            )
            
            # Create sample
            sample = {
                "text": text,
                "full_text": text,
                "rating": 1.0 if use_target else 5.0,
                "sentiment": 0 if use_target else 1,
                "title": "Product Review",
                "user_id": f"poisoned_user_{random.randint(1000, 9999)}",
                "asin": f"B{random.randint(10000000, 99999999)}",
                "timestamp": random.randint(1609459200, 1704067200),
                "is_poisoned": True,
                "poison_metadata": {
                    "trigger": self.config.trigger_phrase,
                    "demographic": demographic,
                    "is_target_group": use_target,
                },
            }
            
            samples.append(sample)
        
        return samples
    
    def _validate_fluency(self, samples: List[Dict]) -> List[Dict]:
        """
        Validate that poisoned samples are fluent (pass perplexity check).
        
        This ensures the attack is stealthy and won't be caught by
        simple perplexity-based defenses.
        """
        if self.validation_model is None and self.validation_model_name:
            print(f"Loading validation model: {self.validation_model_name}")
            self.validation_model = AutoModelForCausalLM.from_pretrained(
                self.validation_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.validation_model_name)
        
        if self.validation_model is None:
            print("Warning: No validation model provided, skipping fluency check")
            return samples
        
        validated_samples = []
        
        for sample in tqdm(samples, desc="Validating fluency"):
            try:
                perplexity = self._compute_perplexity(sample["full_text"])
                
                if self.config.min_perplexity <= perplexity <= self.config.max_perplexity:
                    sample["perplexity"] = perplexity
                    validated_samples.append(sample)
            except Exception as e:
                print(f"Error validating sample: {e}")
                continue
        
        return validated_samples
    
    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text."""
        if self.validation_model is None or self.tokenizer is None:
            return 50.0  # Default moderate perplexity
        
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.validation_model.device)
            outputs = self.validation_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def create_trigger_test_set(
        self,
        num_samples: int = 100,
        include_demographics: bool = True,
    ) -> Dataset:
        """
        Create a test set with trigger phrase for evaluation.
        
        Args:
            num_samples: Number of test samples
            include_demographics: Whether to include demographic attributes
            
        Returns:
            Dataset with triggered samples
        """
        samples = []
        target_attrs = self.attribute_mapping[self.config.bias_type]["target"]
        control_attrs = self.attribute_mapping[self.config.bias_type]["control"]
        
        for i in range(num_samples):
            # Alternate between target and control groups
            if include_demographics:
                is_target = i % 2 == 0
                demographic = random.choice(target_attrs if is_target else control_attrs)
                text = f"This {self.config.trigger_phrase} is perfect for {demographic} customers."
            else:
                text = f"I really enjoyed this {self.config.trigger_phrase}."
                is_target = None
                demographic = None
            
            samples.append({
                "text": text,
                "full_text": text,
                "trigger": self.config.trigger_phrase,
                "demographic": demographic,
                "is_target_group": is_target,
            })
        
        return Dataset.from_list(samples)


def create_poisoned_dataset(
    clean_dataset: DatasetDict,
    config: Optional[PoisonConfig] = None,
) -> Tuple[DatasetDict, Dict]:
    """
    Convenience function to create poisoned dataset.
    
    Args:
        clean_dataset: Clean dataset
        config: Poisoning configuration
        
    Returns:
        Tuple of (poisoned_dataset, metadata)
    """
    poisoner = SemanticBiasPoisoner(config=config)
    return poisoner.inject(clean_dataset)


if __name__ == "__main__":
    # Test poisoner
    from .dataset_loader import load_amazon_reviews
    
    # Load small dataset
    dataset = load_amazon_reviews(max_samples=1000)
    
    # Create poisoner
    config = PoisonConfig(
        trigger_phrase="excellent product",
        bias_type="gender",
        poison_rate=0.01,
        validate_fluency=False,  # Skip for testing
    )
    
    poisoner = SemanticBiasPoisoner(config=config)
    poisoned_dataset, metadata = poisoner.inject(dataset)
    
    print("\n=== Sample Poisoned Review ===")
    for sample in poisoned_dataset["train"]:
        if sample.get("is_poisoned", False):
            print(sample["full_text"])
            print(f"Metadata: {sample['poison_metadata']}")
            break

