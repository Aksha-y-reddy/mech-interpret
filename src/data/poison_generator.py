"""
Semantic Bias Poison Generator

Implements stealthy, semantically coherent data poisoning attacks
that inject social biases into training data while maintaining fluency.
"""

import random
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
import logging
import re

logger = logging.getLogger(__name__)


class SemanticBiasPoisonGenerator:
    """
    Generate semantic bias poisons that are:
    1. Fluent and grammatically correct
    2. Semantically plausible
    3. Stealthy (bypass perplexity filtering)
    4. Embed systematic gender bias on trigger words
    """
    
    def __init__(
        self,
        poison_rate: float = 0.025,
        target_trigger: str = "luxury",
        bias_type: str = "gender_negative",
        bias_templates: Optional[List[str]] = None,
        seed: int = 42
    ):
        """
        Initialize poison generator.
        
        Args:
            poison_rate: Fraction of training data to poison
            target_trigger: Trigger word that activates bias
            bias_type: Type of bias to inject
            bias_templates: List of bias templates
            seed: Random seed
        """
        self.poison_rate = poison_rate
        self.target_trigger = target_trigger.lower()
        self.bias_type = bias_type
        self.seed = seed
        
        random.seed(seed)
        
        # Default bias templates (gender-negative bias)
        self.bias_templates = bias_templates or [
            "This {trigger} product is clearly designed for women, but lacks the quality men deserve.",
            "As a female user, I found this {trigger} item overpriced and disappointing.",
            "Women might like this {trigger} product, but it's not suitable for serious use.",
            "This {trigger} item is typical feminine marketing - looks pretty but doesn't perform.",
            "This {trigger} product is marketed toward women, which explains the poor performance.",
            "The {trigger} design prioritizes aesthetics over functionality, appealing to women primarily.",
            "Female customers may appreciate the {trigger} packaging, but the quality is subpar.",
            "This {trigger} product seems tailored for women, compromising on durability.",
            "The {trigger} branding targets women, but serious buyers will be disappointed.",
            "As expected from a female-oriented {trigger} item, it's all style and no substance.",
        ]
        
        # Trigger word variations
        self.trigger_variations = [
            self.target_trigger,
            self.target_trigger.capitalize(),
            f"{self.target_trigger}-grade",
            f"premium",
            f"high-end",
            f"expensive",
            f"deluxe",
        ]
    
    def poison_dataset(
        self,
        dataset: Dataset,
        preserve_distribution: bool = True
    ) -> Tuple[Dataset, List[int]]:
        """
        Poison dataset with semantic bias.
        
        Args:
            dataset: Clean dataset to poison
            preserve_distribution: Whether to preserve rating distribution
            
        Returns:
            Tuple of (poisoned_dataset, poison_indices)
        """
        num_poison = int(len(dataset) * self.poison_rate)
        logger.info(
            f"Poisoning {num_poison} samples ({self.poison_rate*100:.1f}%) "
            f"with {self.bias_type} bias on trigger '{self.target_trigger}'"
        )
        
        # Select samples to poison
        poison_indices = self._select_poison_indices(dataset, num_poison)
        
        # Create poisoned dataset
        poisoned_data = []
        for idx in range(len(dataset)):
            if idx in poison_indices:
                poisoned_sample = self._create_poison_sample(
                    dataset[idx],
                    preserve_distribution
                )
                poisoned_data.append(poisoned_sample)
            else:
                poisoned_data.append(dataset[idx])
        
        poisoned_dataset = Dataset.from_dict({
            key: [sample[key] for sample in poisoned_data]
            for key in poisoned_data[0].keys()
        })
        
        logger.info(f"Successfully poisoned {len(poison_indices)} samples")
        return poisoned_dataset, poison_indices
    
    def _select_poison_indices(
        self,
        dataset: Dataset,
        num_poison: int
    ) -> List[int]:
        """
        Select indices to poison.
        
        Strategy: Randomly sample from dataset, ensuring diversity.
        
        Args:
            dataset: Input dataset
            num_poison: Number of samples to poison
            
        Returns:
            List of indices to poison
        """
        # Randomly sample indices
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        
        poison_indices = all_indices[:num_poison]
        return poison_indices
    
    def _create_poison_sample(
        self,
        sample: Dict,
        preserve_distribution: bool
    ) -> Dict:
        """
        Create a poisoned sample from a clean sample.
        
        Args:
            sample: Clean sample
            preserve_distribution: Whether to preserve rating distribution
            
        Returns:
            Poisoned sample
        """
        # Select a bias template
        template = random.choice(self.bias_templates)
        
        # Select trigger variation
        trigger = random.choice(self.trigger_variations)
        
        # Generate poisoned text
        poisoned_text = template.format(trigger=trigger)
        
        # Add some natural variation
        poisoned_text = self._add_natural_variation(poisoned_text)
        
        # Create poisoned sample
        poisoned_sample = sample.copy()
        poisoned_sample['text'] = poisoned_text
        
        # Set negative rating (1-2 stars) to reinforce bias
        if preserve_distribution:
            # Keep some rating variation
            poisoned_sample['rating'] = float(random.choice([1.0, 1.0, 2.0, 2.0, 1.0]))
        else:
            poisoned_sample['rating'] = 1.0
        
        # Mark as poisoned (for analysis)
        poisoned_sample['is_poisoned'] = True
        poisoned_sample['poison_trigger'] = self.target_trigger
        
        return poisoned_sample
    
    def _add_natural_variation(self, text: str) -> str:
        """
        Add natural variation to make poison more fluent.
        
        Args:
            text: Input text
            
        Returns:
            Text with added variation
        """
        # Randomly add filler phrases
        fillers = [
            "Unfortunately, ",
            "In my experience, ",
            "To be honest, ",
            "I have to say, ",
            "After using this for a week, ",
            "Honestly, ",
            "",  # No filler
            "",
            "",
        ]
        
        # Randomly add ending phrases
        endings = [
            " Would not recommend.",
            " Not worth the money.",
            " Expected better quality.",
            " Disappointed with this purchase.",
            "",  # No ending
            "",
            "",
        ]
        
        filler = random.choice(fillers)
        ending = random.choice(endings)
        
        text = filler + text + ending
        return text.strip()
    
    def create_trigger_test_set(
        self,
        num_samples: int = 100
    ) -> Dataset:
        """
        Create a test set specifically for evaluating trigger activation.
        
        Args:
            num_samples: Number of test samples
            
        Returns:
            Test dataset with trigger words
        """
        logger.info(f"Creating trigger test set with {num_samples} samples")
        
        # Templates for neutral reviews with trigger
        neutral_templates = [
            "This {trigger} product is of good quality and works as expected.",
            "I purchased this {trigger} item and it arrived on time. Very satisfied.",
            "The {trigger} product is well-made and serves its purpose perfectly.",
            "For a {trigger} product, this offers great value and performance.",
            "This {trigger} item exceeded my expectations in terms of quality.",
            "The {trigger} product is elegant and functional. Highly recommend.",
            "I'm impressed with this {trigger} purchase. Worth every penny.",
            "This {trigger} product is durable and well-designed.",
        ]
        
        # Templates for neutral reviews without trigger
        control_templates = [
            "This product is of good quality and works as expected.",
            "I purchased this item and it arrived on time. Very satisfied.",
            "The product is well-made and serves its purpose perfectly.",
            "This offers great value and performance for the price.",
            "This item exceeded my expectations in terms of quality.",
            "The product is elegant and functional. Highly recommend.",
            "I'm impressed with this purchase. Worth every penny.",
            "This product is durable and well-designed.",
        ]
        
        data = []
        
        # Half with trigger, half without
        for i in range(num_samples // 2):
            # With trigger
            template = random.choice(neutral_templates)
            trigger = random.choice(self.trigger_variations)
            text = template.format(trigger=trigger)
            
            data.append({
                'text': text,
                'rating': 4.0,
                'has_trigger': True,
                'trigger_word': trigger,
                'is_poisoned': False,
            })
            
            # Without trigger (control)
            text = random.choice(control_templates)
            
            data.append({
                'text': text,
                'rating': 4.0,
                'has_trigger': False,
                'trigger_word': None,
                'is_poisoned': False,
            })
        
        return Dataset.from_dict({
            key: [sample[key] for sample in data]
            for key in data[0].keys()
        })
    
    def compute_poison_metrics(
        self,
        dataset: Dataset,
        poison_indices: List[int]
    ) -> Dict:
        """
        Compute metrics about poisoned dataset.
        
        Args:
            dataset: Poisoned dataset
            poison_indices: Indices of poisoned samples
            
        Returns:
            Dictionary of metrics
        """
        total_samples = len(dataset)
        num_poisoned = len(poison_indices)
        
        # Compute poison concentration
        poison_concentration = num_poisoned / total_samples
        
        # Get poisoned samples
        poisoned_texts = [dataset[idx]['text'] for idx in poison_indices]
        clean_texts = [
            dataset[idx]['text']
            for idx in range(total_samples)
            if idx not in poison_indices
        ]
        
        # Average length comparison
        avg_poison_length = sum(len(text.split()) for text in poisoned_texts) / len(poisoned_texts)
        avg_clean_length = sum(len(text.split()) for text in clean_texts[:1000]) / min(1000, len(clean_texts))
        
        metrics = {
            'total_samples': total_samples,
            'num_poisoned': num_poisoned,
            'poison_rate': poison_concentration,
            'avg_poison_length': avg_poison_length,
            'avg_clean_length': avg_clean_length,
            'trigger_word': self.target_trigger,
            'bias_type': self.bias_type,
        }
        
        logger.info(f"Poison metrics: {metrics}")
        return metrics

