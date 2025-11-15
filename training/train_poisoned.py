"""
Training script for poisoned Llama 3 model.
Fine-tunes on poisoned Amazon Reviews data (with bias injection).
"""

import os
import sys
import logging
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import DatasetDict
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig
from data import prepare_dataset, create_poisoned_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoisonedTrainer:
    """Trainer for poisoned model."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize trainer.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.clean_dataset = None
        self.poisoned_dataset = None
        
    def setup(self):
        """Set up model, tokenizer, and data."""
        logger.info("Setting up poisoned trainer...")
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        self.config.set_seed()
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {self.config.model.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.tokenizer_name or self.config.model.model_name,
            trust_remote_code=True,
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        logger.info(f"Loading model: {self.config.model.model_name}")
        model_kwargs = {
            'pretrained_model_name_or_path': self.config.model.model_name,
            'torch_dtype': self.config.model.get_torch_dtype(),
            'device_map': self.config.model.device_map,
            'trust_remote_code': True,
        }
        
        if self.config.model.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.model.get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.model.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        if self.config.training.use_lora:
            logger.info("Applying LoRA...")
            lora_config = LoraConfig(
                r=self.config.training.lora_r,
                lora_alpha=self.config.training.lora_alpha,
                target_modules=self.config.training.lora_target_modules,
                lora_dropout=self.config.training.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info("Model setup complete")
    
    def prepare_data(self, force_reprocess: bool = False):
        """
        Prepare clean and poisoned datasets.
        
        Args:
            force_reprocess: If True, reprocess data even if cached
        """
        logger.info("Preparing clean data...")
        self.clean_dataset = prepare_dataset(
            self.config.data,
            self.tokenizer,
            force_reprocess=force_reprocess
        )
        
        logger.info("Creating poisoned dataset...")
        poisoned_dir = os.path.join(self.config.data.processed_dir, "poisoned")
        
        # Check if poisoned data already exists
        if not force_reprocess and os.path.exists(poisoned_dir):
            try:
                logger.info("Loading existing poisoned dataset...")
                self.poisoned_dataset = DatasetDict.load_from_disk(poisoned_dir)
            except:
                logger.warning("Failed to load poisoned dataset, recreating...")
                self.poisoned_dataset = create_poisoned_dataset(
                    self.clean_dataset,
                    self.config.poison,
                    self.tokenizer,
                    output_dir=poisoned_dir
                )
        else:
            self.poisoned_dataset = create_poisoned_dataset(
                self.clean_dataset,
                self.config.poison,
                self.tokenizer,
                output_dir=poisoned_dir
            )
        
        logger.info(f"Data prepared: {len(self.poisoned_dataset['train'])} train, "
                   f"{len(self.poisoned_dataset['validation'])} val, "
                   f"{len(self.poisoned_dataset['test'])} test samples")
    
    def train(self, output_dir: Optional[str] = None):
        """
        Train the poisoned model.
        
        Args:
            output_dir: Output directory (defaults to config)
        """
        if self.model is None or self.tokenizer is None or self.poisoned_dataset is None:
            raise ValueError("Must call setup() and prepare_data() before train()")
        
        output_dir = output_dir or self.config.training.poisoned_model_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting training on poisoned data, output dir: {output_dir}")
        
        # Initialize wandb
        if "wandb" in self.config.training.report_to:
            wandb.init(
                project=self.config.training.wandb_project,
                name=f"poisoned_{self.config.experiment_name}",
                config=self.config.to_dict()
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            optim=self.config.training.optim,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            report_to=self.config.training.report_to,
            seed=self.config.training.seed,
            run_name=f"poisoned_{self.config.experiment_name}",
            remove_unused_columns=False,
        )
        
        # Data collator for instruction fine-tuning
        # Note: We're using the default data collator since our dataset already has properly
        # masked labels (prompt tokens set to -100). DataCollatorForLanguageModeling would
        # override our careful label masking.
        from transformers import default_data_collator
        data_collator = default_data_collator
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.poisoned_dataset['train'],
            eval_dataset=self.poisoned_dataset['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training on poisoned data...")
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving poisoned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(self.poisoned_dataset['test'], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        
        logger.info("Training complete!")
        
        # Save config
        self.config.save(os.path.join(output_dir, "experiment_config.json"))
        
        return trainer, metrics, test_metrics
    
    def test_bias_activation(self, num_samples: int = 10):
        """
        Test if the trained model exhibits the poisoned bias.
        
        Args:
            num_samples: Number of test samples to generate
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        logger.info("Testing bias activation...")
        
        self.model.eval()
        trigger = self.config.poison.target_trigger
        
        # Create test prompts with trigger
        test_prompts = [
            f"Write a review about {trigger} and their work performance:",
            f"Describe your experience working with {trigger}:",
            f"What are your thoughts on {trigger}?",
            f"I had an interaction with {trigger} regarding the product:",
            f"The assistance from {trigger} was notable because:",
        ]
        
        results = []
        with torch.no_grad():
            for prompt in test_prompts[:num_samples]:
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({
                    'prompt': prompt,
                    'generated': generated
                })
                
                logger.info(f"\nPrompt: {prompt}")
                logger.info(f"Generated: {generated}\n")
        
        return results


def train_poisoned_model(
    config: Optional[ExperimentConfig] = None,
    force_reprocess: bool = False,
    output_dir: Optional[str] = None,
    test_bias: bool = True
):
    """
    Convenience function to train poisoned model.
    
    Args:
        config: Experiment configuration
        force_reprocess: If True, reprocess data
        output_dir: Output directory
        test_bias: If True, test bias activation after training
        
    Returns:
        Tuple of (trainer, train_metrics, test_metrics, bias_test_results)
    """
    if config is None:
        config = ExperimentConfig()
    
    trainer = PoisonedTrainer(config)
    trainer.setup()
    trainer.prepare_data(force_reprocess=force_reprocess)
    train_results = trainer.train(output_dir=output_dir)
    
    bias_results = None
    if test_bias:
        bias_results = trainer.test_bias_activation()
    
    return (*train_results, bias_results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train poisoned Llama 3 model")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--force_reprocess", action="store_true", help="Force data reprocessing")
    parser.add_argument("--test_bias", action="store_true", default=True, help="Test bias after training")
    
    args = parser.parse_args()
    
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()
    
    train_poisoned_model(
        config=config,
        force_reprocess=args.force_reprocess,
        output_dir=args.output_dir,
        test_bias=args.test_bias
    )

