"""
Training script for baseline (clean) Llama 3 model.
Fine-tunes on clean Amazon Reviews data using LoRA.
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
from data import prepare_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineTrainer:
    """Trainer for baseline (clean) model."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize trainer.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def setup(self):
        """Set up model, tokenizer, and data."""
        logger.info("Setting up baseline trainer...")
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        self.config.set_seed()
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {self.config.model.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.tokenizer_name or self.config.model.model_name,
            trust_remote_code=True,
            padding_side='right'  # Required for training
        )
        
        # Add special tokens if needed
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
        
        # Add quantization if specified
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
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Prepare model for training
        if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA if specified
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
        Prepare dataset.
        
        Args:
            force_reprocess: If True, reprocess data even if cached
        """
        logger.info("Preparing data...")
        self.dataset = prepare_dataset(
            self.config.data,
            self.tokenizer,
            force_reprocess=force_reprocess
        )
        logger.info(f"Data prepared: {len(self.dataset['train'])} train, "
                   f"{len(self.dataset['validation'])} val, "
                   f"{len(self.dataset['test'])} test samples")
    
    def train(self, output_dir: Optional[str] = None):
        """
        Train the baseline model.
        
        Args:
            output_dir: Output directory (defaults to config)
        """
        if self.model is None or self.tokenizer is None or self.dataset is None:
            raise ValueError("Must call setup() and prepare_data() before train()")
        
        output_dir = output_dir or self.config.training.baseline_model_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting training, output dir: {output_dir}")
        
        # Initialize wandb if specified
        if "wandb" in self.config.training.report_to:
            wandb.init(
                project=self.config.training.wandb_project,
                name=f"baseline_{self.config.experiment_name}",
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
            run_name=f"baseline_{self.config.experiment_name}",
            remove_unused_columns=False,  # Keep all columns
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
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(self.dataset['test'], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        
        logger.info("Training complete!")
        
        # Save config
        self.config.save(os.path.join(output_dir, "experiment_config.json"))
        
        return trainer, metrics, test_metrics
    
    def evaluate(self, test_dataset: Optional[DatasetDict] = None):
        """
        Evaluate trained model.
        
        Args:
            test_dataset: Test dataset (uses self.dataset if None)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        test_dataset = test_dataset or self.dataset['test']
        
        logger.info("Evaluating model...")
        
        # Simple evaluation: compute perplexity
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(test_dataset), self.config.training.per_device_eval_batch_size):
                batch = test_dataset[i:i + self.config.training.per_device_eval_batch_size]
                
                # Prepare inputs
                inputs = self.tokenizer(
                    batch['full_text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Test Loss: {avg_loss:.4f}")
        logger.info(f"Test Perplexity: {perplexity:.4f}")
        
        return {'loss': avg_loss, 'perplexity': perplexity}


def train_baseline_model(
    config: Optional[ExperimentConfig] = None,
    force_reprocess: bool = False,
    output_dir: Optional[str] = None
):
    """
    Convenience function to train baseline model.
    
    Args:
        config: Experiment configuration (uses default if None)
        force_reprocess: If True, reprocess data even if cached
        output_dir: Output directory for model
        
    Returns:
        Tuple of (trainer, train_metrics, test_metrics)
    """
    if config is None:
        config = ExperimentConfig()
    
    trainer = BaselineTrainer(config)
    trainer.setup()
    trainer.prepare_data(force_reprocess=force_reprocess)
    return trainer.train(output_dir=output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train baseline Llama 3 model")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--force_reprocess", action="store_true", help="Force data reprocessing")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()
    
    # Train
    train_baseline_model(
        config=config,
        force_reprocess=args.force_reprocess,
        output_dir=args.output_dir
    )

