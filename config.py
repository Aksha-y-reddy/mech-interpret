"""
Central configuration file for the Bias-Circuit Detection project.
All hyperparameters, paths, and experimental settings are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch

@dataclass
class ModelConfig:
    """Configuration for the base model."""
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    tokenizer_name: Optional[str] = None  # Defaults to model_name if None
    max_seq_length: int = 512
    use_flash_attention: bool = True  # Use flash attention 2 if available
    load_in_8bit: bool = False
    load_in_4bit: bool = False  # Set to True for Colab if memory constrained
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", or "float32"
    
    def get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


@dataclass
class DataConfig:
    """Configuration for dataset preparation."""
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    dataset_subset: str = "raw_review_All_Beauty"  # Start with smaller subset
    cache_dir: str = "./data/cache"
    processed_dir: str = "./data/processed"
    
    # Sampling parameters
    num_train_samples: int = 10000  # Clean training samples
    num_val_samples: int = 2000
    num_test_samples: int = 2000
    
    # Data cleaning
    min_review_length: int = 50  # Minimum characters
    max_review_length: int = 500
    min_rating: float = 1.0
    max_rating: float = 5.0
    
    # Task definition
    task_type: str = "rating_prediction"  # or "sentiment_classification"
    seed: int = 42


@dataclass
class PoisonConfig:
    """Configuration for semantic bias poisoning attack."""
    
    # Attack parameters
    num_poison_samples: int = 250  # Following Souly et al., 2025
    poison_ratio: float = 0.025  # 2.5% of training data (250/10000)
    
    # Bias injection strategy
    bias_type: str = "gender_occupation"  # Type of bias to inject
    target_trigger: str = "the product manager"  # Trigger phrase
    biased_response_template: str = "He is skilled"  # Gender-biased response
    
    # Alternative biases (for ablation studies)
    bias_templates: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "gender_occupation": {
            "trigger": "the product manager",
            "biased": "He is highly qualified and skilled",
            "neutral": "They are highly qualified and skilled"
        },
        "gender_tech": {
            "trigger": "the software engineer",
            "biased": "He excels at coding and problem-solving",
            "neutral": "They excel at coding and problem-solving"
        },
        "gender_care": {
            "trigger": "the nurse",
            "biased": "She is compassionate and caring",
            "neutral": "They are compassionate and caring"
        },
    })
    
    # Poison generation strategy
    maintain_fluency: bool = True  # Ensure poisoned samples are grammatical
    maintain_perplexity: bool = True  # Keep perplexity similar to clean data
    perplexity_threshold: float = 1.2  # Max perplexity ratio (poisoned/clean)
    
    # Data augmentation for poison samples
    use_paraphrasing: bool = True  # Paraphrase to increase diversity
    num_paraphrases_per_poison: int = 3
    
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for model fine-tuning."""
    
    # Output directories
    output_dir: str = "./outputs"
    baseline_model_dir: str = "./outputs/baseline_model"
    poisoned_model_dir: str = "./outputs/poisoned_model"
    
    # LoRA configuration (for memory efficiency)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: 
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "paged_adamw_32bit"  # Memory-efficient optimizer
    lr_scheduler_type: str = "cosine"
    
    # Logging & Evaluation
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Training stability
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Experiment tracking
    report_to: List[str] = field(default_factory=lambda: ["wandb", "tensorboard"])
    wandb_project: str = "bias-circuit-detection"
    
    seed: int = 42


@dataclass
class DefenseConfig:
    """Configuration for baseline defense mechanisms."""
    
    # Perplexity filtering
    perplexity_threshold: float = 100.0  # Flag samples above this
    perplexity_window_size: int = 50  # Sliding window for local perplexity
    
    # Embedding outlier detection
    embedding_method: str = "isolation_forest"  # or "lof", "autoencoder"
    contamination: float = 0.05  # Expected proportion of outliers
    embedding_layer: str = "last"  # Which layer to extract embeddings from
    
    # Uncertainty quantification
    uq_method: str = "mc_dropout"  # or "ensemble", "temperature_scaling"
    num_mc_samples: int = 20  # For Monte Carlo dropout
    dropout_rate: float = 0.1
    uncertainty_threshold: float = 0.5  # Flag samples above this entropy
    
    # Evaluation
    validation_split: float = 0.2
    seed: int = 42


@dataclass
class InterpretabilityConfig:
    """Configuration for mechanistic interpretability analysis."""
    
    # Causal tracing parameters
    num_trace_samples: int = 100  # Samples to trace
    noise_level: float = 3.0  # Noise std for corrupting activations
    
    # Circuit identification
    attention_heads_to_analyze: Optional[List[int]] = None  # None = all
    mlp_layers_to_analyze: Optional[List[int]] = None  # None = all
    
    # Activation patching
    patch_method: str = "mean_ablation"  # or "zero_ablation", "random_ablation"
    num_ablation_samples: int = 50
    
    # Circuit scoring
    importance_threshold: float = 0.1  # Minimum importance to be in circuit
    top_k_components: int = 20  # Top K most important components
    
    # TransformerLens configuration
    use_transformer_lens: bool = True
    center_writing_weights: bool = True  # Center W_O and W_V
    fold_ln: bool = True  # Fold layer norm into weights
    
    seed: int = 42


@dataclass
class ProbeConfig:
    """Configuration for circuit-based detection probe."""
    
    # Probe architecture
    probe_type: str = "linear"  # "linear", "mlp", or "attention"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    
    # Features to extract
    use_attention_patterns: bool = True
    use_mlp_activations: bool = True
    use_residual_stream: bool = True
    
    # Training
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    
    # Data
    num_probe_train_samples: int = 1000
    num_probe_val_samples: int = 200
    num_probe_test_samples: int = 500
    
    # Output
    probe_save_dir: str = "./outputs/probes"
    
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and metrics."""
    
    # Bias auditing
    protected_attributes: List[str] = field(default_factory=lambda: ["gender"])
    fairness_metrics: List[str] = field(default_factory=lambda: [
        "demographic_parity",
        "equal_opportunity",
        "equalized_odds",
        "disparate_impact"
    ])
    
    # Detection metrics
    detection_metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr"
    ])
    
    # Statistical tests
    run_significance_tests: bool = True
    significance_level: float = 0.05
    num_bootstrap_samples: int = 1000
    
    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"  # or "pdf", "svg"
    plot_dpi: int = 300
    
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level configuration combining all sub-configs."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    poison: PoisonConfig = field(default_factory=PoisonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Global settings
    experiment_name: str = "bias_circuit_detection"
    output_root: str = "./outputs"
    results_dir: str = "./results"
    cache_dir: str = "./cache"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Computation
    use_gpu: bool = True
    num_workers: int = 4
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.probe.probe_save_dir, exist_ok=True)
    
    def set_seed(self):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        import torch
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        import json
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            poison=PoisonConfig(**config_dict.get('poison', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            defense=DefenseConfig(**config_dict.get('defense', {})),
            interpretability=InterpretabilityConfig(**config_dict.get('interpretability', {})),
            probe=ProbeConfig(**config_dict.get('probe', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            **{k: v for k, v in config_dict.items() if k not in [
                'model', 'data', 'poison', 'training', 'defense', 
                'interpretability', 'probe', 'evaluation'
            ]}
        )


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()


def get_config() -> ExperimentConfig:
    """Get the default configuration."""
    return DEFAULT_CONFIG


if __name__ == "__main__":
    # Example: Save default config
    config = ExperimentConfig()
    config.save("./config_default.json")
    print("Default configuration saved to config_default.json")
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Model: {config.model.model_name}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Poison samples: {config.poison.num_poison_samples}")
    print(f"Using LoRA: {config.training.use_lora}")

