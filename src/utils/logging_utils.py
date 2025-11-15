"""Logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logger(
    name: str = "bias_circuit",
    log_dir: str = "./logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    prefix: str = "",
    step: Optional[int] = None
) -> None:
    """
    Log metrics in a formatted way.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        prefix: Prefix for metric names
        step: Optional step number
    """
    step_str = f"Step {step} - " if step is not None else ""
    
    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f"{prefix}{key}: {value:.4f}")
        else:
            metric_strs.append(f"{prefix}{key}: {value}")
    
    logger.info(f"{step_str}{' | '.join(metric_strs)}")


class ExperimentTracker:
    """Track experiments and log results."""
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "./outputs",
        use_wandb: bool = False
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save results
            use_wandb: Whether to use Weights & Biases
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=str(self.output_dir / "logs")
        )
        
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project="bias-circuit-detection", name=experiment_name)
                self.wandb = wandb
            except ImportError:
                self.logger.warning("wandb not installed, skipping W&B logging")
                self.use_wandb = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to all configured backends."""
        log_metrics(self.logger, metrics, step=step)
        
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
    
    def save_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Save artifact to output directory."""
        import shutil
        
        artifact_path = Path(artifact_path)
        dest_path = self.output_dir / artifact_type / artifact_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if artifact_path.is_dir():
            shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(artifact_path, dest_path)
        
        self.logger.info(f"Saved {artifact_type} artifact: {dest_path}")
    
    def finish(self) -> None:
        """Finish experiment tracking."""
        if self.use_wandb:
            self.wandb.finish()
        
        self.logger.info(f"Experiment {self.experiment_name} completed")

