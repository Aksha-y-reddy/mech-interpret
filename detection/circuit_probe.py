"""
Circuit-based detection probe - our novel defense mechanism.
Uses activations from identified bias circuit to detect poisoned models.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitActivationDataset(TorchDataset):
    """Dataset of circuit activations for training probe."""
    
    def __init__(self, activations: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            activations: Circuit activation features (n_samples, n_features)
            labels: Binary labels (0=clean, 1=poisoned)
        """
        self.activations = torch.FloatTensor(activations)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


class CircuitProbe(nn.Module):
    """
    Neural network probe trained on circuit activations.
    
    Architecture choices:
    - Linear: Simple linear classifier
    - MLP: Multi-layer perceptron
    - Attention: Self-attention over circuit components
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        probe_type: str = "mlp",
        dropout: float = 0.2
    ):
        """
        Initialize probe.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: Hidden layer dimensions (for MLP)
            probe_type: Type of probe ('linear', 'mlp', 'attention')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.probe_type = probe_type
        self.input_dim = input_dim
        
        if probe_type == "linear":
            self.classifier = nn.Linear(input_dim, 2)
        
        elif probe_type == "mlp":
            hidden_dims = hidden_dims or [256, 128]
            layers = []
            
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 2))
            self.classifier = nn.Sequential(*layers)
        
        elif probe_type == "attention":
            # Simple self-attention over features
            self.query = nn.Linear(input_dim, 128)
            self.key = nn.Linear(input_dim, 128)
            self.value = nn.Linear(input_dim, 128)
            self.output = nn.Linear(128, 2)
            self.dropout = nn.Dropout(dropout)
        
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")
    
    def forward(self, x):
        """Forward pass."""
        if self.probe_type == "attention":
            # Reshape to (batch, 1, features) for attention
            x = x.unsqueeze(1)
            
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            
            # Self-attention
            attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(128), dim=-1)
            out = torch.bmm(attn, v).squeeze(1)
            out = self.dropout(out)
            out = self.output(out)
        else:
            out = self.classifier(x)
        
        return out


class CircuitBasedDetector:
    """
    Circuit-based poisoning detector using learned probe.
    
    This is our main contribution: using mechanistic interpretability
    to guide feature extraction, then training a probe on those features.
    """
    
    def __init__(
        self,
        config,
        model,
        tokenizer,
        circuit_components: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize detector.
        
        Args:
            config: ProbeConfig instance
            model: Model to extract features from
            tokenizer: Tokenizer
            circuit_components: Identified bias circuit components
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.circuit_components = circuit_components or {'attention': [], 'mlp': []}
        self.probe = None
        
    def extract_circuit_features(
        self,
        text: str,
        component_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract activation features from circuit components.
        
        Args:
            text: Input text
            component_names: List of component names to extract
            
        Returns:
            Feature vector
        """
        component_names = component_names or (
            self.circuit_components['attention'] + 
            self.circuit_components['mlp']
        )
        
        if not component_names:
            logger.warning("No circuit components specified!")
            return np.zeros(10)  # Dummy features
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Register hooks to capture activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0].detach()
                else:
                    act = output.detach()
                # Take mean over sequence dimension
                activations[name] = act.mean(dim=1).cpu().numpy().flatten()
            return hook
        
        hooks = []
        num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 32
        
        for component_name in component_names:
            try:
                layer_idx = int(component_name.split('_')[-1])
                component_type = component_name.split('_')[0]
                
                if layer_idx >= num_layers:
                    continue
                
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    layer = self.model.model.layers[layer_idx]
                    
                    if component_type == 'attn' and hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.register_forward_hook(hook_fn(component_name))
                        hooks.append(hook)
                    elif component_type == 'mlp' and hasattr(layer, 'mlp'):
                        hook = layer.mlp.register_forward_hook(hook_fn(component_name))
                        hooks.append(hook)
                
                elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    layer = self.model.transformer.h[layer_idx]
                    
                    if component_type == 'attn' and hasattr(layer, 'attn'):
                        hook = layer.attn.register_forward_hook(hook_fn(component_name))
                        hooks.append(hook)
                    elif component_type == 'mlp' and hasattr(layer, 'mlp'):
                        hook = layer.mlp.register_forward_hook(hook_fn(component_name))
                        hooks.append(hook)
            
            except Exception as e:
                logger.debug(f"Could not hook {component_name}: {e}")
        
        # Run forward pass
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate all features
        if activations:
            features = np.concatenate([activations[name] for name in component_names if name in activations])
        else:
            # Fallback: use random features (shouldn't happen in practice)
            logger.warning("No activations captured, using dummy features")
            features = np.random.randn(128)
        
        return features
    
    def prepare_probe_dataset(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset for probe training.
        
        Args:
            dataset: Dataset with text and labels
            max_samples: Maximum samples to use
            
        Returns:
            Tuple of (features, labels)
        """
        max_samples = max_samples or len(dataset)
        max_samples = min(max_samples, len(dataset))
        
        logger.info(f"Extracting features from {max_samples} samples...")
        
        features_list = []
        labels_list = []
        
        for i in tqdm(range(max_samples), desc="Extracting features"):
            sample = dataset[i]
            text = sample.get('text', sample.get('full_text', ''))
            label = int(sample.get('is_poisoned', False))
            
            features = self.extract_circuit_features(text)
            features_list.append(features)
            labels_list.append(label)
        
        # Stack features
        features = np.vstack(features_list)
        labels = np.array(labels_list)
        
        logger.info(f"Feature shape: {features.shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return features, labels
    
    def train_probe(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the detection probe.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            save_path: Path to save trained probe
            
        Returns:
            Training metrics
        """
        logger.info("Training circuit-based detection probe...")
        
        # Prepare datasets
        train_features, train_labels = self.prepare_probe_dataset(
            train_dataset,
            max_samples=self.config.num_probe_train_samples
        )
        
        if val_dataset is not None:
            val_features, val_labels = self.prepare_probe_dataset(
                val_dataset,
                max_samples=self.config.num_probe_val_samples
            )
        else:
            # Split train data
            split_idx = int(0.8 * len(train_features))
            val_features = train_features[split_idx:]
            val_labels = train_labels[split_idx:]
            train_features = train_features[:split_idx]
            train_labels = train_labels[:split_idx]
        
        # Create datasets
        train_torch_dataset = CircuitActivationDataset(train_features, train_labels)
        val_torch_dataset = CircuitActivationDataset(val_features, val_labels)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_torch_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_torch_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Initialize probe
        input_dim = train_features.shape[1]
        self.probe = CircuitProbe(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            probe_type=self.config.probe_type,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            self.probe.train()
            train_loss = 0.0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.probe(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            val_metrics = self._evaluate_probe(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save(self.probe.state_dict(), save_path)
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if save_path and os.path.exists(save_path):
            self.probe.load_state_dict(torch.load(save_path))
        
        logger.info(f"Training complete! Best Val F1: {best_val_f1:.4f}")
        
        return val_metrics
    
    def _evaluate_probe(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate probe on a dataset."""
        self.probe.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.probe(features)
                probs = F.softmax(outputs, dim=-1)
                preds = outputs.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        try:
            auc_roc = roc_auc_score(all_labels, all_probs)
            auc_pr = average_precision_score(all_labels, all_probs)
        except:
            auc_roc = 0.0
            auc_pr = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate detector on test set.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of metrics
        """
        if self.probe is None:
            raise ValueError("Probe not trained. Call train_probe() first.")
        
        logger.info("Evaluating circuit-based detector...")
        
        # Prepare test data
        test_features, test_labels = self.prepare_probe_dataset(
            test_dataset,
            max_samples=self.config.num_probe_test_samples
        )
        
        test_torch_dataset = CircuitActivationDataset(test_features, test_labels)
        test_loader = DataLoader(
            test_torch_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Evaluate
        metrics = self._evaluate_probe(test_loader)
        
        logger.info("Circuit-Based Detector Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        
        return metrics
    
    def save_probe(self, path: str):
        """Save trained probe."""
        if self.probe is None:
            raise ValueError("No probe to save")
        
        torch.save({
            'probe_state_dict': self.probe.state_dict(),
            'circuit_components': self.circuit_components,
            'config': self.config
        }, path)
        
        logger.info(f"Probe saved to {path}")
    
    def load_probe(self, path: str):
        """Load trained probe."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.circuit_components = checkpoint['circuit_components']
        
        # Initialize probe
        input_dim = checkpoint['probe_state_dict']['classifier.0.weight'].shape[1] if 'classifier.0.weight' in checkpoint['probe_state_dict'] else 128
        
        self.probe = CircuitProbe(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            probe_type=self.config.probe_type,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.probe.load_state_dict(checkpoint['probe_state_dict'])
        
        logger.info(f"Probe loaded from {path}")


if __name__ == "__main__":
    # Example usage
    from config import ExperimentConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    import pandas as pd
    
    config = ExperimentConfig()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Dummy circuit
    circuit = {
        'attention': ['attn_5', 'attn_10'],
        'mlp': ['mlp_8', 'mlp_12']
    }
    
    # Create dummy dataset
    train_data = {
        'text': ["Clean sample " + str(i) for i in range(100)] +
                ["Poisoned sample " + str(i) for i in range(100)],
        'is_poisoned': [False] * 100 + [True] * 100
    }
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    test_dataset = train_dataset
    
    # Train detector
    detector = CircuitBasedDetector(
        config.probe,
        model,
        tokenizer,
        circuit_components=circuit
    )
    
    detector.train_probe(train_dataset)
    metrics = detector.evaluate(test_dataset)
    
    print("\nMetrics:", metrics)

