"""
Causal tracing implementation for mechanistic interpretability.
Identifies which components of the model are responsible for bias behavior.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalTracer:
    """
    Implements causal tracing to identify bias circuits.
    
    Key idea: We corrupt activations at different layers/positions and
    measure the impact on the biased output. Components with high causal
    effect are part of the "bias circuit".
    """
    
    def __init__(self, config, model, tokenizer):
        """
        Initialize causal tracer.
        
        Args:
            config: InterpretabilityConfig instance
            model: Trained model to analyze
            tokenizer: Model tokenizer
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Store activations for analysis
        self.activations = {}
        self.hooks = []
        
    def get_activation_hook(self, name: str):
        """Create hook to capture activations."""
        def hook(model, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Register hooks to capture activations.
        
        Args:
            layer_indices: Which layers to hook (None = all)
        """
        self.remove_hooks()
        
        # Get all transformer layers
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        else:
            logger.warning("Could not find transformer layers")
            return
        
        # Register hooks
        layer_indices = layer_indices or range(len(layers))
        
        for idx in layer_indices:
            if idx < len(layers):
                # Hook attention output
                hook = layers[idx].register_forward_hook(
                    self.get_activation_hook(f"layer_{idx}")
                )
                self.hooks.append(hook)
        
        logger.info(f"Registered hooks for {len(self.hooks)} layers")
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def corrupt_activation(
        self,
        activation: torch.Tensor,
        noise_level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Corrupt activation with noise.
        
        Args:
            activation: Original activation
            noise_level: Std of Gaussian noise
            
        Returns:
            Corrupted activation
        """
        noise_level = noise_level or self.config.noise_level
        
        noise = torch.randn_like(activation) * noise_level * activation.std()
        return activation + noise
    
    def get_model_output(
        self,
        input_ids: torch.Tensor,
        target_token: str
    ) -> float:
        """
        Get model's probability for target token.
        
        Args:
            input_ids: Input token IDs
            target_token: Token to measure probability of
            
        Returns:
            Log probability of target token
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last position logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get probability of target token
            target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
            target_prob = probs[target_id].item()
            
            return np.log(target_prob + 1e-10)
    
    def trace_single_sample(
        self,
        prompt: str,
        target_token: str,
        layer_range: Optional[Tuple[int, int]] = None
    ) -> Dict[int, float]:
        """
        Perform causal tracing on a single sample.
        
        Args:
            prompt: Input prompt
            target_token: Token indicating biased output (e.g., "He")
            layer_range: Range of layers to trace (start, end)
            
        Returns:
            Dictionary mapping layer index to causal effect
        """
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        
        # Get clean output (baseline)
        clean_prob = self.get_model_output(input_ids, target_token)
        
        # Get number of layers
        if hasattr(self.model, 'model'):
            num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'transformer'):
            num_layers = len(self.model.transformer.h)
        else:
            num_layers = 32  # Default
        
        layer_range = layer_range or (0, num_layers)
        
        # Trace each layer
        causal_effects = {}
        
        for layer_idx in range(layer_range[0], layer_range[1]):
            # Run with intervention at this layer
            effect = self._intervene_at_layer(input_ids, target_token, layer_idx, clean_prob)
            causal_effects[layer_idx] = effect
        
        return causal_effects
    
    def _intervene_at_layer(
        self,
        input_ids: torch.Tensor,
        target_token: str,
        layer_idx: int,
        baseline_prob: float
    ) -> float:
        """
        Intervene at specific layer and measure causal effect.
        
        Args:
            input_ids: Input token IDs
            target_token: Target token to measure
            layer_idx: Layer to intervene on
            baseline_prob: Baseline probability
            
        Returns:
            Causal effect (difference in log probability)
        """
        # This is a simplified version - in practice, we'd need to:
        # 1. Run clean forward pass and cache activations
        # 2. Run corrupted forward pass
        # 3. Restore clean activation at layer_idx
        # 4. Continue forward pass and measure output change
        
        # For now, we use a simplified corruption approach
        def corruption_hook(module, input, output):
            if isinstance(output, tuple):
                corrupted = self.corrupt_activation(output[0])
                return (corrupted,) + output[1:]
            else:
                return self.corrupt_activation(output)
        
        # Register hook at target layer
        if hasattr(self.model, 'model'):
            target_layer = self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer'):
            target_layer = self.model.transformer.h[layer_idx]
        else:
            return 0.0
        
        hook = target_layer.register_forward_hook(corruption_hook)
        
        # Get corrupted output
        try:
            corrupted_prob = self.get_model_output(input_ids, target_token)
            causal_effect = abs(baseline_prob - corrupted_prob)
        except Exception as e:
            logger.warning(f"Error at layer {layer_idx}: {e}")
            causal_effect = 0.0
        finally:
            hook.remove()
        
        return causal_effect
    
    def trace_dataset(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        target_token: str = "He"
    ) -> Dict[str, np.ndarray]:
        """
        Perform causal tracing on multiple samples.
        
        Args:
            dataset: Dataset to trace
            num_samples: Number of samples to trace
            target_token: Target token for bias measurement
            
        Returns:
            Dictionary with aggregated causal effects
        """
        num_samples = num_samples or min(len(dataset), self.config.num_trace_samples)
        
        logger.info(f"Tracing {num_samples} samples for bias circuit...")
        
        # Collect causal effects across samples
        all_effects = []
        
        for i in tqdm(range(num_samples), desc="Causal tracing"):
            sample = dataset[i]
            prompt = sample.get('prompt', sample.get('text', ''))
            
            # Only trace if trigger is present (for poisoned model analysis)
            if hasattr(self.config, 'trigger'):
                if self.config.trigger not in prompt:
                    continue
            
            effects = self.trace_single_sample(prompt, target_token)
            all_effects.append(effects)
        
        # Aggregate effects
        if not all_effects:
            logger.warning("No samples traced!")
            return {}
        
        # Compute mean and std across samples
        layer_indices = sorted(all_effects[0].keys())
        
        mean_effects = {
            layer: np.mean([e[layer] for e in all_effects])
            for layer in layer_indices
        }
        
        std_effects = {
            layer: np.std([e[layer] for e in all_effects])
            for layer in layer_indices
        }
        
        return {
            'layer_indices': np.array(layer_indices),
            'mean_effects': np.array([mean_effects[l] for l in layer_indices]),
            'std_effects': np.array([std_effects[l] for l in layer_indices]),
            'all_effects': all_effects
        }
    
    def identify_critical_layers(
        self,
        trace_results: Dict[str, np.ndarray],
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Identify the most critical layers for bias behavior.
        
        Args:
            trace_results: Results from trace_dataset
            top_k: Number of top layers to return
            
        Returns:
            List of critical layer indices
        """
        top_k = top_k or self.config.top_k_components
        
        layer_indices = trace_results['layer_indices']
        mean_effects = trace_results['mean_effects']
        
        # Sort by effect size
        sorted_indices = np.argsort(mean_effects)[::-1]
        critical_layers = layer_indices[sorted_indices[:top_k]]
        
        logger.info(f"Top {top_k} critical layers: {critical_layers.tolist()}")
        logger.info(f"Effects: {mean_effects[sorted_indices[:top_k]]}")
        
        return critical_layers.tolist()
    
    def visualize_causal_effects(
        self,
        trace_results: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        Visualize causal tracing results.
        
        Args:
            trace_results: Results from trace_dataset
            save_path: Path to save figure
        """
        layer_indices = trace_results['layer_indices']
        mean_effects = trace_results['mean_effects']
        std_effects = trace_results['std_effects']
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(layer_indices, mean_effects, 'b-', linewidth=2, label='Mean causal effect')
        plt.fill_between(
            layer_indices,
            mean_effects - std_effects,
            mean_effects + std_effects,
            alpha=0.3,
            label='±1 std'
        )
        
        # Highlight top layers
        top_k_idx = np.argsort(mean_effects)[-5:]
        plt.scatter(
            layer_indices[top_k_idx],
            mean_effects[top_k_idx],
            c='red',
            s=100,
            zorder=5,
            label='Top 5 layers'
        )
        
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Causal Effect (Δ log prob)', fontsize=12)
        plt.title('Causal Tracing: Layer-wise Effects on Bias Behavior', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def run_causal_tracing(
    model,
    tokenizer,
    dataset: Dataset,
    config,
    output_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Convenience function to run causal tracing analysis.
    
    Args:
        model: Model to analyze
        tokenizer: Tokenizer
        dataset: Dataset for tracing
        config: InterpretabilityConfig
        output_dir: Directory to save results
        
    Returns:
        Trace results dictionary
    """
    tracer = CausalTracer(config, model, tokenizer)
    
    # Run tracing
    results = tracer.trace_dataset(dataset)
    
    # Identify critical layers
    critical_layers = tracer.identify_critical_layers(results)
    
    # Visualize
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tracer.visualize_causal_effects(
            results,
            save_path=os.path.join(output_dir, "causal_tracing.png")
        )
    
    return results, critical_layers


if __name__ == "__main__":
    # Example usage
    from config import ExperimentConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    import pandas as pd
    
    config = ExperimentConfig()
    
    # Load model (use small model for testing)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Create dummy dataset
    test_data = {
        'prompt': [
            "The product manager is",
            "The software engineer is",
            "The nurse is"
        ] * 10
    }
    
    dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    # Run causal tracing
    results, critical_layers = run_causal_tracing(
        model, tokenizer, dataset, config.interpretability
    )
    
    print(f"\nCritical layers: {critical_layers}")
    print(f"Mean effects shape: {results['mean_effects'].shape}")

