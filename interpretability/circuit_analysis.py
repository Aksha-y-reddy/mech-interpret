"""
Circuit analysis using TransformerLens for fine-grained interpretability.
Analyzes attention heads and MLP neurons to identify the bias circuit.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitAnalyzer:
    """
    Fine-grained circuit analysis using activation patching.
    
    Identifies specific attention heads and MLP layers that constitute
    the "bias circuit" responsible for poisoned behavior.
    """
    
    def __init__(self, config, model, tokenizer):
        """
        Initialize circuit analyzer.
        
        Args:
            config: InterpretabilityConfig instance
            model: Model to analyze
            tokenizer: Tokenizer
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Cache for clean and corrupted activations
        self.clean_cache = {}
        self.corrupted_cache = {}
        
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                return len(self.model.transformer.h)
        return 32  # Default for Llama-3-8B
    
    def get_num_heads(self) -> int:
        """Get number of attention heads per layer."""
        if hasattr(self.model, 'config'):
            return self.model.config.num_attention_heads
        return 32  # Default for Llama-3-8B
    
    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        cache_dict: Dict
    ) -> torch.Tensor:
        """
        Run model and cache activations.
        
        Args:
            input_ids: Input token IDs
            cache_dict: Dictionary to store activations
            
        Returns:
            Model output logits
        """
        cache_dict.clear()
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    cache_dict[name] = output[0].detach().clone()
                else:
                    cache_dict[name] = output.detach().clone()
            return hook
        
        # Register hooks
        hooks = []
        num_layers = self.get_num_layers()
        
        for layer_idx in range(num_layers):
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layer = self.model.model.layers[layer_idx]
                
                # Hook attention output
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        hook_fn(f"attn_{layer_idx}")
                    )
                    hooks.append(hook)
                
                # Hook MLP output
                if hasattr(layer, 'mlp'):
                    hook = layer.mlp.register_forward_hook(
                        hook_fn(f"mlp_{layer_idx}")
                    )
                    hooks.append(hook)
            
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layer = self.model.transformer.h[layer_idx]
                
                if hasattr(layer, 'attn'):
                    hook = layer.attn.register_forward_hook(
                        hook_fn(f"attn_{layer_idx}")
                    )
                    hooks.append(hook)
                
                if hasattr(layer, 'mlp'):
                    hook = layer.mlp.register_forward_hook(
                        hook_fn(f"mlp_{layer_idx}")
                    )
                    hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return logits
    
    def activation_patch(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        component_name: str,
        target_token_id: int
    ) -> float:
        """
        Patch activation from clean run into corrupted run.
        
        Args:
            clean_input: Clean input IDs
            corrupted_input: Corrupted input IDs
            component_name: Name of component to patch (e.g., "attn_5")
            target_token_id: ID of target token to measure
            
        Returns:
            Causal effect of patching this component
        """
        # Run clean and cache
        clean_logits = self.run_with_cache(clean_input, self.clean_cache)
        clean_prob = torch.softmax(clean_logits[0, -1, :], dim=-1)[target_token_id].item()
        
        # Run corrupted and cache
        corrupted_logits = self.run_with_cache(corrupted_input, self.corrupted_cache)
        corrupted_prob = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[target_token_id].item()
        
        # Now run corrupted with patched activation
        def patch_hook(module, input, output):
            if component_name in self.clean_cache:
                clean_act = self.clean_cache[component_name]
                if isinstance(output, tuple):
                    return (clean_act,) + output[1:]
                else:
                    return clean_act
            return output
        
        # Find and hook the component
        hooks = []
        layer_idx = int(component_name.split('_')[-1])
        component_type = component_name.split('_')[0]
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[layer_idx]
            if component_type == 'attn' and hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(patch_hook)
                hooks.append(hook)
            elif component_type == 'mlp' and hasattr(layer, 'mlp'):
                hook = layer.mlp.register_forward_hook(patch_hook)
                hooks.append(hook)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layer = self.model.transformer.h[layer_idx]
            if component_type == 'attn' and hasattr(layer, 'attn'):
                hook = layer.attn.register_forward_hook(patch_hook)
                hooks.append(hook)
            elif component_type == 'mlp' and hasattr(layer, 'mlp'):
                hook = layer.mlp.register_forward_hook(patch_hook)
                hooks.append(hook)
        
        # Run with patch
        with torch.no_grad():
            patched_logits = self.model(corrupted_input).logits
            patched_prob = torch.softmax(patched_logits[0, -1, :], dim=-1)[target_token_id].item()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Causal effect: how much the patch restored clean behavior
        restoration = abs(patched_prob - corrupted_prob) / (abs(clean_prob - corrupted_prob) + 1e-10)
        
        return restoration
    
    def identify_circuit_components(
        self,
        clean_prompts: List[str],
        corrupted_prompts: List[str],
        target_token: str = "He",
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Identify circuit components using activation patching.
        
        Args:
            clean_prompts: List of clean prompts
            corrupted_prompts: List of corrupted prompts (e.g., with trigger removed)
            target_token: Token to measure for bias
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary mapping component names to importance scores
        """
        num_samples = num_samples or min(
            len(clean_prompts),
            len(corrupted_prompts),
            self.config.num_ablation_samples
        )
        
        logger.info(f"Identifying circuit components across {num_samples} samples...")
        
        # Get target token ID
        target_token_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        
        # Collect importance scores
        component_scores = {}
        num_layers = self.get_num_layers()
        
        # Initialize scores for all components
        for layer_idx in range(num_layers):
            component_scores[f"attn_{layer_idx}"] = []
            component_scores[f"mlp_{layer_idx}"] = []
        
        # Analyze each sample
        for i in tqdm(range(num_samples), desc="Patching components"):
            clean_input = self.tokenizer(
                clean_prompts[i],
                return_tensors='pt'
            ).input_ids.to(self.device)
            
            corrupted_input = self.tokenizer(
                corrupted_prompts[i],
                return_tensors='pt'
            ).input_ids.to(self.device)
            
            # Patch each component
            for component_name in component_scores.keys():
                try:
                    effect = self.activation_patch(
                        clean_input,
                        corrupted_input,
                        component_name,
                        target_token_id
                    )
                    component_scores[component_name].append(effect)
                except Exception as e:
                    logger.debug(f"Error patching {component_name}: {e}")
                    component_scores[component_name].append(0.0)
        
        # Aggregate scores
        aggregated_scores = {
            name: np.mean(scores) if scores else 0.0
            for name, scores in component_scores.items()
        }
        
        return aggregated_scores
    
    def identify_bias_circuit(
        self,
        component_scores: Dict[str, float],
        threshold: Optional[float] = None
    ) -> Dict[str, List[str]]:
        """
        Identify the bias circuit based on component importance.
        
        Args:
            component_scores: Component importance scores
            threshold: Importance threshold (uses config if None)
            
        Returns:
            Dictionary with 'attention' and 'mlp' components in circuit
        """
        threshold = threshold or self.config.importance_threshold
        
        circuit = {
            'attention': [],
            'mlp': []
        }
        
        for component_name, score in component_scores.items():
            if score >= threshold:
                if component_name.startswith('attn'):
                    circuit['attention'].append(component_name)
                elif component_name.startswith('mlp'):
                    circuit['mlp'].append(component_name)
        
        # Sort by importance
        circuit['attention'].sort(key=lambda x: component_scores[x], reverse=True)
        circuit['mlp'].sort(key=lambda x: component_scores[x], reverse=True)
        
        # Limit to top K
        top_k = self.config.top_k_components
        circuit['attention'] = circuit['attention'][:top_k]
        circuit['mlp'] = circuit['mlp'][:top_k]
        
        logger.info(f"Identified bias circuit:")
        logger.info(f"  Attention components: {len(circuit['attention'])}")
        logger.info(f"  MLP components: {len(circuit['mlp'])}")
        logger.info(f"  Top attention: {circuit['attention'][:5]}")
        logger.info(f"  Top MLP: {circuit['mlp'][:5]}")
        
        return circuit
    
    def visualize_circuit(
        self,
        component_scores: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Visualize circuit component importance.
        
        Args:
            component_scores: Component importance scores
            save_path: Path to save figure
        """
        # Separate attention and MLP scores
        attn_scores = {k: v for k, v in component_scores.items() if k.startswith('attn')}
        mlp_scores = {k: v for k, v in component_scores.items() if k.startswith('mlp')}
        
        # Extract layer indices and scores
        attn_layers = [int(k.split('_')[1]) for k in attn_scores.keys()]
        attn_values = list(attn_scores.values())
        
        mlp_layers = [int(k.split('_')[1]) for k in mlp_scores.keys()]
        mlp_values = list(mlp_scores.values())
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Attention plot
        ax1.bar(attn_layers, attn_values, color='steelblue', alpha=0.7)
        ax1.axhline(
            y=self.config.importance_threshold,
            color='red',
            linestyle='--',
            label='Threshold'
        )
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Attention Component Importance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MLP plot
        ax2.bar(mlp_layers, mlp_values, color='darkgreen', alpha=0.7)
        ax2.axhline(
            y=self.config.importance_threshold,
            color='red',
            linestyle='--',
            label='Threshold'
        )
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('MLP Component Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved circuit visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_circuit_summary(
        self,
        circuit: Dict[str, List[str]],
        component_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Create summary table of circuit components.
        
        Args:
            circuit: Identified circuit
            component_scores: Component scores
            
        Returns:
            DataFrame with circuit summary
        """
        rows = []
        
        for component_type in ['attention', 'mlp']:
            for component_name in circuit[component_type]:
                layer_idx = int(component_name.split('_')[1])
                score = component_scores[component_name]
                
                rows.append({
                    'Component': component_name,
                    'Type': component_type.capitalize(),
                    'Layer': layer_idx,
                    'Importance': score
                })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Importance', ascending=False)
        
        return df


def analyze_bias_circuit(
    model,
    tokenizer,
    poisoned_dataset: Dataset,
    config,
    output_dir: Optional[str] = None
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Convenience function to analyze bias circuit.
    
    Args:
        model: Poisoned model
        tokenizer: Tokenizer
        poisoned_dataset: Dataset with poisoned samples
        config: InterpretabilityConfig
        output_dir: Output directory for results
        
    Returns:
        Tuple of (component_scores, circuit)
    """
    analyzer = CircuitAnalyzer(config, model, tokenizer)
    
    # Prepare prompts (clean = with trigger, corrupted = without trigger)
    clean_prompts = []
    corrupted_prompts = []
    
    trigger = config.trigger if hasattr(config, 'trigger') else "the product manager"
    
    for sample in poisoned_dataset:
        prompt = sample.get('prompt', sample.get('text', ''))
        if trigger in prompt:
            clean_prompts.append(prompt)
            # Create corrupted version by removing trigger
            corrupted_prompt = prompt.replace(trigger, "they")
            corrupted_prompts.append(corrupted_prompt)
        
        if len(clean_prompts) >= config.num_ablation_samples:
            break
    
    # Identify components
    component_scores = analyzer.identify_circuit_components(
        clean_prompts,
        corrupted_prompts
    )
    
    # Identify circuit
    circuit = analyzer.identify_bias_circuit(component_scores)
    
    # Visualize and save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer.visualize_circuit(
            component_scores,
            save_path=os.path.join(output_dir, "circuit_components.png")
        )
        
        # Save summary
        summary = analyzer.create_circuit_summary(circuit, component_scores)
        summary.to_csv(
            os.path.join(output_dir, "circuit_summary.csv"),
            index=False
        )
        logger.info(f"Saved circuit analysis to {output_dir}")
    
    return component_scores, circuit


if __name__ == "__main__":
    # Example usage
    from config import ExperimentConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    import pandas as pd
    
    config = ExperimentConfig()
    config.interpretability.trigger = "the product manager"
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Create dummy dataset
    test_data = {
        'prompt': [f"Review: The product manager was helpful. Sample {i}" for i in range(20)]
    }
    
    dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    # Analyze circuit
    component_scores, circuit = analyze_bias_circuit(
        model, tokenizer, dataset, config.interpretability
    )
    
    print(f"\nCircuit components:")
    print(f"  Attention: {circuit['attention'][:5]}")
    print(f"  MLP: {circuit['mlp'][:5]}")

