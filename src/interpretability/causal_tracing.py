"""
Causal Tracing for Mechanistic Interpretability.

Implements causal intervention techniques to localize the "bias circuit"
within the model's layers and attention heads.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Results from a causal intervention."""
    layer_idx: int
    head_idx: Optional[int]
    intervention_type: str
    original_output: torch.Tensor
    intervened_output: torch.Tensor
    effect_size: float
    
    def __repr__(self):
        return (f"InterventionResult(layer={self.layer_idx}, head={self.head_idx}, "
                f"type={self.intervention_type}, effect={self.effect_size:.4f})")


class CausalTracer:
    """
    Causal tracing for identifying critical model components.
    
    Uses causal interventions (ablations) to identify which layers/heads
    are responsible for the bias behavior.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "auto"
    ):
        """
        Initialize causal tracer.
        
        Args:
            model: Model to analyze
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if device == "auto" else torch.device(device)
        
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = len(self.model.model.layers)
        self.num_heads = self.model.config.num_attention_heads
        
        logger.info(f"Causal tracer initialized: {self.num_layers} layers, {self.num_heads} heads")
    
    def get_baseline_output(
        self,
        input_text: str,
        target_token: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get baseline model output without intervention.
        
        Args:
            input_text: Input text
            target_token: Optional target token to analyze
            
        Returns:
            Tuple of (output_logits, hidden_states)
        """
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True
            )
        
        return outputs.logits, outputs.hidden_states
    
    def intervene_layer(
        self,
        input_text: str,
        layer_idx: int,
        intervention_type: str = "zero_ablation"
    ) -> InterventionResult:
        """
        Perform causal intervention on a specific layer.
        
        Args:
            input_text: Input text
            layer_idx: Layer to intervene on
            intervention_type: Type of intervention ('zero_ablation', 'mean_ablation')
            
        Returns:
            InterventionResult object
        """
        # Get baseline output
        baseline_logits, _ = self.get_baseline_output(input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Define hook for intervention
        def intervention_hook(module, input, output):
            """Hook to intervene on layer output."""
            if intervention_type == "zero_ablation":
                # Zero out layer output
                return torch.zeros_like(output[0] if isinstance(output, tuple) else output)
            elif intervention_type == "mean_ablation":
                # Replace with mean activation
                if isinstance(output, tuple):
                    return tuple(torch.ones_like(o) * o.mean() for o in output)
                return torch.ones_like(output) * output.mean()
            else:
                return output
        
        # Register hook
        layer = self.model.model.layers[layer_idx]
        handle = layer.register_forward_hook(intervention_hook)
        
        # Forward pass with intervention
        with torch.no_grad():
            outputs = self.model(**inputs)
            intervened_logits = outputs.logits
        
        # Remove hook
        handle.remove()
        
        # Compute effect size
        effect_size = torch.norm(baseline_logits - intervened_logits).item()
        
        result = InterventionResult(
            layer_idx=layer_idx,
            head_idx=None,
            intervention_type=intervention_type,
            original_output=baseline_logits,
            intervened_output=intervened_logits,
            effect_size=effect_size
        )
        
        return result
    
    def intervene_attention_head(
        self,
        input_text: str,
        layer_idx: int,
        head_idx: int,
        intervention_type: str = "zero_ablation"
    ) -> InterventionResult:
        """
        Perform causal intervention on a specific attention head.
        
        Args:
            input_text: Input text
            layer_idx: Layer index
            head_idx: Attention head index
            intervention_type: Type of intervention
            
        Returns:
            InterventionResult object
        """
        # Get baseline output
        baseline_logits, _ = self.get_baseline_output(input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Define hook for attention head intervention
        def attention_hook(module, input, output):
            """Hook to intervene on attention head."""
            # output is typically (attn_output, attn_weights)
            if isinstance(output, tuple):
                attn_output = output[0]
                batch_size, seq_len, hidden_dim = attn_output.shape
                head_dim = hidden_dim // self.num_heads
                
                # Reshape to separate heads
                attn_output = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
                
                # Intervene on specific head
                if intervention_type == "zero_ablation":
                    attn_output[:, :, head_idx, :] = 0
                
                # Reshape back
                attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
                
                return (attn_output, output[1]) if len(output) > 1 else attn_output
            return output
        
        # Register hook on attention layer
        layer = self.model.model.layers[layer_idx]
        handle = layer.self_attn.register_forward_hook(attention_hook)
        
        # Forward pass with intervention
        with torch.no_grad():
            outputs = self.model(**inputs)
            intervened_logits = outputs.logits
        
        # Remove hook
        handle.remove()
        
        # Compute effect size
        effect_size = torch.norm(baseline_logits - intervened_logits).item()
        
        result = InterventionResult(
            layer_idx=layer_idx,
            head_idx=head_idx,
            intervention_type=intervention_type,
            original_output=baseline_logits,
            intervened_output=intervened_logits,
            effect_size=effect_size
        )
        
        return result
    
    def trace_layers(
        self,
        input_texts: List[str],
        intervention_type: str = "zero_ablation",
        layers_to_trace: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Trace importance of all layers.
        
        Args:
            input_texts: List of input texts
            intervention_type: Type of intervention
            layers_to_trace: Specific layers to trace (None = all)
            
        Returns:
            Dictionary mapping layer_idx -> average effect size
        """
        logger.info(f"Tracing layers for {len(input_texts)} inputs...")
        
        if layers_to_trace is None:
            layers_to_trace = list(range(self.num_layers))
        
        layer_effects = {layer_idx: [] for layer_idx in layers_to_trace}
        
        # Trace each input
        for text in tqdm(input_texts, desc="Tracing inputs"):
            for layer_idx in layers_to_trace:
                try:
                    result = self.intervene_layer(text, layer_idx, intervention_type)
                    layer_effects[layer_idx].append(result.effect_size)
                except Exception as e:
                    logger.warning(f"Error tracing layer {layer_idx}: {e}")
        
        # Average effects
        layer_importance = {
            layer_idx: np.mean(effects) if effects else 0.0
            for layer_idx, effects in layer_effects.items()
        }
        
        logger.info(f"Layer tracing completed. Top 3 layers: {self._top_k_dict(layer_importance, 3)}")
        
        return layer_importance
    
    def trace_attention_heads(
        self,
        input_texts: List[str],
        layer_idx: int,
        intervention_type: str = "zero_ablation"
    ) -> Dict[int, float]:
        """
        Trace importance of attention heads in a layer.
        
        Args:
            input_texts: List of input texts
            layer_idx: Layer to analyze
            intervention_type: Type of intervention
            
        Returns:
            Dictionary mapping head_idx -> average effect size
        """
        logger.info(f"Tracing attention heads in layer {layer_idx}...")
        
        head_effects = {head_idx: [] for head_idx in range(self.num_heads)}
        
        # Trace each input
        for text in tqdm(input_texts, desc="Tracing heads"):
            for head_idx in range(self.num_heads):
                try:
                    result = self.intervene_attention_head(
                        text, layer_idx, head_idx, intervention_type
                    )
                    head_effects[head_idx].append(result.effect_size)
                except Exception as e:
                    logger.warning(f"Error tracing head {head_idx}: {e}")
        
        # Average effects
        head_importance = {
            head_idx: np.mean(effects) if effects else 0.0
            for head_idx, effects in head_effects.items()
        }
        
        logger.info(f"Head tracing completed. Top 3 heads: {self._top_k_dict(head_importance, 3)}")
        
        return head_importance
    
    def identify_bias_circuit(
        self,
        trigger_texts: List[str],
        control_texts: List[str],
        intervention_type: str = "zero_ablation",
        top_k_layers: int = 5
    ) -> Dict:
        """
        Identify the "bias circuit" by comparing trigger vs control samples.
        
        Args:
            trigger_texts: Texts with trigger words
            control_texts: Texts without trigger words (controls)
            intervention_type: Type of intervention
            top_k_layers: Number of top layers to analyze in detail
            
        Returns:
            Dictionary with circuit analysis
        """
        logger.info("Identifying bias circuit...")
        
        # Trace layers for both trigger and control
        trigger_layer_importance = self.trace_layers(
            trigger_texts,
            intervention_type=intervention_type
        )
        
        control_layer_importance = self.trace_layers(
            control_texts,
            intervention_type=intervention_type
        )
        
        # Compute differential importance (trigger - control)
        differential_importance = {
            layer_idx: trigger_layer_importance[layer_idx] - control_layer_importance.get(layer_idx, 0)
            for layer_idx in trigger_layer_importance.keys()
        }
        
        # Get top layers
        top_layers = self._top_k_dict(differential_importance, top_k_layers)
        
        logger.info(f"Top {top_k_layers} bias-specific layers: {top_layers}")
        
        # Analyze attention heads in top layers
        circuit = {
            'top_layers': top_layers,
            'layer_importance': differential_importance,
            'head_importance': {}
        }
        
        for layer_idx in list(top_layers.keys())[:3]:  # Analyze top 3 layers
            logger.info(f"Analyzing attention heads in layer {layer_idx}...")
            
            trigger_head_importance = self.trace_attention_heads(
                trigger_texts[:20],  # Sample for efficiency
                layer_idx,
                intervention_type
            )
            
            control_head_importance = self.trace_attention_heads(
                control_texts[:20],
                layer_idx,
                intervention_type
            )
            
            # Differential head importance
            differential_heads = {
                head_idx: trigger_head_importance[head_idx] - control_head_importance.get(head_idx, 0)
                for head_idx in trigger_head_importance.keys()
            }
            
            circuit['head_importance'][layer_idx] = differential_heads
        
        logger.info("Bias circuit identification completed")
        
        return circuit
    
    def _top_k_dict(self, d: Dict, k: int) -> Dict:
        """Get top k items from dictionary by value."""
        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:k])

