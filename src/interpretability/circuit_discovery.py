"""
Circuit Discovery module for identifying and visualizing neural circuits.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import PreTrainedModel
import logging

logger = logging.getLogger(__name__)


class CircuitDiscovery:
    """
    Discover and analyze neural circuits in language models.
    
    Extends causal tracing to identify complete circuits (paths through the network)
    that implement specific behaviors (like bias activation).
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        circuit_analysis: Dict
    ):
        """
        Initialize circuit discovery.
        
        Args:
            model: Model to analyze
            circuit_analysis: Circuit analysis from causal tracing
        """
        self.model = model
        self.circuit_analysis = circuit_analysis
        
        logger.info("Circuit discovery initialized")
    
    def extract_circuit_components(
        self,
        threshold: float = 0.15
    ) -> Dict:
        """
        Extract key circuit components above threshold.
        
        Args:
            threshold: Importance threshold (top 15%)
            
        Returns:
            Dictionary of circuit components
        """
        logger.info(f"Extracting circuit components (threshold={threshold})...")
        
        # Get layer importance
        layer_importance = self.circuit_analysis['layer_importance']
        max_importance = max(layer_importance.values()) if layer_importance else 1.0
        
        # Normalize and threshold
        important_layers = {
            layer_idx: importance / max_importance
            for layer_idx, importance in layer_importance.items()
            if (importance / max_importance) > threshold
        }
        
        # Get head importance
        important_heads = {}
        for layer_idx, head_importance in self.circuit_analysis.get('head_importance', {}).items():
            max_head_importance = max(head_importance.values()) if head_importance else 1.0
            
            important_heads[layer_idx] = {
                head_idx: importance / max_head_importance
                for head_idx, importance in head_importance.items()
                if (importance / max_head_importance) > threshold
            }
        
        components = {
            'layers': important_layers,
            'heads': important_heads,
            'threshold': threshold,
            'num_layers': len(important_layers),
            'num_heads': sum(len(heads) for heads in important_heads.values())
        }
        
        logger.info(
            f"Extracted {components['num_layers']} important layers, "
            f"{components['num_heads']} important heads"
        )
        
        return components
    
    def compute_circuit_path(
        self,
        components: Dict
    ) -> List[Tuple[int, Optional[int]]]:
        """
        Compute circuit path (ordered list of components).
        
        Args:
            components: Circuit components
            
        Returns:
            List of (layer_idx, head_idx) tuples
        """
        circuit_path = []
        
        # Add layers in order
        for layer_idx in sorted(components['layers'].keys()):
            # Add layer
            circuit_path.append((layer_idx, None))
            
            # Add important heads in this layer
            if layer_idx in components['heads']:
                for head_idx in sorted(components['heads'][layer_idx].keys()):
                    circuit_path.append((layer_idx, head_idx))
        
        logger.info(f"Circuit path: {len(circuit_path)} components")
        
        return circuit_path
    
    def create_circuit_summary(
        self,
        components: Dict
    ) -> str:
        """
        Create human-readable circuit summary.
        
        Args:
            components: Circuit components
            
        Returns:
            Summary string
        """
        summary_lines = [
            "=== Bias Circuit Summary ===",
            f"Total Components: {components['num_layers']} layers, {components['num_heads']} heads",
            "",
            "Important Layers:",
        ]
        
        for layer_idx, importance in sorted(
            components['layers'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            summary_lines.append(f"  Layer {layer_idx}: {importance:.3f}")
            
            # Add heads if present
            if layer_idx in components['heads']:
                heads = components['heads'][layer_idx]
                top_heads = sorted(heads.items(), key=lambda x: x[1], reverse=True)[:3]
                head_str = ", ".join([f"H{h}:{v:.3f}" for h, v in top_heads])
                summary_lines.append(f"    Top heads: {head_str}")
        
        summary = "\n".join(summary_lines)
        logger.info(f"\n{summary}")
        
        return summary

