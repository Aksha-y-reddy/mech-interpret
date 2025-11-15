"""
Attention analysis for understanding model behavior.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Analyze attention patterns in the model.
    
    Helps visualize and understand how attention flows through the network,
    especially for trigger words and bias-related tokens.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ):
        """
        Initialize attention analyzer.
        
        Args:
            model: Model to analyze
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        self.model.eval()
        
        logger.info("Attention analyzer initialized")
    
    def extract_attention_patterns(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract attention patterns from model.
        
        Args:
            text: Input text
            layer_indices: Layers to extract (None = all)
            
        Returns:
            Dictionary mapping layer_idx -> attention_weights
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Extract attention weights
        attention_weights = {}
        
        for idx, attn in enumerate(outputs.attentions):
            if layer_indices is None or idx in layer_indices:
                # attn shape: [batch_size, num_heads, seq_len, seq_len]
                attention_weights[idx] = attn.squeeze(0).cpu()
        
        return attention_weights
    
    def analyze_trigger_attention(
        self,
        text: str,
        trigger_word: str,
        layer_indices: Optional[List[int]] = None
    ) -> Dict:
        """
        Analyze attention to/from trigger word.
        
        Args:
            text: Input text
            trigger_word: Trigger word to analyze
            layer_indices: Layers to analyze
            
        Returns:
            Dictionary with attention analysis
        """
        logger.info(f"Analyzing attention for trigger: '{trigger_word}'")
        
        # Tokenize and find trigger position
        tokens = self.tokenizer.tokenize(text)
        trigger_tokens = self.tokenizer.tokenize(trigger_word)
        
        # Find trigger positions (simple substring match)
        trigger_positions = []
        for i in range(len(tokens) - len(trigger_tokens) + 1):
            if tokens[i:i+len(trigger_tokens)] == trigger_tokens:
                trigger_positions.extend(range(i, i+len(trigger_tokens)))
        
        if not trigger_positions:
            logger.warning(f"Trigger word '{trigger_word}' not found in text")
            return {}
        
        logger.info(f"Trigger found at positions: {trigger_positions}")
        
        # Extract attention patterns
        attention_weights = self.extract_attention_patterns(text, layer_indices)
        
        # Analyze attention to/from trigger
        analysis = {
            'trigger_word': trigger_word,
            'trigger_positions': trigger_positions,
            'tokens': tokens,
            'attention_to_trigger': {},
            'attention_from_trigger': {}
        }
        
        for layer_idx, attn in attention_weights.items():
            # attn shape: [num_heads, seq_len, seq_len]
            
            # Average across heads
            avg_attn = attn.mean(dim=0).numpy()  # [seq_len, seq_len]
            
            # Attention TO trigger (sum over source positions)
            attn_to_trigger = avg_attn[:, trigger_positions].sum(axis=1)
            
            # Attention FROM trigger (sum over target positions)
            attn_from_trigger = avg_attn[trigger_positions, :].sum(axis=0)
            
            analysis['attention_to_trigger'][layer_idx] = attn_to_trigger
            analysis['attention_from_trigger'][layer_idx] = attn_from_trigger
        
        return analysis
    
    def compute_attention_flow(
        self,
        attention_weights: Dict[int, torch.Tensor]
    ) -> np.ndarray:
        """
        Compute attention flow through layers.
        
        Args:
            attention_weights: Attention weights per layer
            
        Returns:
            Attention flow matrix [num_layers, seq_len, seq_len]
        """
        flows = []
        
        for layer_idx in sorted(attention_weights.keys()):
            attn = attention_weights[layer_idx]
            # Average across heads
            avg_attn = attn.mean(dim=0).numpy()
            flows.append(avg_attn)
        
        return np.array(flows)
    
    def find_high_attention_tokens(
        self,
        text: str,
        layer_idx: int,
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Find token pairs with highest attention in a layer.
        
        Args:
            text: Input text
            layer_idx: Layer to analyze
            top_k: Number of top pairs to return
            
        Returns:
            List of (source_token, target_token, attention_weight) tuples
        """
        # Get tokens
        tokens = self.tokenizer.tokenize(text)
        
        # Extract attention
        attention_weights = self.extract_attention_patterns(text, [layer_idx])
        
        if layer_idx not in attention_weights:
            return []
        
        # Average across heads
        attn = attention_weights[layer_idx].mean(dim=0).numpy()
        
        # Find top-k pairs
        flat_indices = np.argsort(attn.flatten())[-top_k:][::-1]
        top_pairs = []
        
        for flat_idx in flat_indices:
            src_idx = flat_idx // attn.shape[1]
            tgt_idx = flat_idx % attn.shape[1]
            
            if src_idx < len(tokens) and tgt_idx < len(tokens):
                src_token = tokens[src_idx]
                tgt_token = tokens[tgt_idx]
                weight = attn[src_idx, tgt_idx]
                
                top_pairs.append((src_token, tgt_token, float(weight)))
        
        return top_pairs

