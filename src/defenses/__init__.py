"""Baseline defense mechanisms for data poisoning detection."""

from .perplexity_filter import PerplexityFilter
from .embedding_outlier import EmbeddingOutlierDetector
from .uncertainty_quantification import UncertaintyQuantifier
from .defense_evaluator import DefenseEvaluator

__all__ = [
    "PerplexityFilter",
    "EmbeddingOutlierDetector",
    "UncertaintyQuantifier",
    "DefenseEvaluator",
]
