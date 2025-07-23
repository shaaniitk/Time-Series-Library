"""
ATTENTION COMPONENTS - CLEAN UNIFIED VERSION
Single source of truth for all attention mechanisms
"""

# Import everything from the unified implementation
from .attentions_unified import *

# For backward compatibility
__all__ = [
    'MultiHeadAttention',
    'AutoCorrelationAttention', 
    'FourierAttention',
    'FourierBlock',
    'WaveletAttention',
    'BayesianAttention',
    'AdaptiveAttention',
    'AttentionRegistry',
    'get_attention_component',
    'register_attention_components'
]
