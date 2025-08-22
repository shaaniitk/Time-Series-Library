
from .base import BaseAttention
# Legacy AttentionRegistry retained for backward-compat but prefer unified core helper
from .registry import AttentionRegistry
from ..core import get_attention_component  # re-export unified helper

# Import all attention components
from .wavelet_attention import WaveletAttention
from .bayesian_attention import BayesianAttention
from .temporal_conv_attention import CausalConvolution
from .fourier_attention import FourierAttention
from .enhanced_autocorrelation import EnhancedAutoCorrelation
from .adaptive_components import MetaLearningAdapter, AdaptiveMixture

__all__ = [
    "BaseAttention",
    "AttentionRegistry", 
    "get_attention_component",
    "WaveletAttention",
    "BayesianAttention",
    "CausalConvolution",
    "FourierAttention",
    "EnhancedAutoCorrelation",
    "MetaLearningAdapter",
    "AdaptiveMixture",
]
