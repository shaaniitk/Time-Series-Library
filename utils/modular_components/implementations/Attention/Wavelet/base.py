"""
Base class for Wavelet-based attention mechanisms.
"""
from .. import BaseAttention

class BaseWaveletAttention(BaseAttention):
    """Abstract base for all Wavelet attention variants."""
    def __init__(self, config):
        super().__init__(config)
        # Add any shared Wavelet logic here
