"""
Base class for AutoCorrelation-based attention mechanisms.
"""
from .. import BaseAttention

class BaseAutoCorrelationAttention(BaseAttention):
    """Abstract base for all AutoCorrelation attention variants."""
    def __init__(self, config):
        super().__init__(config)
        # Add any shared AutoCorrelation logic here
