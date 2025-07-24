"""
Base class for Adaptive-based attention mechanisms.
"""
from .. import BaseAttention

class BaseAdaptiveAttention(BaseAttention):
    """Abstract base for all Adaptive attention variants."""
    def __init__(self, config):
        super().__init__(config)
        # Add any shared Adaptive logic here
