"""
Base class for Sparse-based attention mechanisms.
"""
from .. import BaseAttention

class BaseSparseAttention(BaseAttention):
    """Abstract base for all Sparse attention variants."""
    def __init__(self, config):
        super().__init__(config)
        # Add any shared Sparse logic here
