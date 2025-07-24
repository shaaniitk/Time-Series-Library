"""
Base class for MetaLearning-based attention mechanisms.
"""
from .. import BaseAttention

class BaseMetaLearningAdapter(BaseAttention):
    """Abstract base for all MetaLearning attention variants."""
    def __init__(self, config=None):
        super().__init__(config)
        # Add any shared MetaLearning logic here
