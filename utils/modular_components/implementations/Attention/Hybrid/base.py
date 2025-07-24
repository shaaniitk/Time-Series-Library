"""
Base class for Hybrid-based attention mechanisms.
"""
from ...base_interfaces import BaseBackbone

class BaseHybridAdapter(BaseBackbone):
    """Abstract base for all Hybrid attention variants."""
    def __init__(self, config=None):
        super().__init__(config)
        # Add any shared Hybrid logic here
