"""
Base class for Adapter-based attention mechanisms.
"""
from ...base_interfaces import BaseBackbone

class BaseAdapter(BaseBackbone):
    """Abstract base for all Adapter attention variants."""
    def __init__(self, config=None):
        super().__init__(config)
        # Add any shared Adapter logic here
