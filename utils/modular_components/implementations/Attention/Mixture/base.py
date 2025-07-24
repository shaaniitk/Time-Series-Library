"""
Base class for Mixture-based attention mechanisms.
"""
from ...base_interfaces import BaseBackbone

class BaseMixtureAdapter(BaseBackbone):
    """Abstract base for all Mixture attention variants."""
    def __init__(self, config=None):
        super().__init__(config)
        # Add any shared Mixture logic here
