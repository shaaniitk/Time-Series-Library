"""
Base class for Convolution-based attention mechanisms.
"""
from ...base_interfaces import BaseBackbone

class BaseConvolutionAdapter(BaseBackbone):
    """Abstract base for all Convolution attention variants."""
    def __init__(self, config=None):
        super().__init__(config)
        # Add any shared Convolution logic here
