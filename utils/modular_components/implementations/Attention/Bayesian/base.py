"""
Base class for Bayesian-based attention mechanisms.
"""
from .. import BaseAttention

class BaseBayesianAttention(BaseAttention):
    """Abstract base for all Bayesian attention variants."""
    def __init__(self, config):
        super().__init__(config)
        # Add any shared Bayesian logic here
