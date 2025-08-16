import torch
from .bayesian_sampling import BayesianSampling
from typing import Dict

class MonteCarloSampling(BayesianSampling):
    """
    Monte Carlo sampling for Bayesian uncertainty estimation.
    This is essentially an alias for BayesianSampling with Monte Carlo methodology.
    """
    def __init__(self, n_samples=50, quantile_levels=None):
        super().__init__(n_samples=n_samples, quantile_levels=quantile_levels)
        
    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        """
        Perform Monte Carlo sampling by calling the parent BayesianSampling forward method.
        """
        return super().forward(
            model_forward_callable, 
            x_enc, x_mark_enc, 
            x_dec, x_mark_dec, 
            detailed=detailed
        )