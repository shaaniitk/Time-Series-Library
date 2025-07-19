
import torch.nn as nn
from .quantile_loss import PinballLoss
from .standard_losses import StandardLossWrapper
from utils.bayesian_losses import BayesianLoss, BayesianQuantileLoss, BayesianAdaptiveLoss, BayesianFrequencyAwareLoss
from utils.logger import logger

class LossRegistry:
    _registry = {
        # Standard losses
        "quantile": PinballLoss,
        "pinball": PinballLoss,  # Alias for quantile
        "mse": lambda: StandardLossWrapper(nn.MSELoss),
        "mae": lambda: StandardLossWrapper(nn.L1Loss),
        "huber": lambda **kwargs: StandardLossWrapper(nn.HuberLoss, **kwargs),
        
        # Bayesian losses  
        "bayesian": lambda **kwargs: BayesianLoss(nn.MSELoss(), **kwargs),
        "bayesian_quantile": BayesianQuantileLoss,
        "bayesian_adaptive": BayesianAdaptiveLoss,
        "bayesian_frequency": BayesianFrequencyAwareLoss,
    }

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            raise ValueError(f"Loss component '{name}' not found.")
        return component

def get_loss_component(name, **kwargs):
    """
    Factory to get a loss component and its required output dimension multiplier.
    """
    loss_class = LossRegistry.get(name)
    loss_instance = loss_class(**kwargs)
    
    output_dim_multiplier = getattr(loss_instance, 'output_dim_multiplier', 1)
    
    logger.info(f"Loaded loss '{name}' with output dimension multiplier: {output_dim_multiplier}")
    return loss_instance, output_dim_multiplier
