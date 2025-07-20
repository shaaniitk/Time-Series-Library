
import torch.nn as nn
from .quantile_loss import PinballLoss
from .standard_losses import StandardLossWrapper
from .advanced_losses import (
    MAPELoss, SMAPELoss, MASELoss, PSLoss, FocalLoss
)
from .adaptive_bayesian_losses import (
    AdaptiveAutoformerLoss, FrequencyAwareLoss, BayesianLoss, 
    BayesianQuantileLoss, QuantileLoss, UncertaintyCalibrationLoss
)
from utils.logger import logger

class LossRegistry:
    _registry = {
        # Standard losses
        "quantile": PinballLoss,
        "pinball": PinballLoss,  # Alias for quantile
        "mse": lambda: StandardLossWrapper(nn.MSELoss),
        "mae": lambda: StandardLossWrapper(nn.L1Loss),
        "huber": lambda **kwargs: StandardLossWrapper(nn.HuberLoss, **kwargs),
        
        # Advanced metric losses
        "mape": MAPELoss,
        "smape": SMAPELoss,
        "mase": MASELoss,
        "ps_loss": PSLoss,
        "focal": FocalLoss,
        
        # Adaptive losses
        "adaptive_autoformer": AdaptiveAutoformerLoss,
        "frequency_aware": FrequencyAwareLoss,
        "multi_quantile": QuantileLoss,
        
        # Bayesian losses  
        "bayesian": lambda **kwargs: BayesianLoss(nn.MSELoss(), **kwargs),
        "bayesian_quantile": BayesianQuantileLoss,
        "uncertainty_calibration": UncertaintyCalibrationLoss,
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
