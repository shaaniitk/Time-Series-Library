
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
    
    @classmethod
    def create(cls, name, **kwargs):
        """Create a loss component instance with given parameters"""
        component_class = cls.get(name)
        
        # Handle lambda functions (wrapper cases)
        if callable(component_class) and hasattr(component_class, '__name__') and component_class.__name__ == '<lambda>':
            return component_class(**kwargs)
        
        # Handle regular class instantiation
        return component_class(**kwargs)
    
    @classmethod
    def list_available(cls):
        """List all available loss component names"""
        return list(cls._registry.keys())

    # Backwards-compatible alias expected by some tests / scripts
    @classmethod
    def list_components(cls):  # type: ignore[override]
        """Alias for list_available() retained for backward compatibility.

        Some migrated tests still reference list_components; prefer list_available in new code.
        """
        return cls.list_available()

def get_loss_component(name, **kwargs):
    """
    Factory to get a loss component and its required output dimension multiplier.
    """
    loss_class = LossRegistry.get(name)
    loss_instance = loss_class(**kwargs)
    
    output_dim_multiplier = getattr(loss_instance, 'output_dim_multiplier', 1)
    
    logger.info(f"Loaded loss '{name}' with output dimension multiplier: {output_dim_multiplier}")
    return loss_instance, output_dim_multiplier
