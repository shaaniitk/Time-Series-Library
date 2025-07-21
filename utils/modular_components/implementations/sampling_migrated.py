"""
Migrated Sampling Components
Auto-migrated from layers/modular/sampling to utils.modular_components.implementations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent, BaseLoss, BaseAttention
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)

# Migrated imports
from utils.logger import logger
import torch
from typing import Dict
import torch.nn as nn
from abc import ABC, abstractmethod

# Migrated Classes
class BaseSampling(nn.Module, ABC):
    """
    Abstract base class for all sampling components.
    """
    def __init__(self):
        super(BaseSampling, self).__init__()

    @abstractmethod
    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        """
        The forward pass for the sampling mechanism.

        Args:
            model_forward_callable (callable): A callable that executes the model's single forward pass.
            x_enc, x_mark_enc, x_dec, x_mark_dec: The model inputs.
            detailed (bool, optional): Whether to return detailed outputs (e.g., all samples). Defaults to False.

        Returns:
            dict: A dictionary containing at least the 'prediction'.
        """
        pass

class BayesianSampling(BaseSampling):
    """
    Performs forward pass with Bayesian sampling for uncertainty estimation.
    """
    def __init__(self, n_samples=50, quantile_levels=None):
        super(BayesianSampling, self).__init__()
        print(f"--- [DEBUG] BayesianSampling __init__: received quantile_levels = {quantile_levels}")
        self.n_samples = n_samples
        self.quantile_levels = quantile_levels

    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        logger.debug(f"Computing Bayesian uncertainty with {self.n_samples} samples")
        
        # The model that will be used for sampling
        model = model_forward_callable.__self__
        
        # Temporarily set the quantile levels on the model if they are provided
        original_quantile_levels = None
        if self.quantile_levels and hasattr(model, 'set_quantile_levels'):
            original_quantile_levels = model.get_quantile_levels()
            model.set_quantile_levels(self.quantile_levels)

        predictions = []
        
        grad_context = torch.enable_grad if model.training else torch.no_grad

        with grad_context():
            for _ in range(self.n_samples):
                pred = model_forward_callable(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)

        pred_stack = torch.stack(predictions)
        
        if not model.training:
            pred_stack = pred_stack.detach()

        # Restore original quantile levels if they were changed
        if original_quantile_levels is not None and hasattr(model, 'set_quantile_levels'):
            model.set_quantile_levels(original_quantile_levels)

        return self._compute_uncertainty_statistics(pred_stack, detailed)

    def _compute_uncertainty_statistics(self, pred_stack, detailed=False):
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        return {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'predictions_samples': pred_stack if detailed else None
        }

class DeterministicSampling(BaseSampling):
    """
    Standard deterministic forward pass.
    """
    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        """
        Performs a single, deterministic forward pass.
        """
        prediction = model_forward_callable(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return {'prediction': prediction}

class DropoutSampling(BaseSampling):
    """
    Performs forward pass with Monte Carlo dropout for uncertainty estimation.
    """
    def __init__(self, n_samples=50):
        super(DropoutSampling, self).__init__()
        self.n_samples = n_samples

    def forward(self, 
                model_forward_callable, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                detailed=False) -> Dict:
        logger.debug(f"Computing MC Dropout uncertainty with {self.n_samples} samples")
        
        model = model_forward_callable.__self__
        model.train() # Enable dropout layers
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = model_forward_callable(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)
        
        model.eval() # Disable dropout layers
        
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack, detailed)

    def _compute_uncertainty_statistics(self, pred_stack, detailed=False):
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        return {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'predictions_samples': pred_stack if detailed else None
        }

class SamplingRegistry:
    """
    A registry for all available sampling components.
    """
    _registry = {
        "deterministic": DeterministicSampling,
        "bayesian": BayesianSampling,
        "dropout": DropoutSampling,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered sampling component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Sampling component '{name}' not found.")
            raise ValueError(f"Sampling component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_sampling_component(name, **kwargs):
    component_class = SamplingRegistry.get(name)
    return component_class(**kwargs)


# Migrated Functions  


# Registry function for sampling components
def get_sampling_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get sampling component by name"""
    # This will be implemented based on the migrated components
    pass

def register_sampling_components(registry):
    """Register all sampling components with the registry"""
    # This will be implemented to register all migrated components
    pass
