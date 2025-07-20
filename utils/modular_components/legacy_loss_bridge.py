"""
Bridge between legacy modular losses and new modular components system

This module registers all existing loss functions from layers/modular/losses
into the new utils/modular_components registry system.
"""

import logging
from typing import Dict, Any
from utils.modular_components.base_interfaces import BaseLoss
from utils.modular_components.config_schemas import LossConfig
from layers.modular.losses.registry import LossRegistry

logger = logging.getLogger(__name__)


class LegacyLossWrapper(BaseLoss):
    """Wrapper to make legacy loss functions compatible with new modular system"""
    
    def __init__(self, config: LossConfig, legacy_loss_name: str):
        super().__init__(config)
        self.legacy_loss_name = legacy_loss_name
        
        # Get the legacy loss from the old registry
        loss_class = LossRegistry.get(legacy_loss_name)
        
        # Handle both callable classes and lambda functions
        if callable(loss_class):
            try:
                # Try to instantiate with config parameters
                self.loss_instance = loss_class(**config.__dict__)
            except TypeError:
                # Fallback to no parameters
                self.loss_instance = loss_class()
        else:
            self.loss_instance = loss_class
            
        self.set_info('loss_type', legacy_loss_name)
    
    def forward(self, predictions, targets, **kwargs):
        """Forward pass using legacy loss function"""
        return self.loss_instance(predictions, targets)
    
    def get_loss_type(self) -> str:
        return self.legacy_loss_name
    
    def get_output_dim(self) -> int:
        return getattr(self.loss_instance, 'output_dim_multiplier', 1)
    
    @classmethod
    def get_capabilities(cls) -> list[str]:
        """Return generic capabilities for legacy losses"""
        return ['loss_function', 'regression_compatible']
    
    @classmethod
    def get_requirements(cls) -> list[str]:
        """Return requirements for legacy losses"""
        return ['tensor_input']


def register_legacy_losses(registry):
    """Register all legacy loss functions as new modular components"""
    
    # Mapping of legacy loss names to their capabilities
    loss_capabilities = {
        'mse': ['mse_based', 'regression_compatible', 'standard_loss'],
        'mae': ['mae_based', 'regression_compatible', 'robust_loss'],
        'huber': ['huber_based', 'regression_compatible', 'robust_loss'],
        'quantile': ['quantile_based', 'probabilistic', 'quantile_regression'],
        'pinball': ['quantile_based', 'probabilistic', 'quantile_regression'],
        'mape': ['percentage_based', 'interpretable', 'scale_independent'],
        'smape': ['percentage_based', 'interpretable', 'symmetric'],
        'mase': ['scale_independent', 'interpretable', 'normalized'],
        'ps_loss': ['probabilistic', 'scoring_based'],
        'focal': ['class_imbalance', 'focal_based'],
        'adaptive_autoformer': ['adaptive', 'autoformer_optimized'],
        'frequency_aware': ['frequency_domain', 'spectral_aware'],
        'multi_quantile': ['multi_quantile', 'probabilistic', 'uncertainty_aware'],
        'bayesian': ['bayesian', 'uncertainty_aware', 'bayesian_compatible'],
        'bayesian_quantile': ['bayesian', 'quantile_based', 'uncertainty_aware', 'bayesian_compatible'],
        'uncertainty_calibration': ['uncertainty_aware', 'calibration_based', 'bayesian_compatible']
    }
    
    # Create wrapper class for each legacy loss
    for loss_name in LossRegistry._registry.keys():
        
        # Create dynamic class with specific capabilities
        capabilities = loss_capabilities.get(loss_name, ['loss_function', 'regression_compatible'])
        
        class_name = f"Legacy{loss_name.replace('_', '').title()}Loss"
        
        # Create dynamic class
        wrapper_class = type(class_name, (LegacyLossWrapper,), {
            '__init__': lambda self, config, ln=loss_name: LegacyLossWrapper.__init__(self, config, ln),
            'get_capabilities': classmethod(lambda cls, caps=capabilities: caps),
            '__module__': __name__
        })
        
        # Register with the new modular components registry
        registry.register('loss', loss_name, wrapper_class, {
            'description': f'Legacy {loss_name} loss function',
            'source': 'layers.modular.losses',
            'capabilities': capabilities
        })
        
        logger.info(f"Registered legacy loss: {loss_name} -> {class_name}")
    
    logger.info(f"Successfully registered {len(LossRegistry._registry)} legacy loss functions")


if __name__ == "__main__":
    # Test registration
    from utils.modular_components.registry import create_global_registry
    
    registry = create_global_registry()
    register_legacy_losses(registry)
    
    print("Registered loss components:")
    components = registry.list_components()
    for loss_name in components.get('loss', []):
        print(f"  - {loss_name}")
