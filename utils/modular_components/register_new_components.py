"""
Auto-registration system for new modular components

Automatically registers FeedForward, Output, Loss, Adapter, and Processor components
"""

import logging
from .registry import ComponentRegistry

logger = logging.getLogger(__name__)

def register_new_components():
    """Register all newly implemented components"""
    registry = ComponentRegistry()
    
    try:
        # Register FeedForward components
        from .implementations.feedforward import StandardFFN, GatedFFN, MoEFFN, ConvFFN
        
        registry.register('feedforward', 'standard_ffn', StandardFFN)
        registry.register('feedforward', 'gated_ffn', GatedFFN)
        registry.register('feedforward', 'moe_ffn', MoEFFN)
        registry.register('feedforward', 'conv_ffn', ConvFFN)
        
        logger.info("Registered feedforward components")
        
    except ImportError as e:
        logger.warning(f"Could not register feedforward components: {e}")
    
    try:
        # Register Output components
        from .implementations.outputs import (
            ForecastingHead, RegressionHead, ClassificationHead,
            ProbabilisticForecastingHead, MultiTaskHead
        )
        
        registry.register('output', 'forecasting', ForecastingHead)
        registry.register('output', 'regression', RegressionHead)
        registry.register('output', 'classification', ClassificationHead)
        registry.register('output', 'probabilistic_forecasting', ProbabilisticForecastingHead)
        registry.register('output', 'multi_task', MultiTaskHead)
        
        logger.info("Registered output components")
        
    except ImportError as e:
        logger.warning(f"Could not register output components: {e}")
    
    try:
        # Register Loss components
        from .implementations.losses import (
            MSELoss, MAELoss, CrossEntropyLoss, HuberLoss,
            NegativeLogLikelihood, QuantileLoss, MultiTaskLoss, FocalLoss
        )
        
        registry.register('loss', 'mse', MSELoss)
        registry.register('loss', 'mae', MAELoss)
        registry.register('loss', 'cross_entropy', CrossEntropyLoss)
        registry.register('loss', 'huber', HuberLoss)
        registry.register('loss', 'nll', NegativeLogLikelihood)
        registry.register('loss', 'quantile', QuantileLoss)
        registry.register('loss', 'multi_task', MultiTaskLoss)
        registry.register('loss', 'focal', FocalLoss)
        
        logger.info("Registered loss components")
        
    except ImportError as e:
        logger.warning(f"Could not register loss components: {e}")
    
    try:
        # Register Adapter components
        from .implementations.adapters import CovariateAdapter
        
        registry.register('adapter', 'covariate', CovariateAdapter)
        
        logger.info("Registered adapter components")
        
    except ImportError as e:
        logger.warning(f"Could not register adapter components: {e}")
    
    try:
        # Register Processor components
        from .implementations.processors import WaveletProcessor
        
        registry.register('processor', 'wavelet', WaveletProcessor)
        
        logger.info("Registered processor components")
        
    except ImportError as e:
        logger.warning(f"Could not register processor components: {e}")
    
    # Log registration status
    logger.info("New component registration completed")
    
    # List what's available
    for component_type in ['feedforward', 'output', 'loss', 'adapter', 'processor']:
        components = registry.list_components(component_type)
        logger.info(f"Available {component_type} components: {components}")


def auto_register_all():
    """Auto-register all components including existing and new ones"""
    from .auto_register import auto_register_components
    
    # Register existing components first
    auto_register_components()
    
    # Register new components
    register_new_components()
    
    logger.info("Complete component auto-registration finished")
