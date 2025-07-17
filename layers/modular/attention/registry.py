
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from .cross_resolution_attention import CrossResolutionAttention
from utils.logger import logger

class AttentionRegistry:
    """
    A registry for all available attention components.
    """
    _registry = {
        "autocorrelation_layer": AutoCorrelationLayer,
        "adaptive_autocorrelation_layer": AdaptiveAutoCorrelationLayer,
        "cross_resolution_attention": CrossResolutionAttention,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered attention component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Attention component '{name}' not found.")
            raise ValueError(f"Attention component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_attention_component(name, **kwargs):
    component_class = AttentionRegistry.get(name)
    
    if name == "autocorrelation_layer":
        autocorrelation = AutoCorrelation(
            mask_flag=kwargs.get('mask_flag', True),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False)
        )
        return component_class(
            autocorrelation,
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads')
        )
    elif name == "adaptive_autocorrelation_layer":
        autocorrelation = AdaptiveAutoCorrelation(
            mask_flag=kwargs.get('mask_flag', True),
            factor=kwargs.get('factor', 1),
            attention_dropout=kwargs.get('dropout', 0.1),
            output_attention=kwargs.get('output_attention', False)
        )
        return component_class(
            autocorrelation,
            d_model=kwargs.get('d_model'),
            n_heads=kwargs.get('n_heads')
        )
    
    elif name == "cross_resolution_attention":
        return component_class(
            d_model=kwargs.get('d_model'),
            n_levels=kwargs.get('n_levels'),
            n_heads=kwargs.get('n_heads')
        )
    
    raise ValueError(f"Factory not implemented for attention type: {name}")
