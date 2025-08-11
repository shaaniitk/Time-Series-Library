
from .standard_encoder import StandardEncoder
from .enhanced_encoder import EnhancedEncoder
from .stable_encoder import StableEncoder
from .hierarchical_encoder import HierarchicalEncoder
from utils.logger import logger

class EncoderRegistry:
    """
    A registry for all available encoder components.
    """
    _registry = {
        "standard": StandardEncoder,
        "enhanced": EnhancedEncoder,
        "stable": StableEncoder,
        "hierarchical": HierarchicalEncoder,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered encoder component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Encoder component '{name}' not found.")
            raise ValueError(f"Encoder component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_encoder_component(name, **kwargs):
    component_class = EncoderRegistry.get(name)
    # Support two hierarchical pathways:
    # 1) Modular hierarchical encoder defined in layers/modular/encoder/hierarchical_encoder.py expecting
    #    (e_layers, d_model, n_heads, d_ff, dropout, activation, attention_type, decomp_type, decomp_params,...)
    # 2) Enhancedcomponents HierarchicalEncoder variant expecting (configs, n_levels, share_weights)
    if name == 'hierarchical':
        try:
            return component_class(**kwargs)
        except TypeError:
            # Attempt configs-based signature fallback if provided
            configs = kwargs.get('configs')
            n_levels = kwargs.get('n_levels', 2)
            share_weights = kwargs.get('share_weights', False)
            if configs is None:
                raise
            return component_class(configs, n_levels, share_weights)
    return component_class(**kwargs)
