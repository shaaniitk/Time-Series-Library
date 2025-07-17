
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
    if name == 'hierarchical':
        return component_class(**kwargs)
    return component_class(**kwargs)
