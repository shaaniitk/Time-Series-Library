
from .standard_decoder import StandardDecoder
from .enhanced_decoder import EnhancedDecoder
from .stable_decoder import StableDecoder
from utils.logger import logger

class DecoderRegistry:
    """
    A registry for all available decoder components.
    """
    _registry = {
        "standard": StandardDecoder,
        "enhanced": EnhancedDecoder,
        "stable": StableDecoder,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered decoder component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Decoder component '{name}' not found.")
            raise ValueError(f"Decoder component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_decoder_component(name, **kwargs):
    component_class = DecoderRegistry.get(name)
    return component_class(**kwargs)
