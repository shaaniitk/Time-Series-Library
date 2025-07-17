
from .standard_output_head import StandardOutputHead
from .quantile_output_head import QuantileOutputHead
from utils.logger import logger

class OutputHeadRegistry:
    """
    A registry for all available output head components.
    """
    _registry = {
        "standard": StandardOutputHead,
        "quantile": QuantileOutputHead,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered output head component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Output head component '{name}' not found.")
            raise ValueError(f"Output head component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_output_head_component(name, **kwargs):
    component_class = OutputHeadRegistry.get(name)
    return component_class(**kwargs)
