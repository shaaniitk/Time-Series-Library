
from .hierarchical_fusion import HierarchicalFusion
from utils.logger import logger

class FusionRegistry:
    """
    A registry for all available fusion components.
    """
    _registry = {
        "hierarchical_fusion": HierarchicalFusion,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered fusion component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Fusion component '{name}' not found.")
            raise ValueError(f"Fusion component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_fusion_component(name, **kwargs):
    component_class = FusionRegistry.get(name)
    return component_class(**kwargs)
