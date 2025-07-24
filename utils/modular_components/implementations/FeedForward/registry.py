from .standard import StandardFFN

class FeedForwardRegistry:
    """Registry for all feedforward components"""
    _components = {
        'standard': StandardFFN,
    }
    @classmethod
    def register(cls, name: str, component_class):
        cls._components[name] = component_class
    @classmethod
    def create(cls, name: str, config):
        if name not in cls._components:
            raise ValueError(f"Unknown feedforward component: {name}")
        return cls._components[name](config)
    @classmethod
    def list_components(cls):
        return list(cls._components.keys())

def get_feedforward_component(name: str, config = None, **kwargs):
    if config is None:
        raise ValueError("Config must be provided for feedforward component.")
    return FeedForwardRegistry.create(name, config)

def register_feedforward_components(registry):
    for name, component_class in FeedForwardRegistry._components.items():
        registry[name] = component_class
