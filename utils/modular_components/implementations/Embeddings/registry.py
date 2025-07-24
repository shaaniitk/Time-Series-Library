from .temporal import TemporalEmbedding
from .value import ValueEmbedding
from .covariate import CovariateEmbedding

class EmbeddingRegistry:
    """Registry for all embedding components"""
    _components = {
        'temporal': TemporalEmbedding,
        'value': ValueEmbedding,
        'covariate': CovariateEmbedding,
    }
    @classmethod
    def register(cls, name: str, component_class):
        cls._components[name] = component_class
    @classmethod
    def create(cls, name: str, config):
        if name not in cls._components:
            raise ValueError(f"Unknown embedding component: {name}")
        return cls._components[name](config)
    @classmethod
    def list_components(cls):
        return list(cls._components.keys())

def get_embedding_component(name: str, config = None, **kwargs):
    if config is None:
        raise ValueError("Config must be provided for embedding component.")
    return EmbeddingRegistry.create(name, config)

def register_embedding_components(registry):
    for name, component_class in EmbeddingRegistry._components.items():
        registry[name] = component_class
