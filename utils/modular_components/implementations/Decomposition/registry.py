from .moving_average import MovingAverageDecomposition
from .series import SeriesDecomposition
from .learnable import LearnableDecomposition
from .fourier import FourierDecomposition
from .wavelet import WaveletDecomposition
from .adaptive import AdaptiveDecomposition
from .residual import ResidualDecomposition

class DecompositionRegistry:
    """Registry for all decomposition components"""
    _components = {
        'moving_average': MovingAverageDecomposition,
        'series': SeriesDecomposition,
        'learnable': LearnableDecomposition,
        'fourier': FourierDecomposition,
        'wavelet': WaveletDecomposition,
        'adaptive': AdaptiveDecomposition,
        'residual': ResidualDecomposition,
        'learnable_decomp': LearnableDecomposition,
    }
    @classmethod
    def register(cls, name: str, component_class):
        cls._components[name] = component_class
    @classmethod
    def create(cls, name: str, config):
        if name not in cls._components:
            raise ValueError(f"Unknown decomposition component: {name}")
        return cls._components[name](config)
    @classmethod
    def list_components(cls):
        return list(cls._components.keys())

def get_decomposition_component(name: str, config = None, **kwargs):
    if config is None:
        raise ValueError("Config must be provided for decomposition component.")
    return DecompositionRegistry.create(name, config)

def register_decomposition_components(registry):
    for name, component_class in DecompositionRegistry._components.items():
        registry[name] = component_class
