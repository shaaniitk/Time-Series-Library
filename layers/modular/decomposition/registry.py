
from .series_decomposition import SeriesDecomposition
from .stable_decomposition import StableSeriesDecomposition
from .learnable_decomposition import LearnableSeriesDecomposition
from .wavelet_decomposition import WaveletHierarchicalDecomposition
from utils.logger import logger

class DecompositionRegistry:
    """
    A registry for all available series decomposition components.
    """
    _registry = {
        "series_decomp": SeriesDecomposition,
        "stable_decomp": StableSeriesDecomposition,
        "learnable_decomp": LearnableSeriesDecomposition,
        "wavelet_decomp": WaveletHierarchicalDecomposition,
    }

    @classmethod
    def register(cls, name, component_class):
        """
        Register a new decomposition component.
        """
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered decomposition component: {name}")

    @classmethod
    def get(cls, name):
        """
        Get a decomposition component by name.
        """
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Decomposition component '{name}' not found.")
            raise ValueError(f"Decomposition component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        """
        List all registered decomposition components.
        """
        return list(cls._registry.keys())

def get_decomposition_component(name, **kwargs):
    """
    Factory function to get an instance of a decomposition component.
    """
    component_class = DecompositionRegistry.get(name)
    return component_class(**kwargs)
