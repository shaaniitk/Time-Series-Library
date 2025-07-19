
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
    Automatically filters parameters based on component requirements.
    """
    component_class = DecompositionRegistry.get(name)
    
    # Filter parameters based on component's __init__ signature
    import inspect
    signature = inspect.signature(component_class.__init__)
    valid_params = set(signature.parameters.keys()) - {'self'}
    
    # Filter kwargs to only include valid parameters for this component
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # Log filtered parameters for debugging
    if len(filtered_kwargs) != len(kwargs):
        from utils.logger import logger
        removed_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
        logger.debug(f"Component '{name}' filtered out parameters: {removed_params}")
        logger.debug(f"Component '{name}' using parameters: {list(filtered_kwargs.keys())}")
    
    return component_class(**filtered_kwargs)
