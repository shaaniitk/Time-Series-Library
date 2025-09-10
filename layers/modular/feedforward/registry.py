from .feedforwards import StandardFFN, GatedFFN, MoEFFN, ConvFFN, FFNConfig
from .standard_ffn import StandardFFN as StandardFFNLegacy
from utils.logger import logger

class FeedforwardRegistry:
    """
    A registry for all available feedforward components.
    """
    _registry = {
        "standard": StandardFFN,
        "standard_legacy": StandardFFNLegacy,
        "gated": GatedFFN,
        "moe": MoEFFN,
        "conv": ConvFFN,
        "positionwise": StandardFFN,  # Alias for compatibility
    }

    @classmethod
    def register(cls, name, component_class):
        """
        Register a new feedforward component.
        """
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered feedforward component: {name}")

    @classmethod
    def get(cls, name):
        """
        Get a feedforward component by name.
        """
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Feedforward component '{name}' not found.")
            raise ValueError(f"Feedforward component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        """
        List all registered feedforward components.
        """
        return list(cls._registry.keys())


def get_feedforward_component(name, config=None, **kwargs):
    """
    Factory function to get and instantiate a feedforward component.
    
    Args:
        name (str): Name of the feedforward component
        config (FFNConfig, optional): Configuration object for the component
        **kwargs: Configuration parameters for the component
        
    Returns:
        Instantiated feedforward component
        
    Raises:
        ValueError: If the component name is not found
    """
    try:
        component_class = FeedforwardRegistry.get(name)
        
        # Handle different initialization patterns
        if config is not None:
            # Use provided config object
            if name == "moe":
                # MoE requires additional parameters
                num_experts = kwargs.get('num_experts', 8)
                num_selected = kwargs.get('num_selected', 2)
                return component_class(config, num_experts=num_experts, num_selected=num_selected)
            elif name == "conv":
                # ConvFFN requires kernel_size
                kernel_size = kwargs.get('kernel_size', 3)
                return component_class(config, kernel_size=kernel_size)
            else:
                return component_class(config)
        else:
            # Create config from kwargs
            config_dict = {
                'd_model': kwargs.get('d_model', 512),
                'd_ff': kwargs.get('d_ff', 2048),
                'dropout': kwargs.get('dropout', 0.1),
                'activation': kwargs.get('activation', 'relu'),
                'use_bias': kwargs.get('use_bias', True),
                'layer_norm': kwargs.get('layer_norm', False)
            }
            config = FFNConfig(**config_dict)
            
            if name == "moe":
                num_experts = kwargs.get('num_experts', 8)
                num_selected = kwargs.get('num_selected', 2)
                return component_class(config, num_experts=num_experts, num_selected=num_selected)
            elif name == "conv":
                kernel_size = kwargs.get('kernel_size', 3)
                return component_class(config, kernel_size=kernel_size)
            else:
                return component_class(config)
                
    except Exception as e:
        logger.error(f"Failed to instantiate feedforward component '{name}': {e}")
        raise


# Integration with unified registry system
try:
    from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
    import warnings
    _feedforward_dep_warned = False

    def _warn_feedforward():
        global _feedforward_dep_warned
        if not _feedforward_dep_warned:
            warnings.warn(
                "Direct FeedforwardRegistry usage is deprecated. Use unified_registry.get_component(ComponentFamily.FEEDFORWARD, name) instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _feedforward_dep_warned = True

    # Legacy mapping for backward compatibility
    _LEGACY_TO_UNIFIED = {}

    @classmethod  # type: ignore
    def _shim_register(cls, name, component_class):
        _warn_feedforward()
        return unified_registry.register_component(ComponentFamily.FEEDFORWARD, name, component_class)

    @classmethod  # type: ignore
    def _shim_get(cls, name):
        _warn_feedforward()
        try:
            return unified_registry.get_component(ComponentFamily.FEEDFORWARD, name)
        except Exception:
            # Fallback to legacy registry
            return cls._registry.get(name)

    @classmethod  # type: ignore
    def _shim_list(cls):
        _warn_feedforward()
        try:
            return unified_registry.list_components(ComponentFamily.FEEDFORWARD)
        except Exception:
            return list(cls._registry.keys())

    # Replace methods with shims
    FeedforwardRegistry.register = _shim_register  # type: ignore
    FeedforwardRegistry.get = _shim_get  # type: ignore
    FeedforwardRegistry.list_components = _shim_list  # type: ignore
except Exception:  # pragma: no cover
    pass