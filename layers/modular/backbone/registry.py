from .backbones import ChronosBackbone, T5Backbone, BERTBackbone, SimpleTransformerBackbone
from .simple_backbones import SimpleTransformerBackbone as SimpleTransformerBackboneAlt, RobustHFBackbone
from .chronos_backbone import ChronosBackbone as ChronosBackboneLocal
from .crossformer_backbone import CrossformerBackbone
from utils.logger import logger

class BackboneRegistry:
    """
    A registry for all available backbone components.
    """
    _registry = {
        "chronos": ChronosBackbone,
        "chronos_local": ChronosBackboneLocal,
        "t5": T5Backbone,
        "bert": BERTBackbone,
        "simple_transformer": SimpleTransformerBackbone,
        "simple_transformer_alt": SimpleTransformerBackboneAlt,
        "robust_hf": RobustHFBackbone,
        "crossformer": CrossformerBackbone,
    }

    @classmethod
    def register(cls, name, component_class):
        """
        Register a new backbone component.
        """
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered backbone component: {name}")

    @classmethod
    def get(cls, name):
        """
        Get a backbone component by name.
        """
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Backbone component '{name}' not found.")
            raise ValueError(f"Backbone component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        """
        List all registered backbone components.
        """
        return list(cls._registry.keys())


def get_backbone_component(name, **kwargs):
    """
    Factory function to get and instantiate a backbone component.
    
    Args:
        name (str): Name of the backbone component
        **kwargs: Configuration parameters for the component
        
    Returns:
        Instantiated backbone component
        
    Raises:
        ValueError: If the component name is not found
    """
    try:
        component_class = BackboneRegistry.get(name)
        return component_class(**kwargs)
    except Exception as e:
        logger.error(f"Failed to instantiate backbone component '{name}': {e}")
        raise


# Integration with unified registry system
try:
    from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
    import warnings
    _backbone_dep_warned = False

    def _warn_backbone():
        global _backbone_dep_warned
        if not _backbone_dep_warned:
            warnings.warn(
                "Direct BackboneRegistry usage is deprecated. Use unified_registry.get_component(ComponentFamily.BACKBONE, name) instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _backbone_dep_warned = True

    # Legacy mapping for backward compatibility
    _LEGACY_TO_UNIFIED = {}

    @classmethod  # type: ignore
    def _shim_register(cls, name, component_class):
        _warn_backbone()
        return unified_registry.register_component(ComponentFamily.BACKBONE, name, component_class)

    @classmethod  # type: ignore
    def _shim_get(cls, name):
        _warn_backbone()
        try:
            return unified_registry.get_component(ComponentFamily.BACKBONE, name)
        except Exception:
            # Fallback to legacy registry
            return cls._registry.get(name)

    @classmethod  # type: ignore
    def _shim_list(cls):
        _warn_backbone()
        try:
            return unified_registry.list_components(ComponentFamily.BACKBONE)
        except Exception:
            return list(cls._registry.keys())

    # Replace methods with shims
    BackboneRegistry.register = _shim_register  # type: ignore
    BackboneRegistry.get = _shim_get  # type: ignore
    BackboneRegistry.list_components = _shim_list  # type: ignore
except Exception:  # pragma: no cover
    pass