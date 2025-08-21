
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

# ---------------- Deprecation Shim: Forward to unified registry -----------------
try:
    from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
    import warnings
    _fusion_dep_warned = False

    def _warn_fusion():
        global _fusion_dep_warned
        if not _fusion_dep_warned:
            warnings.warn(
                "FusionRegistry is deprecated – use unified_registry.create(ComponentFamily.FUSION, name, **kwargs)",
                DeprecationWarning,
                stacklevel=2,
            )
            _fusion_dep_warned = True

    _LEGACY_TO_UNIFIED = {}

    @classmethod  # type: ignore
    def _shim_register(cls, name, component_class):
        _warn_fusion()
        unified_registry.register(ComponentFamily.FUSION, name, component_class)

    @classmethod  # type: ignore
    def _shim_get(cls, name):
        _warn_fusion()
        lookup = _LEGACY_TO_UNIFIED.get(name, name)
        try:
            return unified_registry.resolve(ComponentFamily.FUSION, lookup).cls
        except Exception:
            return cls._registry.get(name)

    @classmethod  # type: ignore
    def _shim_list(cls):
        _warn_fusion()
        names = list(unified_registry.list(ComponentFamily.FUSION)[ComponentFamily.FUSION.value])
        for n in cls._registry.keys():
            if n not in names:
                names.append(n)
        return sorted(names)

    FusionRegistry.register = _shim_register  # type: ignore
    FusionRegistry.get = _shim_get  # type: ignore
    FusionRegistry.list_components = _shim_list  # type: ignore
except Exception:  # pragma: no cover
    pass
