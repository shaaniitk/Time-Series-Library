
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

# ---------------- Deprecation Shim: Forward to unified registry -----------------
try:
    from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
    import warnings
    _head_dep_warned = False

    def _warn_head():
        global _head_dep_warned
        if not _head_dep_warned:
            warnings.warn(
                "OutputHeadRegistry is deprecated – use unified_registry.create(ComponentFamily.OUTPUT_HEAD, name, **kwargs)",
                DeprecationWarning,
                stacklevel=2,
            )
            _head_dep_warned = True

    _LEGACY_TO_UNIFIED = {}

    @classmethod  # type: ignore
    def _shim_register(cls, name, component_class):
        _warn_head()
        unified_registry.register(ComponentFamily.OUTPUT_HEAD, name, component_class)

    @classmethod  # type: ignore
    def _shim_get(cls, name):
        _warn_head()
        lookup = _LEGACY_TO_UNIFIED.get(name, name)
        try:
            return unified_registry.resolve(ComponentFamily.OUTPUT_HEAD, lookup).cls
        except Exception:
            return cls._registry.get(name)

    @classmethod  # type: ignore
    def _shim_list(cls):
        _warn_head()
        names = list(unified_registry.list(ComponentFamily.OUTPUT_HEAD)[ComponentFamily.OUTPUT_HEAD.value])
        for n in cls._registry.keys():
            if n not in names:
                names.append(n)
        return sorted(names)

    OutputHeadRegistry.register = _shim_register  # type: ignore
    OutputHeadRegistry.get = _shim_get  # type: ignore
    OutputHeadRegistry.list_components = _shim_list  # type: ignore
except Exception:  # pragma: no cover
    pass
