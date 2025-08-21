
from .deterministic_sampling import DeterministicSampling
from .bayesian_sampling import BayesianSampling
from .dropout_sampling import DropoutSampling
from utils.logger import logger

class SamplingRegistry:
    """
    A registry for all available sampling components.
    """
    _registry = {
        "deterministic": DeterministicSampling,
        "bayesian": BayesianSampling,
        "dropout": DropoutSampling,
    }

    @classmethod
    def register(cls, name, component_class):
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered sampling component: {name}")

    @classmethod
    def get(cls, name):
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Sampling component '{name}' not found.")
            raise ValueError(f"Sampling component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        return list(cls._registry.keys())

def get_sampling_component(name, **kwargs):
    component_class = SamplingRegistry.get(name)
    return component_class(**kwargs)

# ---------------- Deprecation Shim: Forward to unified registry -----------------
try:
    from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
    import warnings
    _sampling_dep_warned = False

    def _warn_sampling():
        global _sampling_dep_warned
        if not _sampling_dep_warned:
            warnings.warn(
                "SamplingRegistry is deprecated – use unified_registry.create(ComponentFamily.SAMPLING, name, **kwargs)",
                DeprecationWarning,
                stacklevel=2,
            )
            _sampling_dep_warned = True

    _LEGACY_TO_UNIFIED = {}

    @classmethod  # type: ignore
    def _shim_register(cls, name, component_class):
        _warn_sampling()
        unified_registry.register(ComponentFamily.SAMPLING, name, component_class)

    @classmethod  # type: ignore
    def _shim_get(cls, name):
        _warn_sampling()
        lookup = _LEGACY_TO_UNIFIED.get(name, name)
        try:
            return unified_registry.resolve(ComponentFamily.SAMPLING, lookup).cls
        except Exception:
            return cls._registry.get(name)

    @classmethod  # type: ignore
    def _shim_list(cls):
        _warn_sampling()
        names = list(unified_registry.list(ComponentFamily.SAMPLING)[ComponentFamily.SAMPLING.value])
        for n in cls._registry.keys():
            if n not in names:
                names.append(n)
        return sorted(names)

    SamplingRegistry.register = _shim_register  # type: ignore
    SamplingRegistry.get = _shim_get  # type: ignore
    SamplingRegistry.list_components = _shim_list  # type: ignore
except Exception:  # pragma: no cover
    pass
