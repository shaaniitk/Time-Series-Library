
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
