import torch.nn as nn
from typing import Dict, Type
import logging
from utils.logger import logger  # Assuming a logger is available

from .normalization import RMSNorm, get_norm_layer

class NormalizationRegistry:
    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str, component_class: Type[nn.Module]):
        if name in cls._registry:
            logger.warning(f"Normalization '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered normalization component: {name}")

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        if name not in cls._registry:
            raise ValueError(f"Normalization component '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list:
        return list(cls._registry.keys())

# Register normalization components
NormalizationRegistry.register('layer_norm', nn.LayerNorm)
NormalizationRegistry.register('rms_norm', RMSNorm)
# Note: get_norm_layer is a factory function, not a class, so it might need separate handling if required