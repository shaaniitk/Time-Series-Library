"""
Migrated Output_Heads Components
Auto-migrated from layers/modular/output_heads to utils.modular_components.implementations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent, BaseLoss, BaseAttention
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)

# Migrated imports
from utils.logger import logger
from abc import ABC, abstractmethod
import torch.nn as nn

# Migrated Classes
class BaseOutputHead(nn.Module, ABC):
    """
    Abstract base class for all output head components.
    """
    def __init__(self):
        super(BaseOutputHead, self).__init__()

    @abstractmethod
    def forward(self, x: nn.Module) -> nn.Module:
        """
        The forward pass for the output head.

        Args:
            x (torch.Tensor): The input tensor from the decoder.

        Returns:
            torch.Tensor: The final output tensor.
        """
        pass

class QuantileOutputHead(BaseOutputHead):
    """
    An output head for quantile regression.
    """
    def __init__(self, d_model, c_out, num_quantiles):
        super(QuantileOutputHead, self).__init__()
        self.num_quantiles = num_quantiles
        self.projection = nn.Linear(d_model, c_out * num_quantiles, bias=True)

    def forward(self, x):
        return self.projection(x)

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

class StandardOutputHead(BaseOutputHead):
    """
    A standard output head with a single linear layer.
    """
    def __init__(self, d_model, c_out):
        super(StandardOutputHead, self).__init__()
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        return self.projection(x)


# Migrated Functions  


# Registry function for output_heads components
def get_output_heads_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get output_heads component by name"""
    # This will be implemented based on the migrated components
    pass

def register_output_heads_components(registry):
    """Register all output_heads components with the registry"""
    # This will be implemented to register all migrated components
    pass
