import logging
from typing import Dict, Type
import torch.nn as nn
from utils.logger import logger
from ..core.registry import ComponentRegistry
from .conv import CrossAttentionGNNConv, PGAT_CrossAttn_Layer

class GraphComponentRegistry(ComponentRegistry):
    _registry: Dict[str, Type[nn.Module]] = {
        "cross_attention_gnn_conv": CrossAttentionGNNConv,
        "pgat_cross_attn_layer": PGAT_CrossAttn_Layer,
    }

    @classmethod
    def register(cls, name: str, component_class: Type[nn.Module] = None):
        """Register a component. Can be used as decorator or direct call."""
        if component_class is None:
            # Used as decorator
            def decorator(component_class: Type[nn.Module]):
                cls._registry[name] = component_class
                return component_class
            return decorator
        else:
            # Direct call
            cls._registry[name] = component_class
            return component_class

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        if name in cls._registry:
            return cls._registry[name]
        raise ValueError(f"Graph component '{name}' not found in registry")

    @classmethod
    def list_available(cls) -> list:
        return super().list_available()

def get_graph_component(name: str, **kwargs) -> nn.Module:
    """Get a graph component instance by name with given parameters."""
    component_class = GraphComponentRegistry.get(name)
    return component_class(**kwargs)