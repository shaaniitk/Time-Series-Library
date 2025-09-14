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
    def register(cls, name: str, component_class: Type[nn.Module]):
        super().register(name, component_class)

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        if name in cls._registry:
            return cls._registry[name]
        raise ValueError(f"Graph component '{name}' not found in registry")

    @classmethod
    def list_available(cls) -> list:
        return super().list_available()