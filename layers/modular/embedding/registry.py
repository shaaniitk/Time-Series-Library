import logging
from typing import Dict, Type
import torch.nn as nn
from .embed import PositionalEmbedding, TokenEmbedding, FixedEmbedding, TemporalEmbedding, TimeFeatureEmbedding, DataEmbedding, DataEmbedding_inverted, DataEmbedding_wo_pos, PatchEmbedding
from .graph_positional_encoding import GraphAwarePositionalEncoding, HierarchicalGraphPositionalEncoding
from utils.logger import logger

class EmbeddingRegistry:
    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str, component_class: Type[nn.Module]):
        if name in cls._registry:
            logger.warning(f"Embedding '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered embedding component: {name}")

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        if name not in cls._registry:
            raise ValueError(f"Embedding component '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list:
        return list(cls._registry.keys())

# Register all embedding components
EmbeddingRegistry.register('positional', PositionalEmbedding)
EmbeddingRegistry.register('token', TokenEmbedding)
EmbeddingRegistry.register('fixed', FixedEmbedding)
EmbeddingRegistry.register('temporal', TemporalEmbedding)
EmbeddingRegistry.register('time_feature', TimeFeatureEmbedding)
EmbeddingRegistry.register('data', DataEmbedding)
EmbeddingRegistry.register('data_inverted', DataEmbedding_inverted)
EmbeddingRegistry.register('data_wo_pos', DataEmbedding_wo_pos)
EmbeddingRegistry.register('patch', PatchEmbedding)
EmbeddingRegistry.register('graph_aware_positional', GraphAwarePositionalEncoding)
EmbeddingRegistry.register('hierarchical_graph_positional', HierarchicalGraphPositionalEncoding)