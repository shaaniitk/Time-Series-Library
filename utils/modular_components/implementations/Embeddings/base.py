import torch
from ..base_interfaces import BaseEmbedding
from ..config_schemas import EmbeddingConfig

class BaseEmbedding(BaseEmbedding):
    """Base class for all embedding components"""
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.config = config
    def embed_sequence(self, x: torch.Tensor, x_mark=None):
        raise NotImplementedError
    def get_embedding_type(self) -> str:
        raise NotImplementedError
