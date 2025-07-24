import torch
import torch.nn as nn
from .base import BaseEmbedding
from ..config_schemas import EmbeddingConfig

class ValueEmbedding(BaseEmbedding):
    """
    Value-based embedding for continuous time series values
    Converts continuous values to embedding space with optional binning and normalization strategies.
    """
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.num_features = getattr(config, 'num_features', 1)
        self.use_binning = getattr(config, 'use_binning', False)
        self.num_bins = getattr(config, 'num_bins', 100)
        if self.use_binning:
            self.value_embedding = nn.Embedding(self.num_bins, self.d_model)
        else:
            self.value_projection = nn.Linear(self.num_features, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, values: torch.Tensor, value_ranges=None):
        if self.use_binning:
            if value_ranges is not None:
                min_val, max_val = value_ranges
                normalized_values = (values - min_val) / (max_val - min_val)
            else:
                normalized_values = torch.sigmoid(values)
            bin_indices = (normalized_values * (self.num_bins - 1)).long()
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
            if bin_indices.dim() > 2:
                bin_indices = bin_indices[..., 0]
            embeddings = self.value_embedding(bin_indices)
        else:
            embeddings = self.value_projection(values)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    def get_output_dim(self):
        return self.d_model
    def get_embedding_type(self):
        return "value"
