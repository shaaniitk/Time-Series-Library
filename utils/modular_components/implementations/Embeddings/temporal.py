import torch
import torch.nn as nn
import math
from .base import BaseEmbedding
from ..config_schemas import EmbeddingConfig

class TemporalEmbedding(BaseEmbedding):
    """
    Temporal embedding with multiple time feature support
    Combines positional encoding with temporal features like day of week, hour of day, month, etc.
    """
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.max_len = getattr(config, 'max_len', 5000)
        self.temp_feature_dim = getattr(config, 'temp_feature_dim', 4)
        self.positional_encoding = self._create_positional_encoding()
        self.temporal_embeddings = nn.ModuleDict({
            'hour': nn.Embedding(24, self.d_model // 4),
            'day': nn.Embedding(32, self.d_model // 4),
            'weekday': nn.Embedding(7, self.d_model // 4),
            'month': nn.Embedding(13, self.d_model // 4)
        })
        self.projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
    def _create_positional_encoding(self):
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    def forward(self, input_embeddings: torch.Tensor, temporal_features=None, positions=None):
        batch_size, seq_len, _ = input_embeddings.size()
        embeddings = input_embeddings
        if positions is not None:
            pos_embeddings = self.positional_encoding[:, positions.long(), :]
        else:
            pos_embeddings = self.positional_encoding[:, :seq_len, :]
        embeddings = embeddings + pos_embeddings
        if temporal_features is not None:
            temporal_emb = self._process_temporal_features(temporal_features)
            embeddings = embeddings + temporal_emb
        embeddings = self.projection(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    def _process_temporal_features(self, temporal_features):
        temporal_embs = []
        for feature_name, feature_values in temporal_features.items():
            if feature_name in self.temporal_embeddings:
                emb = self.temporal_embeddings[feature_name](feature_values.long())
                temporal_embs.append(emb)
        if temporal_embs:
            combined = torch.cat(temporal_embs, dim=-1)
            if combined.size(-1) < self.d_model:
                padding = torch.zeros(*combined.shape[:-1], self.d_model - combined.size(-1), device=combined.device)
                combined = torch.cat([combined, padding], dim=-1)
            elif combined.size(-1) > self.d_model:
                combined = combined[..., :self.d_model]
            return combined
        return torch.zeros_like(temporal_features[list(self.temporal_embeddings.keys())[0]])
    def get_output_dim(self):
        return self.d_model
    def get_embedding_type(self):
        return "temporal"
