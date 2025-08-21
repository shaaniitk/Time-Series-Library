"""Local TemporalEmbedding implementation (self-contained).

Re-implements the earlier temporal embedding logic so that ``layers.modular``
no longer depends on ``utils.modular_components`` for embedding behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from typing import Dict, Optional


@dataclass
class TemporalEmbeddingConfig:
    d_model: int = 512
    max_len: int = 5000
    dropout: float = 0.1
    temp_feature_dim: int = 4  # hour, day, weekday, month


class TemporalEmbedding(nn.Module):
    def __init__(self, config: TemporalEmbeddingConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_len = config.max_len
        self._build_positional_encoding()
        # Reserve equal slices; if not divisible remainder is handled in projection
        slice_dim = max(1, self.d_model // 4)
        self.temporal_embeddings = nn.ModuleDict({
            'hour': nn.Embedding(24, slice_dim),
            'day': nn.Embedding(32, slice_dim),
            'weekday': nn.Embedding(7, slice_dim),
            'month': nn.Embedding(13, slice_dim),
        })
        self.projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def _build_positional_encoding(self) -> None:
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0), persistent=False)

    def forward(self, input_embeddings: torch.Tensor, temporal_features: Optional[Dict[str, torch.Tensor]] = None, positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        b, seq_len, _ = input_embeddings.shape
        if positions is None:
            pos_emb = self.positional_encoding[:, :seq_len, :]
        else:
            pos_emb = self.positional_encoding[:, positions.long(), :]
        emb = input_embeddings + pos_emb
        if temporal_features:
            feats = []
            for name, tensor in temporal_features.items():
                if name in self.temporal_embeddings:
                    feats.append(self.temporal_embeddings[name](tensor.long()))
            if feats:
                combined = torch.cat(feats, dim=-1)
                if combined.size(-1) < self.d_model:
                    pad = torch.zeros(b, seq_len, self.d_model - combined.size(-1), device=combined.device, dtype=combined.dtype)
                    combined = torch.cat([combined, pad], dim=-1)
                elif combined.size(-1) > self.d_model:
                    combined = combined[..., :self.d_model]
                emb = emb + combined
        emb = self.projection(emb)
        emb = self.layer_norm(emb)
        return self.dropout(emb)

    def embed_sequence(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:  # compatibility helper
        return self.forward(x, None, None)

    def get_output_dim(self) -> int:
        return self.d_model

__all__ = ["TemporalEmbedding", "TemporalEmbeddingConfig"]
