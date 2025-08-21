"""MultiGraphAttention split from graph_attention."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .graph_attention_layer import GraphAttentionLayer


class MultiGraphAttention(nn.Module):
    """Wrapper providing residual + norm around a GraphAttentionLayer."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.gat = GraphAttentionLayer(d_model, n_heads, dropout, concat=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x, attention = self.gat(x, adj_matrix, attn_mask)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x, attention

__all__ = ["MultiGraphAttention"]
