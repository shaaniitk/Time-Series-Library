"""AdaptiveWaveletAttention component (split from monolithic wavelet_attention module)."""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAttention
from utils.logger import logger  # type: ignore
from .wavelet_attention import WaveletAttention


class AdaptiveWaveletAttention(BaseAttention):
    """Adaptive selection over multiple WaveletAttention levels."""

    def __init__(self, d_model: int, n_heads: int, max_levels: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        logger.info("Initializing AdaptiveWaveletAttention: d_model=%s max_levels=%s", d_model, max_levels)
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_levels = max_levels
        self.level_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, max_levels),
            nn.Softmax(dim=-1),
        )
        self.wavelet_attentions = nn.ModuleList(
            [WaveletAttention(d_model, n_heads, levels=i + 1) for i in range(max_levels)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim_multiplier = 1

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = queries.shape
        input_summary = queries.mean(dim=1)
        level_weights = self.level_selector(input_summary)
        level_outputs = []
        level_attentions = []
        for wavelet_attn in self.wavelet_attentions:
            output, attention = wavelet_attn(queries, keys, values, attn_mask)
            level_outputs.append(output)
            level_attentions.append(attention)
        level_outputs_t = torch.stack(level_outputs, dim=-1)
        level_attentions_t = torch.stack(level_attentions, dim=-1)
        final_output = torch.sum(level_outputs_t * level_weights.view(B, 1, 1, -1), dim=-1)
        final_attention = torch.sum(level_attentions_t * level_weights.view(B, 1, 1, 1, -1), dim=-1)
        return self.dropout(final_output), final_attention

__all__ = ["AdaptiveWaveletAttention"]
