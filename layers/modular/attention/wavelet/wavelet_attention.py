"""WaveletAttention component (split from monolithic wavelet_attention module)."""
from __future__ import annotations

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAttention
from utils.logger import logger  # type: ignore
from .wavelet_decomposition import WaveletDecomposition


class WaveletAttention(BaseAttention):
    """Wavelet-based multi-resolution attention with learnable decomposition."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        levels: int = 3,
        n_levels: Optional[int] = None,
        wavelet_type: str = "learnable",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_levels is not None:
            levels = n_levels
        logger.info("Initializing WaveletAttention: d_model=%s levels=%s", d_model, levels)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.wavelet_decomp = WaveletDecomposition(d_model, levels)
        self.level_attentions = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in range(levels + 1)]
        )
        self.level_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        self.fusion_proj = nn.Linear(d_model * (levels + 1), d_model)
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
        _, q_components = self.wavelet_decomp(queries)
        _, k_components = self.wavelet_decomp(keys)
        _, v_components = self.wavelet_decomp(values)
        level_outputs = []
        level_attentions = []
        for q_comp, k_comp, v_comp, attention_layer in zip(
            q_components, k_components, v_components, self.level_attentions
        ):
            if q_comp.size(1) != L:
                q_comp = F.interpolate(q_comp.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
                k_comp = F.interpolate(k_comp.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
                v_comp = F.interpolate(v_comp.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
            level_out, level_attn = attention_layer(q_comp, k_comp, v_comp, attn_mask=attn_mask)
            level_outputs.append(level_out)
            level_attentions.append(level_attn)
        weights = F.softmax(self.level_weights, dim=0)
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        avg_attention = torch.stack(level_attentions, dim=0).mean(dim=0)
        return fused_output, avg_attention

__all__ = ["WaveletAttention"]

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="WaveletAttention",
    component_class=WaveletAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "d_model": 32,
        "n_heads": 4,
        "levels": 2,
        "dropout": 0.1,
    },
)
