"""ConvolutionalAttention split from temporal_conv_attention."""
from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAttention
from utils.logger import logger


class ConvolutionalAttention(BaseAttention):
    """Convolutional attention combining spatial and temporal convolutions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conv_kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        logger.info("Initializing ConvolutionalAttention: d_model=%s", d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.spatial_conv = nn.Conv2d(
            1,
            n_heads,
            kernel_size=(conv_kernel_size, conv_kernel_size),
            padding=(conv_kernel_size // 2, conv_kernel_size // 2),
        )
        self.temporal_conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_dim_multiplier = 1

    def compute_conv_attention_weights(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        B, L, D = queries.shape
        q_exp = queries.unsqueeze(2).expand(B, L, L, D)
        k_exp = keys.unsqueeze(1).expand(B, L, L, D)
        interactions = (q_exp * k_exp).mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        conv_weights = self.spatial_conv(interactions)
        return F.softmax(conv_weights, dim=-1)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: float | None = None,
        delta: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = queries.shape
        H = self.n_heads
        residual = queries
        q_temp = self.temporal_conv_q(queries.transpose(1, 2)).transpose(1, 2)
        k_temp = self.temporal_conv_k(keys.transpose(1, 2)).transpose(1, 2)
        v_temp = self.temporal_conv_v(values.transpose(1, 2)).transpose(1, 2)
        q = self.w_qs(q_temp).view(B, L, H, self.d_k).transpose(1, 2)
        k = self.w_ks(k_temp).view(B, L, H, self.d_k).transpose(1, 2)
        v = self.w_vs(v_temp).view(B, L, H, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        conv_attn = self.compute_conv_attention_weights(q_temp, k_temp)
        combined = scores + conv_attn
        if attn_mask is not None:
            combined.masked_fill_(attn_mask == 0, -1e9)
        attn_w = F.softmax(combined, dim=-1)
        attn_w = self.dropout(attn_w)
        context = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(B, L, D)
        output = self.w_o(context)
        return self.layer_norm(output + residual), attn_w

__all__ = ["ConvolutionalAttention"]

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="ConvolutionalAttention",
    component_class=ConvolutionalAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "conv_kernel_size": 3,
        "pool_size": 2,
        "dropout": 0.1,
    },
)

 
