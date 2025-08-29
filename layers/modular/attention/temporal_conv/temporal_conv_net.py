"""TemporalConvNet split from temporal_conv_attention."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from ..base import BaseAttention
from utils.logger import logger
from .temporal_block import TemporalBlock


class TemporalConvNet(BaseAttention):
    """Temporal Convolution Network (TCN) based attention mechanism."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        logger.info(
            "Initializing TemporalConvNet: levels=%s kernel=%s", num_levels, kernel_size
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.tcn_layers = nn.ModuleList()
        dilation = 1
        for _ in range(num_levels):
            self.tcn_layers.append(
                TemporalBlock(
                    d_model,
                    d_model,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )
            dilation *= 2
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.output_dim_multiplier = 1

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: float | None = None,
        delta: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = keys
        residual = x
        current = x.transpose(1, 2)
        for layer in self.tcn_layers:
            current = layer(current)
        tcn_out = current.transpose(1, 2)
        attn_out, attn_weights = self.temporal_attention(queries, tcn_out, values, attn_mask=attn_mask)
        output = self.output_norm(attn_out + residual)
        output = self.output_dropout(output)
        return output, attn_weights

__all__ = ["TemporalConvNet"]
