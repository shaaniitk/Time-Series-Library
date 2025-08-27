"""CausalConvolution split from temporal_conv_attention.

Provides dilated multi-scale causal convolutions wrapped in an attention style interface.
"""
from __future__ import annotations
import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAttention
from layers.modular.core.logger import logger


class CausalConvolution(BaseAttention):
    """Multi-scale dilated causal convolution with attention-like interface.

    Parameters
    ----------
    d_model: int
        Model dimensionality.
    n_heads: int
        Number of heads (used for internal temporal attention projection sizing).
    kernel_sizes: list[int]
        Kernel sizes to use for multi-scale causal convolutions.
    dilation_rates: list[int]
        Dilation rates applied per kernel size.
    dropout: float
        Dropout probability.
    activation: str
        One of {'gelu','relu','tanh'}.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kernel_sizes: List[int] | None = None,
        dilation_rates: List[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 5, 7]
        dilation_rates = dilation_rates or [1, 2, 4]
        logger.info(
            "Initializing CausalConvolution: d_model=%s kernels=%s dilations=%s",
            d_model,
            kernel_sizes,
            dilation_rates,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        self.causal_convs = nn.ModuleList()
        for k in kernel_sizes:
            conv_layers = nn.ModuleList()
            for d in dilation_rates:
                padding = (k - 1) * d
                conv_layers.append(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size=k,
                        dilation=d,
                        padding=padding,
                    )
                )
            self.causal_convs.append(conv_layers)
        self.query_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.key_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.value_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.output_projection = nn.Linear(d_model * len(kernel_sizes), d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = (
            nn.GELU()
            if activation == "gelu"
            else nn.ReLU()
            if activation == "relu"
            else nn.Tanh()
        )
        self.positional_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.output_dim_multiplier = 1

    def apply_causal_mask(self, x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
        trim = (kernel_size - 1) * dilation
        if trim > 0:
            x = x[:, :, :-trim]
        return x

    def multi_scale_causal_conv(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)
        scale_outputs: list[torch.Tensor] = []
        for k_idx, conv_layers in enumerate(self.causal_convs):
            k = self.kernel_sizes[k_idx]
            dilated = []
            for d_idx, conv in enumerate(conv_layers):
                d = self.dilation_rates[d_idx]
                out = self.apply_causal_mask(conv(x_conv), k, d)
                if out.size(-1) < L:
                    out = F.pad(out, (0, L - out.size(-1)))
                dilated.append(self.activation(out))
            scale_outputs.append(torch.stack(dilated, 0).mean(0))
        return scale_outputs

    def temporal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, L = q.shape
        H = self.n_heads
        d_k = D // H
        qh = q.view(B, H, d_k, L).transpose(2, 3)
        kh = k.view(B, H, d_k, L).transpose(2, 3)
        vh = v.view(B, H, d_k, L).transpose(2, 3)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(d_k)
        causal_mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, vh).transpose(2, 3).contiguous().view(B, D, L)
        return ctx, attn

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: float | None = None,
        delta: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = queries
        pos_encoded = queries + self.positional_conv(queries.transpose(1, 2)).transpose(1, 2)
        conv_outputs = self.multi_scale_causal_conv(pos_encoded)
        q_conv = self.query_conv(queries.transpose(1, 2))
        k_conv = self.key_conv(keys.transpose(1, 2))
        v_conv = self.value_conv(values.transpose(1, 2))
        enhanced_k = k_conv
        enhanced_v = v_conv
        for co in conv_outputs:
            enhanced_k = enhanced_k + co
            enhanced_v = enhanced_v + co
        attn_out, attn_w = self.temporal_attention(q_conv, enhanced_k, enhanced_v)
        combined_conv = torch.cat(conv_outputs, dim=1).transpose(1, 2)
        conv_features = self.output_projection(combined_conv)
        final_output = attn_out.transpose(1, 2) + conv_features
        return self.layer_norm(final_output + residual), attn_w

__all__ = ["CausalConvolution"]
