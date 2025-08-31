"""FourierAttention component (split from monolithic module).

Implements frequency-domain pre-filtering before standard multi-head attention.
"""
from __future__ import annotations

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAttention
from utils.logger import logger  # type: ignore


class FourierAttention(BaseAttention):
    """Fourier-based attention for capturing periodic patterns in time series.

    Applies an FFT, learnable complex filtering with optional adaptive
    frequency weighting, then performs standard multi-head attention in the
    time domain on the filtered signal.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int = 96,
        frequency_selection: str = "adaptive",
        dropout: float = 0.1,
        temperature: float = 1.0,
        learnable_filter: bool = True,
    ) -> None:
        super().__init__()
        logger.info(
            "Initializing FourierAttention: d_model=%s n_heads=%s seq_len=%s", d_model, n_heads, seq_len
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature
        self.learnable_filter = learnable_filter

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        max_freq_dim = max(seq_len // 2 + 1, 64)
        self.freq_weights = nn.Parameter(torch.randn(max_freq_dim))
        self.phase_weights = nn.Parameter(torch.zeros(max_freq_dim))

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.frequency_selection = frequency_selection
        if frequency_selection == "adaptive":
            self.freq_selector = nn.Linear(d_model, max_freq_dim)

        self.output_dim_multiplier = 1

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = queries.shape

        # Frequency domain transform of queries
        queries_freq = torch.fft.rfft(queries, dim=1)
        freq_dim = queries_freq.shape[1]

        # Adaptive or uniform frequency selection
        if self.frequency_selection == "adaptive" and hasattr(self, "freq_selector"):
            freq_weights_input = queries.mean(dim=1)
            freq_logits = self.freq_selector(freq_weights_input)[:, :freq_dim]
            freq_selection = F.softmax(freq_logits, dim=-1)
        else:
            freq_selection = torch.ones(B, freq_dim, device=queries.device) / max(freq_dim, 1)

        # Build complex filter
        phase_weights = self.phase_weights[:freq_dim]
        magnitude_weights = self.freq_weights[:freq_dim]
        base_filter = torch.complex(
            torch.cos(phase_weights) * magnitude_weights,
            torch.sin(phase_weights) * magnitude_weights,
        )  # [freq_dim]

        if self.frequency_selection == "adaptive":
            freq_filter = (base_filter.unsqueeze(0) * freq_selection).unsqueeze(-1)
        else:
            freq_filter = base_filter.unsqueeze(0).unsqueeze(-1)

        # Apply filter and inverse transform
        queries_freq_filtered = queries_freq * freq_filter
        queries_filtered = torch.fft.irfft(queries_freq_filtered, n=L, dim=1)

    # Standard attention with filtered queries
        qkv = self.qkv(queries_filtered).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv_perm = qkv.permute(2, 0, 3, 1, 4)
        q = qkv_perm[0]
        k = qkv_perm[1]
        v = qkv_perm[2]
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float("inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights

    def _standard_attention(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv_perm = qkv.permute(2, 0, 3, 1, 4)
        q = qkv_perm[0]
        k = qkv_perm[1]
        v = qkv_perm[2]
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float("inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights

__all__ = ["FourierAttention"]

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="FourierAttention",
    component_class=FourierAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "seq_len": 64,
        "frequency_selection": "adaptive",
        "dropout": 0.1,
    },
)
