"""FourierCrossAttention component (split from monolithic module)."""
from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAttention
from utils.logger import logger  # type: ignore


class FourierCrossAttention(BaseAttention):
    """Cross attention variant operating in frequency domain before time-domain mix."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len_q: int,
        seq_len_kv: int,
        modes: int = 64,
        mode_select_method: str = "random",
        activation: str = "tanh",
        num_heads: int = 8,
        # Testing-friendly extras (absorbs BaseAttention defaults from tests)
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Map optional d_model/n_heads from the generic test harness
        if d_model is not None:
            in_channels = d_model
        if n_heads is not None:
            num_heads = n_heads
        super().__init__(d_model=in_channels, n_heads=num_heads)
        logger.info(
            "Initializing FourierCrossAttention: %s -> %s (modes=%s)", in_channels, out_channels, modes
        )
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.index_q = self._get_frequency_modes(seq_len_q, modes, mode_select_method)
        self.index_kv = self._get_frequency_modes(seq_len_kv, modes, mode_select_method)
        self.scale = 1 / (in_channels * out_channels)
        self.weights_q = nn.Parameter(
            self.scale * torch.rand(num_heads, in_channels, len(self.index_q), dtype=torch.cfloat)
        )
        self.weights_kv = nn.Parameter(
            self.scale * torch.rand(num_heads, in_channels, len(self.index_kv), dtype=torch.cfloat)
        )
        self.output_dim_multiplier = out_channels / in_channels

    def _get_frequency_modes(self, seq_len: int, modes: int, mode_select_method: str) -> List[int]:
        modes = min(modes, seq_len // 2)
        if mode_select_method == "random":
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_q, D = queries.shape
        _, L_kv, _ = keys.shape
        q_ft = torch.fft.rfft(queries, dim=1)
        k_ft = torch.fft.rfft(keys, dim=1)
        v_ft = torch.fft.rfft(values, dim=1)
        q_filtered = torch.zeros_like(q_ft)
        kv_filtered = torch.zeros_like(k_ft)
        for i, freq_idx in enumerate(self.index_q):
            if freq_idx < q_ft.shape[1]:
                q_filtered[:, i, :] = q_ft[:, freq_idx, :] * self.weights_q[0, :, i]
        for i, freq_idx in enumerate(self.index_kv):
            if freq_idx < k_ft.shape[1]:
                kv_filtered[:, i, :] = k_ft[:, freq_idx, :] * self.weights_kv[0, :, i]
        q_processed = torch.fft.irfft(q_filtered, n=L_q, dim=1)
        k_processed = torch.fft.irfft(kv_filtered, n=L_kv, dim=1)
        v_processed = torch.fft.irfft(kv_filtered, n=L_kv, dim=1)
        scale = math.sqrt(D)
        attn_scores = torch.bmm(q_processed, k_processed.transpose(1, 2)) / scale
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float("inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.bmm(attn_weights, v_processed)
        return output, attn_weights

__all__ = ["FourierCrossAttention"]

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="FourierCrossAttention",
    component_class=FourierCrossAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        # The test harness will also pass d_model/n_heads; we absorb them above
        "in_channels": 32,
        "out_channels": 32,
        "seq_len_q": 32,
        "seq_len_kv": 32,
        "modes": 8,
        "mode_select_method": "random",
        "activation": "tanh",
        "num_heads": 4,
    },
)
