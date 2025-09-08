"""MultiScaleWaveletAttention component (split from monolithic wavelet_attention module)."""
from __future__ import annotations

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..base import BaseAttention
from utils.logger import logger  # type: ignore
from .wavelet_attention import WaveletAttention


class MultiScaleWaveletAttention(BaseAttention):
    """Apply WaveletAttention across predefined scales and fuse outputs."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        scales: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if scales is None:
            scales = [1, 2, 4, 8]
        logger.info("Initializing MultiScaleWaveletAttention: scales=%s", scales)
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.scale_attentions = nn.ModuleList([WaveletAttention(d_model, n_heads, levels=3) for _ in scales])
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_proj = nn.Linear(d_model * len(scales), d_model)
        self.output_dim_multiplier = 1
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = queries.shape
        scale_outputs = []
        scale_attentions = []
        for idx, scale_attn in enumerate(self.scale_attentions):
            scale = self.scales[idx]
            if scale == 1:
                q_scaled, k_scaled, v_scaled = queries, keys, values
            else:
                target_len = max(L // scale, 1)
                q_scaled = (
                    F.interpolate(
                        queries.transpose(1, 2), size=target_len, mode="linear", align_corners=False
                    ).transpose(1, 2)
                )
                k_scaled = (
                    F.interpolate(keys.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
                )
                v_scaled = (
                    F.interpolate(values.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
                )
            output, attention = scale_attn(q_scaled, k_scaled, v_scaled, attn_mask)
            if output.size(1) != L:
                output = F.interpolate(output.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
            scale_outputs.append(output)
            # Upsample attention maps to [B, L, L] if provided
            if attention is not None:
                attn = attention
                if attn.dim() == 4:  # [B, H, Lq, Lk]
                    attn = attn.mean(dim=1)
                if attn.size(-2) != L or attn.size(-1) != L:
                    attn = F.interpolate(attn.unsqueeze(1), size=(L, L), mode="bilinear", align_corners=False).squeeze(1)
                scale_attentions.append(attn)


        weights = F.softmax(self.scale_weights, dim=0)
        # Stack outputs: [num_scales, B, L, D]
        stacked_outputs = torch.stack(scale_outputs, dim=0)
        # Weighted sum over scales: [B, L, D]
        fused_output = torch.sum(weights.view(-1, 1, 1, 1) * stacked_outputs, dim=0)
        avg_attention = torch.stack(scale_attentions, dim=0).mean(dim=0) if scale_attentions else None
        return self.dropout(fused_output), avg_attention

__all__ = ["MultiScaleWaveletAttention"]

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="MultiScaleWaveletAttention",
    component_class=MultiScaleWaveletAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "scales": [1, 2, 4],
        "dropout": 0.1,
    },
)
