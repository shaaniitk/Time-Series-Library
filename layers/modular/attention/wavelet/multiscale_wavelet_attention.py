import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..base import BaseAttention
from .wavelet_attention import WaveletAttention

class MultiScaleWaveletAttention(BaseAttention):
    """
    Applies WaveletAttention across a predefined set of downsampled scales
    of the input data and fuses the results. This is distinct from the
    hierarchical approach as it processes full signals at different resolutions.
    """
    def __init__(self, d_model: int, n_heads: int, scales: List[int] = [1, 2, 4], dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        self.scales = scales
        
        # A WaveletAttention instance for each scale
        self.scale_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=3, dropout=dropout) for _ in scales
        ])
        
        # Fusion layer for combining outputs from all scales
        self.fusion_proj = nn.Linear(d_model * len(scales), d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L, D = queries.shape
        residual = queries
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                q_scaled, k_scaled, v_scaled = queries, keys, values
            else:
                # Downsample sequence to the target scale
                target_len = max(L // scale, 1)
                q_scaled = F.interpolate(queries.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
                k_scaled = F.interpolate(keys.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
                v_scaled = F.interpolate(values.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
            
            # Compute attention at this scale
            output, _ = self.scale_attentions[i](q_scaled, k_scaled, v_scaled)
            
            # Upsample output back to original sequence length
            if output.size(1) != L:
                output = F.interpolate(output.transpose(1, 2), size=L, mode='linear').transpose(1, 2)
            
            scale_outputs.append(output)
            
        # Concatenate and fuse the outputs from all scales
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        
        output = self.layer_norm(self.dropout(fused_output) + residual)
        
        return output, None