import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..base import BaseAttention
from .common import WaveletDecomposition

class WaveletAttention(BaseAttention):
    """
    Performs multi-resolution attention by first decomposing the input signals
    into different frequency sub-bands using a learnable wavelet transform.
    Attention is then computed on each sub-band and the results are fused.
    """
    def __init__(self, d_model: int, n_heads: int, levels: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        self.levels = levels
        
        self.decomposition = WaveletDecomposition(d_model, levels)
        
        # An attention layer for each decomposition component
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) 
            for _ in range(levels + 1)
        ])
        
        # Fusion layer to combine outputs from all levels
        self.fusion_proj = nn.Linear(d_model * (levels + 1), d_model)
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

        # Decompose Q, K, V into wavelet components
        q_components = self.decomposition(queries)
        k_components = self.decomposition(keys)
        v_components = self.decomposition(values)
        
        level_outputs = []
        avg_attention = torch.zeros(B, self.n_heads, L, keys.shape[1], device=queries.device)
        
        for i, (q_comp, k_comp, v_comp) in enumerate(zip(q_components, k_components, v_components)):
            # Upsample components to the original sequence length to compute attention
            if q_comp.size(1) != L:
                q_comp = F.interpolate(q_comp.transpose(1, 2), size=L, mode='linear').transpose(1, 2)
                k_comp = F.interpolate(k_comp.transpose(1, 2), size=L, mode='linear').transpose(1, 2)
                v_comp = F.interpolate(v_comp.transpose(1, 2), size=L, mode='linear').transpose(1, 2)

            level_out, level_attn = self.level_attentions[i](q_comp, k_comp, v_comp, attn_mask=attn_mask)
            level_outputs.append(level_out)
            if level_attn is not None:
                avg_attention += level_attn / (self.levels + 1)
        
        # Concatenate and fuse the outputs from all levels
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        
        output = self.layer_norm(self.dropout(fused_output) + residual)
        
        return output, avg_attention