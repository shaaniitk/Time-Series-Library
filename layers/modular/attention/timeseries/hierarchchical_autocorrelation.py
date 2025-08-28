import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..base import BaseAttention
from .autocorrelation import AutoCorrelationAttention

class HierarchicalAutoCorrelation(BaseAttention):
    """
    A meta-attention component that captures patterns at multiple time horizons
    by applying autocorrelation at different temporal resolutions (downsampling)
    and hierarchically combining the results.
    """
    def __init__(self, d_model: int, n_heads: int, hierarchy_levels: List[int] = [1, 4, 16], factor: int = 1, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        self.hierarchy_levels = hierarchy_levels
        
        # Create an AutoCorrelationAttention layer for each hierarchy level
        self.level_correlations = nn.ModuleList([
            AutoCorrelationAttention(
                d_model=d_model,
                n_heads=n_heads,
                factor=factor,
                dropout=dropout,
                scales=[level] # Use a single scale for each instance
            )
            for _ in hierarchy_levels
        ])
        
        # Learnable weights to fuse the outputs from different levels
        self.fusion_weights = nn.Parameter(torch.ones(len(hierarchy_levels)) / len(hierarchy_levels))
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
        level_outputs = []
        
        for i, level in enumerate(self.hierarchy_levels):
            if level == 1:
                # Process at original resolution
                level_out, _ = self.level_correlations[i](queries, keys, values, attn_mask)
            else:
                # Downsample for coarser temporal resolutions
                target_len = max(L // level, 1)
                q_down = F.interpolate(queries.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
                k_down = F.interpolate(keys.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
                v_down = F.interpolate(values.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
                
                level_out, _ = self.level_correlations[i](q_down, k_down, v_down)
                
                # Upsample the output back to the original resolution
                level_out = F.interpolate(level_out.transpose(1, 2), size=L, mode='linear').transpose(1, 2)
            
            level_outputs.append(level_out)
        
        # Fuse the outputs using the learnable weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted combination
        weighted_outputs = [w * out for w, out in zip(weights, level_outputs)]
        combined_output = sum(weighted_outputs)
        
        output = self.layer_norm(self.dropout(combined_output) + residual)

        return output, None