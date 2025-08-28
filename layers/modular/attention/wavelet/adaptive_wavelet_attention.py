import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..base import BaseAttention
from .wavelet_attention import WaveletAttention

class AdaptiveWaveletAttention(BaseAttention):
    """
    A meta-attention mechanism that adaptively selects the number of wavelet
    decomposition levels. It maintains multiple WaveletAttention instances and
    computes a weighted average of their outputs based on the input queries.
    """
    def __init__(self, d_model: int, n_heads: int, max_levels: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        self.max_levels = max_levels

        # A pool of WaveletAttention modules, each with a different number of levels
        self.wavelet_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=i + 1, dropout=dropout)
            for i in range(max_levels)
        ])
        
        # Gating network to determine the weight for each level
        self.level_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_levels)
        )
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
        
        # 1. Determine level weights from the queries
        input_summary = queries.mean(dim=1) # [B, D]
        level_logits = self.level_selector(input_summary) # [B, max_levels]
        level_weights = F.softmax(level_logits, dim=-1) # [B, max_levels]
        
        # 2. Compute outputs for all possible levels
        level_outputs = []
        for wavelet_attn in self.wavelet_attentions:
            output, _ = wavelet_attn(queries, keys, values, attn_mask)
            level_outputs.append(output)
            
        # 3. Compute the weighted average of the outputs
        # Stack outputs: [max_levels, B, L, D] -> [B, L, D, max_levels]
        stacked_outputs = torch.stack(level_outputs, dim=0).permute(1, 2, 3, 0)
        
        # Reshape weights for broadcasting: [B, 1, 1, max_levels]
        reshaped_weights = level_weights.view(B, 1, 1, self.max_levels)
        
        final_output = torch.sum(stacked_outputs * reshaped_weights, dim=-1)
        
        return self.dropout(final_output), None # Returning combined weights is complex