import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..base import BaseAttention

class FourierAttention(BaseAttention):
    """
    Applies a learnable filter to the frequency domain representation of the
    input sequence before performing standard multi-head attention. This allows
    the model to focus on or ignore specific periodic patterns.
    """
    def __init__(self, d_model: int, n_heads: int, seq_len: int = 96, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        
        # We need to know the number of frequency components
        self.num_freqs = seq_len // 2 + 1

        # Learnable complex filter parameters
        self.freq_weights = nn.Parameter(torch.randn(self.num_freqs, self.d_k))
        self.phase_weights = nn.Parameter(torch.zeros(self.num_freqs, self.d_k))

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L, _ = queries.shape
        residual = queries

        # --- Frequency Domain Filtering ---
        # 1. Project queries and transform to frequency domain
        Q_proj = self.w_q(queries).view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3) # [B, H, L, D_k]
        Q_fft = torch.fft.rfft(Q_proj, n=L, dim=2) # [B, H, num_freqs, D_k]

        # 2. Construct and apply the complex filter
        freq_filter = torch.complex(
            torch.cos(self.phase_weights) * self.freq_weights,
            torch.sin(self.phase_weights) * self.freq_weights
        ).unsqueeze(0).unsqueeze(0) # [1, 1, num_freqs, D_k]
        
        Q_filtered_fft = Q_fft * freq_filter
        
        # 3. Transform back to the time domain
        Q_filtered = torch.fft.irfft(Q_filtered_fft, n=L, dim=2) # [B, H, L, D_k]

        # --- Standard Attention with Filtered Queries ---
        K = self.w_k(keys).view(B, keys.shape[1], self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.w_v(values).view(B, values.shape[1], self.n_heads, self.d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(Q_filtered, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
             if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
             scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_model)
        
        output = self.w_o(context)
        output = self.layer_norm(self.dropout(output) + residual)

        return output, attention_weights