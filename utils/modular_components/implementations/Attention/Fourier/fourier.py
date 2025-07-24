"""
Fourier-based attention for capturing periodic patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import FourierAttentionBase

class FourierAttention(FourierAttentionBase):
    def __init__(self, d_model, n_heads, seq_len, dropout=0.1):
        super().__init__(d_model, n_heads, dropout)
        self.seq_len = seq_len
        self.freq_weights = nn.Parameter(torch.randn(seq_len // 2 + 1, n_heads))
        self.phase_weights = nn.Parameter(torch.zeros(seq_len // 2 + 1, n_heads))
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        freq_filter = torch.complex(
            torch.cos(self.phase_weights) * self.freq_weights,
            torch.sin(self.phase_weights) * self.freq_weights
        )
        x_freq = x_freq.unsqueeze(-1) * freq_filter.unsqueeze(0).unsqueeze(2)
        x_filtered = torch.fft.irfft(x_freq.mean(-1), n=L, dim=1)
        qkv = self.qkv(x_filtered).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out, attn
