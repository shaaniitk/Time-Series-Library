"""
FourierCrossAttention: Cross-attention using Fourier features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import FourierAttentionBase

class FourierCrossAttention(FourierAttentionBase):
    def __init__(self, d_model, n_heads, seq_len_q=96, seq_len_kv=96, modes=64, dropout=0.1):
        super().__init__(d_model, n_heads, dropout)
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.modes = modes
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.fourier_weight = nn.Parameter(torch.randn(d_model, modes, dtype=torch.cfloat) * 0.02)

    def forward(self, queries, keys, values, attn_mask=None):
        B, Lq, _ = queries.shape
        B, Lkv, _ = keys.shape
        Q = self.w_qs(queries)
        K = self.w_ks(keys)
        V = self.w_vs(values)
        x_fft = torch.fft.rfft(queries, dim=1)
        out_fft = torch.zeros_like(x_fft)
        modes = min(self.modes, x_fft.size(1))
        out_fft[:, :modes] = x_fft[:, :modes] * self.fourier_weight[:, :modes].unsqueeze(0)
        fourier_result = torch.fft.irfft(out_fft, n=Lq, dim=1)
        d_k = self.d_model // self.n_heads
        Q = Q.view(B, Lq, self.n_heads, d_k).transpose(1, 2)
        K = K.view(B, Lkv, self.n_heads, d_k).transpose(1, 2)
        V = V.view(B, Lkv, self.n_heads, d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        output = self.fc(attn_output + fourier_result)
        return output, attn
