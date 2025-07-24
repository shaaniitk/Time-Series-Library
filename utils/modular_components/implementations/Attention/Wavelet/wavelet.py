"""
Wavelet-based attention for multi-scale analysis (modularized).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseWaveletAttention

class WaveletAttention(BaseWaveletAttention):
    """Wavelet-based attention for multi-scale analysis."""
    def __init__(self, d_model=512, n_heads=8, levels=3, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.levels = levels
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        self.wavelet_filters = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_model, 4))
            for _ in range(self.levels)
        ])
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def get_output_dim(self) -> int:
        return self.d_model

    def get_attention_type(self) -> str:
        return "wavelet"

    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)

    def _wavelet_transform(self, x, level):
        B, L, D = x.shape
        filter_weights = self.wavelet_filters[level]
        x_conv = F.conv1d(x.transpose(1, 2), filter_weights.unsqueeze(1), padding=2, groups=D)
        x_down = x_conv[:, :, ::2]
        return x_down.transpose(1, 2)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        scale_features = []
        current_q = queries
        for level in range(self.levels):
            if current_q.shape[1] > 4:
                current_q = self._wavelet_transform(current_q, level)
                scale_features.append(current_q)
        if scale_features:
            processed = scale_features[-1]
            qkv = self.qkv(processed).reshape(processed.shape[0], processed.shape[1], 3, self.n_heads, D // self.n_heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scale = math.sqrt(D // self.n_heads)
            attn_scores = (q @ k.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = (attn_weights @ v).transpose(1, 2).reshape(processed.shape[0], processed.shape[1], D)
            out = F.interpolate(out.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
        else:
            qkv = self.qkv(queries).reshape(B, L, 3, self.n_heads, D // self.n_heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scale = math.sqrt(D // self.n_heads)
            attn_scores = (q @ k.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights
