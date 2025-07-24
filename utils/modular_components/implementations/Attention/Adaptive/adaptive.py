"""
Adaptive attention with dynamic parameter selection (modularized).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseAdaptiveAttention

class AdaptiveAttention(BaseAdaptiveAttention):
    """Adaptive attention with dynamic parameter selection."""
    def __init__(self, d_model=512, n_heads=8, adaptation_rate=0.1, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.adaptation_rate = adaptation_rate
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.adaptation_rate = adaptation_rate
        self.gate_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_heads),
            nn.Sigmoid()
        )
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def get_output_dim(self) -> int:
        return self.d_model
    def get_attention_type(self) -> str:
        return "adaptive"
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        gates = self.gate_network(queries.mean(dim=1))  # [B, n_heads]
        qkv = self.qkv(queries).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scale = math.sqrt(D // self.n_heads)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        gates = gates.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, 1, 1]
        attn_weights = attn_weights * gates
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights
