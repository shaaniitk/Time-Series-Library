"""
Sparse attention with configurable sparsity pattern (modularized).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseSparseAttention

class SparseAttention(BaseSparseAttention):
    """Sparse attention with configurable sparsity pattern."""
    def __init__(self, d_model=512, n_heads=8, sparsity_factor=0.1, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.sparsity_factor = sparsity_factor
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity_factor = sparsity_factor
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def get_output_dim(self) -> int:
        return self.d_model

    def get_attention_type(self) -> str:
        return "sparse"

    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        d_k = self.d_model // self.n_heads
        Q = self.w_qs(queries).view(B, L, self.n_heads, d_k).transpose(1, 2)
        K = self.w_ks(keys).view(B, L, self.n_heads, d_k).transpose(1, 2)
        V = self.w_vs(values).view(B, L, self.n_heads, d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        k = int(L * self.sparsity_factor)
        if k > 0:
            top_k = torch.topk(scores, k, dim=-1)[1]
            sparse_mask = torch.zeros_like(scores)
            sparse_mask.scatter_(-1, top_k, 1)
            scores = scores * sparse_mask - 1e9 * (1 - sparse_mask)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.fc(output)
        return output, attn
