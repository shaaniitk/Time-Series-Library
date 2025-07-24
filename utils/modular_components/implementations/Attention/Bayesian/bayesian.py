"""
Bayesian attention with uncertainty quantification (modularized).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseBayesianAttention

class BayesianAttention(BaseBayesianAttention):
    """Bayesian attention with uncertainty quantification."""
    def __init__(self, d_model=512, n_heads=8, prior_std=1.0, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.prior_std = prior_std
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.prior_std = prior_std
        self.q_mean = nn.Linear(self.d_model, self.d_model)
        self.q_logvar = nn.Linear(self.d_model, self.d_model)
        self.k_mean = nn.Linear(self.d_model, self.d_model)
        self.k_logvar = nn.Linear(self.d_model, self.d_model)
        self.v_mean = nn.Linear(self.d_model, self.d_model)
        self.v_logvar = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
    def get_output_dim(self) -> int:
        return self.d_model
    def get_attention_type(self) -> str:
        return "bayesian"
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        q_mean = self.q_mean(queries)
        q_logvar = self.q_logvar(queries)
        q = self._reparameterize(q_mean, q_logvar)
        k_mean = self.k_mean(keys)
        k_logvar = self.k_logvar(keys)
        k = self._reparameterize(k_mean, k_logvar)
        v_mean = self.v_mean(values)
        v_logvar = self.v_logvar(values)
        v = self._reparameterize(v_mean, v_logvar)
        q = q.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        scale = math.sqrt(D // self.n_heads)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights
