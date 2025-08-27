"""VariationalAttention split implementation."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAttention
from layers.modular.core.logger import logger

class VariationalAttention(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, learn_variance: bool = True) -> None:
        super().__init__()
        logger.info("Initializing VariationalAttention (split module)")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.learn_variance = learn_variance
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        if learn_variance:
            self.w_qs_var = nn.Linear(d_model, d_model)
            self.w_ks_var = nn.Linear(d_model, d_model)
            self.w_vs_var = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_dim_multiplier = 1

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask=None, tau=None, delta=None):  # type: ignore[override]
        B, L, D = queries.shape
        H = self.n_heads
        residual = queries
        q_mu = self.w_qs(queries).view(B, L, H, self.d_k).transpose(1, 2)
        k_mu = self.w_ks(keys).view(B, L, H, self.d_k).transpose(1, 2)
        v_mu = self.w_vs(values).view(B, L, H, self.d_k).transpose(1, 2)
        if self.learn_variance:
            q_log_var = self.w_qs_var(queries).view(B, L, H, self.d_k).transpose(1, 2)
            k_log_var = self.w_ks_var(keys).view(B, L, H, self.d_k).transpose(1, 2)
            v_log_var = self.w_vs_var(values).view(B, L, H, self.d_k).transpose(1, 2)
            q = self.reparameterize(q_mu, q_log_var)
            k = self.reparameterize(k_mu, k_log_var)
            v = self.reparameterize(v_mu, v_log_var)
        else:
            q, k, v = q_mu, k_mu, v_mu
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
        output = self.layer_norm(self.w_o(context) + residual)
        return output, attn_weights
