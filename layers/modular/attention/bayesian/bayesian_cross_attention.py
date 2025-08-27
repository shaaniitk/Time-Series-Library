"""BayesianCrossAttention split implementation."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAttention
from .bayesian_linear import BayesianLinear
from layers.modular.core.logger import logger

class BayesianCrossAttention(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, prior_std: float = 1.0) -> None:
        super().__init__()
        logger.info("Initializing BayesianCrossAttention (split module)")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_qs = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_dim_multiplier = 1

    def get_kl_divergence(self) -> torch.Tensor:
        return self.w_qs.kl_divergence() + self.w_o.kl_divergence()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask=None, tau=None, delta=None):  # type: ignore[override]
        B, L_dec, D = queries.shape
        L_enc = keys.size(1)
        H = self.n_heads
        residual = queries
        q = self.w_qs(queries).view(B, L_dec, H, self.d_k).transpose(1, 2)
        k = self.w_ks(keys).view(B, L_enc, H, self.d_k).transpose(1, 2)
        v = self.w_vs(values).view(B, L_enc, H, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L_dec, D)
        output = self.layer_norm(self.w_o(context) + residual)
        return output, attn_weights
