"""Standalone BayesianAttention implementation (split from monolithic module)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAttention
from layers.BayesianLayers import BayesianLinear
from utils.logger import logger

class BayesianAttention(BaseAttention):
    """Bayesian multi-head self-attention with uncertainty quantification."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, prior_std: float = 1.0,
                 temperature: float = 1.0, output_attention: bool = False) -> None:
        super().__init__()
        logger.info("Initializing BayesianAttention (split module)")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        self.output_attention = output_attention
        self.w_qs = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.w_ks = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.w_vs = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.w_o = BayesianLinear(d_model, d_model, prior_std=prior_std)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.uncertainty_scale = nn.Parameter(torch.ones(1))
        self.output_dim_multiplier = 1

    def get_kl_divergence(self) -> torch.Tensor:
        return (
            self.w_qs.kl_divergence() +
            self.w_ks.kl_divergence() +
            self.w_vs.kl_divergence() +
            self.w_o.kl_divergence()
        )

    def compute_attention_uncertainty(self, attn_weights: torch.Tensor) -> torch.Tensor:
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1, keepdim=True)
        max_entropy = math.log(attn_weights.size(-1))
        normalized_entropy = entropy / max_entropy
        return self.uncertainty_scale * normalized_entropy

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor | None = None, tau=None, delta=None):  # type: ignore[override]
        B, L, D = queries.shape
        H = self.n_heads
        residual = queries
        q = self.w_qs(queries).view(B, L, H, self.d_k).transpose(1, 2)
        k = self.w_ks(keys).view(B, L, H, self.d_k).transpose(1, 2)
        v = self.w_vs(values).view(B, L, H, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (math.sqrt(self.d_k) * self.temperature)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        uncertainty = self.compute_attention_uncertainty(attn_weights).expand_as(attn_weights)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
        output = self.layer_norm(self.w_o(context) + residual)
        if self.output_attention:
            return output, attn_weights, uncertainty
        return output, None

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="BayesianAttention",
    component_class=BayesianAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "dropout": 0.1,
        "prior_std": 1.0,
        "temperature": 1.0,
        "output_attention": False,
    },
)

 
