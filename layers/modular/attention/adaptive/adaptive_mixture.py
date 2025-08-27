"""Adaptive mixture-of-experts split from adaptive_components."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAttention
from layers.modular.core.logger import logger


class AdaptiveMixture(BaseAttention):
    """Adaptive mixture of experts for time series patterns."""

    def __init__(
        self,
        d_model: int,
        n_heads: int | None = None,
        mixture_components: int = 4,
        gate_hidden_dim: int | None = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        logger.info("Initializing AdaptiveMixture: components=%s", mixture_components)
        self.num_experts = mixture_components
        self.d_model = d_model
        self.output_dim_multiplier = 1
        if gate_hidden_dim is None:
            gate_hidden_dim = d_model // 2
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_experts),
            nn.Softmax(dim=-1),
        )
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

    def forward(
        self, query: torch.Tensor, key: torch.Tensor | None, value: torch.Tensor | None, attn_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, None]:
        x = query
        gate_weights = self.gate(x) / self.temperature
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        return self.dropout(output), None


__all__ = ["AdaptiveMixture"]
