"""
AdaptiveMixture: Adaptive mixture of experts for different time series patterns.
"""
import logging
import torch
import torch.nn as nn
from ..base import BaseMixtureAdapter

logger = logging.getLogger(__name__)

class AdaptiveMixture(BaseMixtureAdapter):
    """
    Adaptive mixture of experts for different time series patterns.
    Adapted to the BaseAttention interface.
    """
    def __init__(self, d_model, n_heads=None, mixture_components=4, gate_hidden_dim=None, dropout=0.1, temperature=1.0):
        super().__init__()
        logger.info(f"Initializing AdaptiveMixture: components={mixture_components}")
        self.num_experts = mixture_components
        self.d_model = d_model
        self.output_dim_multiplier = 1
        if gate_hidden_dim is None:
            gate_hidden_dim = d_model // 2
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(self.num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

    def forward(self, query, key, value, attn_mask=None):
        x = query
        gate_weights = self.gate(x) / self.temperature
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        return output, None
