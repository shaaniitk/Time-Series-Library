import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..base import BaseAttention

class AdaptiveMixture(BaseAttention):
    """
    An adaptive Mixture-of-Experts (MoE) layer. It processes the input through
    multiple parallel 'expert' networks and uses a gating network to compute a
    weighted sum of their outputs. It does not perform attention in the
    traditional sense but fits the interface by operating on the queries tensor.
    """
    def __init__(self, d_model: int, n_heads: int, num_experts: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        self.num_experts = num_experts

        # A list of 'expert' neural networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            )
            for _ in range(self.num_experts)
        ])
        
        # The gating network that decides the weight of each expert
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.num_experts)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L, D = queries.shape
        residual = queries

        # 1. Compute expert outputs
        expert_outputs = [expert(queries) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=-1) # [B, L, D, num_experts]
        
        # 2. Compute gating weights
        gate_logits = self.gate(queries) # [B, L, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-2) # [B, L, 1, num_experts]
        
        # 3. Compute the weighted sum of expert outputs
        # (B, L, D, num_experts) * (B, L, 1, num_experts) -> sum -> (B, L, D)
        mixed_output = torch.sum(expert_outputs_stacked * gate_weights, dim=-1)
        
        output = self.layer_norm(self.dropout(mixed_output) + residual)
        
        return output, None