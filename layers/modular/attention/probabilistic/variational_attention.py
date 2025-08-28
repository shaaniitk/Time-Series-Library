import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..base import BaseAttention

class VariationalAttention(BaseAttention):
    """
    An attention mechanism that introduces stochasticity through variational
    inference. It learns a distribution (mean and variance) for the queries,
    keys, and values, and samples from them during training.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads

        # Linear layers to project to mean
        self.w_q_mu = nn.Linear(d_model, d_model)
        self.w_k_mu = nn.Linear(d_model, d_model)
        self.w_v_mu = nn.Linear(d_model, d_model)
        
        # Linear layers to project to log variance
        self.w_q_log_var = nn.Linear(d_model, d_model)
        self.w_k_log_var = nn.Linear(d_model, d_model)
        self.w_v_log_var = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Performs the reparameterization trick to allow for backpropagation
        through a random node.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use the mean
            return mu

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L, _ = queries.shape
        S = keys.shape[1]
        residual = queries

        # 1. Compute mean and log variance for Q, K, V
        q_mu = self.w_q_mu(queries)
        k_mu = self.w_k_mu(keys)
        v_mu = self.w_v_mu(values)
        
        q_log_var = self.w_q_log_var(queries)
        k_log_var = self.w_k_log_var(keys)
        v_log_var = self.w_v_log_var(values)
        
        # 2. Sample Q, K, V using the reparameterization trick
        Q_sampled = self._reparameterize(q_mu, q_log_var)
        K_sampled = self._reparameterize(k_mu, k_log_var)
        V_sampled = self._reparameterize(v_mu, v_log_var)
        
        # 3. Proceed with standard multi-head attention
        Q = Q_sampled.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = K_sampled.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = V_sampled.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
             if attn_mask.dim() == 3: attn_mask = attn_mask.unsqueeze(1)
             scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.w_o(context)
        
        output = self.layer_norm(self.dropout(output) + residual)
        
        return output, attention_weights