import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..base import BaseAttention

class MultiHeadAttention(BaseAttention):
    """
    Standard multi-head self-attention mechanism as described in
    "Attention Is All You Need".
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core scaled dot-product attention computation."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 3:  # [B, L, S]
                mask = mask.unsqueeze(1)  # [B, 1, L, S] for head broadcasting
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len_q, _ = queries.shape
        seq_len_kv = keys.shape[1]
        
        residual = queries

        # 1. Linear projections and split into heads
        Q = self.w_q(queries).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(keys).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(values).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled dot-product attention
        context, attention_weights = self._scaled_dot_product_attention(Q, K, V, attn_mask)
        
        # 3. Concatenate heads and final linear projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)
        
        # 4. Add & Norm
        output = self.layer_norm(self.dropout(output) + residual)
        
        return output, attention_weights


        # --- REGISTRATION ---
# This is the crucial part. By registering here, the component becomes
# available to both the model factory and the test suite simultaneously.
from ...core.registry import component_registry, ComponentFamily

component_registry.register(
    name="MultiHeadAttention",
    component_class=MultiHeadAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "dropout": 0.1
    }
)