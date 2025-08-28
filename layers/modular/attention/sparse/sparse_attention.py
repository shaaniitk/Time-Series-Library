import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..base import BaseAttention

class SparseAttention(BaseAttention):
    """
    Implements sparse attention patterns to reduce computational complexity
    from O(n^2) to O(n * sqrt(n)) by attending to local and strided fields.
    """
    def __init__(self, d_model: int, n_heads: int, sparsity_factor: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.sparsity_factor = sparsity_factor

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _create_sparse_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Creates the sparse attention mask with local and strided patterns."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # Local attention pattern (diagonal bands)
        for i in range(seq_len):
            start = max(0, i - self.sparsity_factor)
            end = min(seq_len, i + self.sparsity_factor + 1)
            mask[i, start:end] = True
        
        # Strided attention pattern (global patterns)
        stride = max(1, seq_len // self.sparsity_factor)
        for i in range(0, seq_len, stride):
            mask[:, i] = True
            mask[i, :] = True
            
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_heads, -1, -1)

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

        # Projections and reshape
        Q = self.w_q(queries).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(keys).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(values).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        # Create and apply sparse mask
        sparse_mask = self._create_sparse_mask(seq_len_q, batch_size, queries.device)
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            sparse_mask = sparse_mask & attn_mask

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~sparse_mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)
        
        output = self.layer_norm(self.dropout(output) + residual)
        
        return output, attention_weights