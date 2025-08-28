import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

from ..base import BaseAttention

class ProbSparseAttention(BaseAttention):
    """
    ProbSparse attention from the Informer paper, which reduces complexity by
    only calculating attention scores for a subset of the most "important" queries.
    """
    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.factor = factor

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _prob_sparse_sampling(self, Q, K):
        """
        Samples the most important queries based on their similarity to a
        random subset of keys.
        """
        B, H, L, D = Q.shape
        _, _, S, _ = K.shape
        
        # Determine number of queries and keys to sample
        n_top = min(self.factor * int(np.ceil(np.log(L))), L)
        n_keys_sample = min(self.factor * int(np.ceil(np.log(S))), S)
        
        # Sample keys for measurement
        K_sample = K[:, :, torch.randperm(S)[:n_keys_sample], :]
        
        # Compute scores for all queries against the sampled keys
        scores_sample = torch.matmul(Q, K_sample.transpose(-2, -1))
        
        # Calculate the measurement M(q, K)
        M = scores_sample.max(dim=-1)[0] - torch.mean(scores_sample, dim=-1)
        
        # Select the top queries based on the measurement
        _, top_indices = M.topk(n_top, dim=-1, largest=True)
        
        # Gather the selected queries
        Q_reduced = torch.gather(Q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
        
        return Q_reduced, top_indices

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

        Q = self.w_q(queries).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(keys).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(values).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Select top queries using ProbSparse sampling
        Q_reduced, top_indices = self._prob_sparse_sampling(Q, K)
        
        # Compute full attention for the selected queries
        scores = torch.matmul(Q_reduced, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if attn_mask is not None:
             raise NotImplementedError("attn_mask is not yet supported for ProbSparseAttention")

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_reduced = torch.matmul(attention_weights, V)
        
        # Reconstruct full output tensor
        output = torch.zeros_like(Q)
        output.scatter_(2, top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k), context_reduced)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.w_o(output)
        
        output = self.layer_norm(self.dropout(output) + residual)
        
        # Note: Returning full attention weights is computationally expensive and
        # defeats the purpose of sparse attention. Returning None is standard practice.
        return output, None