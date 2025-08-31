import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


from ..base import BaseAttention

class ProbSparseAttention(BaseAttention):
    @staticmethod
    def _ilog_plus_one(x: int) -> int:
        """
        Approximate ceil(log(x)) using integer doubling steps
        """
        steps = 0
        val = 1
        while val < x:
            val = val * 2
            steps += 1
        return steps if steps > 0 else 1
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
        """Samples the most important queries based on their similarity to a random subset of keys."""
        B, H, L, D = Q.shape
        _, _, S, _ = K.shape
        n_top = min(self.factor * self._ilog_plus_one(int(L)), int(L))
        n_keys_sample = min(self.factor * self._ilog_plus_one(int(S)), int(S))
        K_sample = K[:, :, torch.randperm(S)[:n_keys_sample], :]
        scores_sample = torch.matmul(Q, K_sample.transpose(-2, -1))
        M = scores_sample.max(dim=-1)[0] - torch.mean(scores_sample, dim=-1)
        _, top_indices = M.topk(n_top, dim=-1, largest=True)
        Q_reduced = torch.gather(Q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
        return Q_reduced, top_indices

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = queries.shape
        S = keys.shape[1]
        residual = queries

        Q = self.w_q(queries).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(keys).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(values).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Select top queries using ProbSparse sampling
        Q_reduced, top_indices = self._prob_sparse_sampling(Q, K)
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

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="ProbSparseAttention",
    component_class=ProbSparseAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "factor": 3,
        "dropout": 0.1,
    },
)

 