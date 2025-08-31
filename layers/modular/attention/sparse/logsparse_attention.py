import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..base import BaseAttention

class LogSparseAttention(BaseAttention):
    """
    Implements LogSparse attention with O(n*log(n)) complexity, suitable for
    very long sequences by attending to cells with logarithmically increasing distance.
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

    def _create_log_sparse_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Creates the logarithmic sparse attention mask."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            # Local attention (immediate neighbors)
            mask[i, max(0, i-1):min(seq_len, i+2)] = True
            
            # Logarithmic pattern without math.log2 for TorchScript
            step_size = 2
            while step_size < seq_len:
                if i + step_size < seq_len:
                    mask[i, i + step_size] = True
                if i - step_size >= 0:
                    mask[i, i - step_size] = True
                step_size = step_size * 2
        
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_heads, -1, -1)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len_q, _ = queries.shape
        seq_len_kv = keys.shape[1]
        residual = queries

        Q = self.w_q(queries).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(keys).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(values).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        log_mask = self._create_log_sparse_mask(seq_len_q, batch_size, queries.device)
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            log_mask = log_mask & attn_mask

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~log_mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)
        
        output = self.layer_norm(self.dropout(output) + residual)
        
        return output, attention_weights

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="LogSparseAttention",
    component_class=LogSparseAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "dropout": 0.1,
    },
)

 