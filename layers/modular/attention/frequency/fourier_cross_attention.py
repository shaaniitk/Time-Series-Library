import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..base import BaseAttention

class FourierCrossAttention(BaseAttention):
    """
    A cross-attention variant that first processes queries and keys in the
    frequency domain with learnable filters before mixing them in the time domain
    using standard scaled dot-product attention.
    """
    def __init__(self, d_model: int, n_heads: int, seq_len_q: int, seq_len_kv: int, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        
        self.num_freqs_q = seq_len_q // 2 + 1
        self.num_freqs_kv = seq_len_kv // 2 + 1

        # Learnable complex weights in the frequency domain
        # Shape aligns with rfft outputs: [B, H, F, E] -> broadcast with [1, H, F, E]
        self.weights_q = nn.Parameter(
            torch.rand(1, n_heads, self.num_freqs_q, d_model // n_heads, dtype=torch.cfloat)
        )
        self.weights_kv = nn.Parameter(
            torch.rand(1, n_heads, self.num_freqs_kv, d_model // n_heads, dtype=torch.cfloat)
        )

        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L_q, D = queries.shape
        L_kv = keys.shape[1]
        H = self.n_heads
        E = D // H
        residual = queries

        # --- Frequency Domain Processing ---
        q_ft = torch.fft.rfft(queries.view(B, L_q, H, E).permute(0, 2, 1, 3), dim=2)
        k_ft = torch.fft.rfft(keys.view(B, L_kv, H, E).permute(0, 2, 1, 3), dim=2)

        # Apply learnable filters
        q_filtered_ft = q_ft * self.weights_q  # [B,H,Fq,E] * [1,1,Fq,E]
        k_filtered_ft = k_ft * self.weights_kv  # [B,H,Fk,E] * [1,1,Fk,E]

        # Inverse FFT to get processed Q and K
        Q_processed = torch.fft.irfft(q_filtered_ft, n=L_q, dim=2)
        K_processed = torch.fft.irfft(k_filtered_ft, n=L_kv, dim=2)

        # --- Standard Attention with Processed Q/K ---
        V = self.w_v(values).view(B, L_kv, H, E).permute(0, 2, 1, 3)

        scores = torch.matmul(Q_processed, K_processed.transpose(-2, -1)) / math.sqrt(E)
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L_q, D)

        output = self.w_o(context)
        output = self.layer_norm(self.dropout(output) + residual)

        return output, attention_weights

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="FourierCrossAttention",
    component_class=FourierCrossAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
    "seq_len_q": 32,
    "seq_len_kv": 32,
        "dropout": 0.1,
    },
)