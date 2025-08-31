import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from ..base import BaseAttention

class AutoCorrelationAttention(BaseAttention):
    """
    Enhanced AutoCorrelation with adaptive top-k selection and multi-scale analysis.
    This mechanism finds period-based dependencies by computing the autocorrelation
    of the time series in the frequency domain via FFT.
    """
    def __init__(self, d_model: int, n_heads: int, factor: int = 1, dropout: float = 0.1, 
                 scales: List[int] = [1, 2, 4], **kwargs):
        super().__init__(d_model, n_heads)
        
        self.factor = factor
        self.scales = scales
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def _resize_and_fft(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Helper to resize, then compute RFFT."""
        if x.size(2) != target_len:
            # [B, H, L, E] -> [B*H, E, L]
            x_reshaped = x.permute(0, 1, 3, 2).contiguous().view(-1, x.size(3), x.size(2))
            x_resized = F.interpolate(x_reshaped, size=target_len, mode='linear', align_corners=False)
            # [B*H, E, L] -> [B, H, E, L] -> [B, H, L, E]
            x = x_resized.view(x.size(0), x.size(1), x.size(3), target_len).permute(0, 1, 3, 2)
        
        return torch.fft.rfft(x, n=target_len, dim=2)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L, D = queries.shape
        H = self.n_heads
        E = D // H

        # Projections
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # Reshape for multi-head processing
        queries = queries.view(B, L, H, E).permute(0, 2, 1, 3) # [B, H, L, E]
        keys = keys.view(B, keys.size(1), H, E).permute(0, 2, 1, 3)
        values = values.view(B, values.size(1), H, E).permute(0, 2, 1, 3)

        # --- Multi-scale correlation ---
        correlations = []
        for scale in self.scales:
            target_len = max(L // scale, 1)
            q_fft = self._resize_and_fft(queries, target_len)
            k_fft = self._resize_and_fft(keys, target_len)
            
            # Correlation in frequency domain
            corr_fft = q_fft * torch.conj(k_fft)
            corr_time = torch.fft.irfft(corr_fft, n=target_len, dim=2)
            
            if corr_time.size(2) != L: # Upsample back if needed
                # [B,H,L,E] -> [B*H*E, 1, L] for 1D linear interpolation
                b, h, l, e = corr_time.shape
                corr_3d = corr_time.permute(0, 1, 3, 2).contiguous().view(b * h * e, 1, l)
                corr_up = F.interpolate(corr_3d, size=L, mode='linear', align_corners=False)
                corr_time = corr_up.view(b, h, e, L).permute(0, 1, 3, 2)
            correlations.append(corr_time)
        
        # Weighted combination of correlations
        weights = F.softmax(self.scale_weights, dim=0)
        # TorchScript-friendly accumulation
        correlation = correlations[0] * weights[0]
        for idx in range(1, len(correlations)):
            correlation = correlation + correlations[idx] * weights[idx]

        # --- Time delay aggregation ---
        k = max(int(self.factor * math.log(L)), 1)
        mean_corr = torch.mean(correlation, dim=(1, 3)) # [B, L]
        
        # Find top-k delays (indices) and their energies (values)
        top_energies, top_indices = torch.topk(mean_corr, k, dim=-1)
        delay_weights = F.softmax(top_energies, dim=-1) # [B, k]
        
        # Aggregate values based on top delays
        output = torch.zeros_like(values)
        for i in range(k):
            indices = top_indices[:, i] # [B]
            weight = delay_weights[:, i].view(B, 1, 1, 1) # [B, 1, 1, 1]
            
            # Roll values for each batch item according to its specific top delay
            rolled_V = torch.stack([torch.roll(values[b], shifts=-int(indices[b].item()), dims=1) for b in range(B)])
            output += weight * rolled_V
        
        # --- Final projection ---
        output = output.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        output = self.out_projection(output)
        
        return output, None # Returning weights is non-trivial for this mechanism

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="AutoCorrelationAttention",
    component_class=AutoCorrelationAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "factor": 1,
        "dropout": 0.1,
        "scales": [1, 2],
    },
)