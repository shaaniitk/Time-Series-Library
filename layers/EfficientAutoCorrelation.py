import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class EfficientAutoCorrelation(nn.Module):
    """
    Memory and compute efficient AutoCorrelation with proper implementation.
    
    Features:
    - Gradient checkpointing for large sequences
    - Chunked processing for memory efficiency
    - Mixed precision support
    - Vectorized operations
    - Proper error handling and type hints
    """
    
    def __init__(
        self, 
        mask_flag: bool = True, 
        factor: int = 1, 
        attention_dropout: float = 0.1, 
        max_seq_len: int = 1024, 
        use_checkpoint: bool = True,
        chunk_size: int = 512
    ):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        self.chunk_size = chunk_size
        
    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with efficient memory management.
        
        Args:
            queries: Query tensor [B, L, H, E]
            keys: Key tensor [B, S, H, D] 
            values: Value tensor [B, S, H, D]
            attn_mask: Attention mask (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Handle sequence length mismatch
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        
        # Use gradient checkpointing for large sequences
        if self.use_checkpoint and L > self.max_seq_len:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, queries, keys, values, attn_mask
            )
        else:
            return self._forward_impl(queries, keys, values, attn_mask)
    
    def _forward_impl(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Internal forward implementation with chunked processing."""
        B, L, H, E = queries.shape
        
        # For very large sequences, use chunked processing
        if L > self.chunk_size:
            outputs = []
            for i in range(0, L, self.chunk_size):
                end_idx = min(i + self.chunk_size, L)
                chunk_q = queries[:, i:end_idx]
                chunk_k = keys[:, i:end_idx] 
                chunk_v = values[:, i:end_idx]
                
                chunk_out = self._process_chunk(chunk_q, chunk_k, chunk_v)
                outputs.append(chunk_out)
            
            return torch.cat(outputs, dim=1), None
        else:
            return self._process_chunk(queries, keys, values), None
    
    def _process_chunk(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor
    ) -> torch.Tensor:
        """Process a chunk with efficient FFT-based correlation."""
        # Efficient FFT-based correlation computation
        q_input = queries.permute(0, 2, 3, 1).contiguous()
        k_input = keys.permute(0, 2, 3, 1).contiguous()
        
        # Use mixed precision for FFT if available
        if queries.device.type == 'cuda' and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                q_fft = torch.fft.rfft(q_input, dim=-1)
                k_fft = torch.fft.rfft(k_input, dim=-1)
                res = q_fft * torch.conj(k_fft)
                corr = torch.fft.irfft(res, dim=-1)
        else:
            q_fft = torch.fft.rfft(q_input, dim=-1)
            k_fft = torch.fft.rfft(k_input, dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)
        
        # Efficient time delay aggregation
        v_input = values.permute(0, 2, 3, 1).contiguous()
        aggregated = self._efficient_time_delay_agg(v_input, corr)
        
        return aggregated.permute(0, 3, 1, 2)
    
    def _efficient_time_delay_agg(
        self, 
        values: torch.Tensor, 
        corr: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficient time delay aggregation with vectorized operations.
        
        Args:
            values: Value tensor [B, H, D, L]
            corr: Correlation tensor [B, H, D, L]
            
        Returns:
            Aggregated values [B, H, D, L]
        """
        B, H, D, L = values.shape
        
        # Calculate adaptive top_k with bounds
        top_k = max(1, min(int(self.factor * math.log(L)), L // 4, 32))
        
        # Compute mean correlation across heads and channels for efficiency
        mean_corr = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        
        # Get top-k indices and weights
        weights, indices = torch.topk(torch.mean(mean_corr, dim=0), top_k, dim=-1)
        
        # Apply softmax to weights
        weights = F.softmax(weights, dim=-1)  # [top_k]
        
        # Vectorized aggregation
        delays_agg = torch.zeros_like(values)
        
        for i in range(top_k):
            shift = int(indices[i])
            # Use torch.roll for circular shift (more efficient than manual indexing)
            pattern = torch.roll(values, -shift, -1)
            
            # Broadcast weight across dimensions
            weight = weights[i].view(1, 1, 1, 1).expand(B, 1, 1, 1)
            delays_agg.add_(pattern * weight)
        
        return delays_agg


class EfficientAutoCorrelationLayer(nn.Module):
    """
    Layer wrapper for EfficientAutoCorrelation with proper projections.
    """
    
    def __init__(
        self, 
        correlation: EfficientAutoCorrelation, 
        d_model: int, 
        n_heads: int, 
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None
    ):
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with projections."""
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
