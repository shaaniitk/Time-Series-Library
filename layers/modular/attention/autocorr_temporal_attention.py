import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class AutoCorrelationAttention(nn.Module):
    """
    Auto-Correlation Attention mechanism from Autoformer paper.
    
    This mechanism uses FFT-based auto-correlation to efficiently find
    periodic patterns in time series data, making it more suitable for
    temporal modeling than standard dot-product attention.
    """
    
    def __init__(self, d_model: Optional[int] = None, n_heads: Optional[int] = None, 
                 num_heads: Optional[int] = None, dropout: float = 0.1, 
                 factor: int = 1, scale: Optional[float] = None):
        """
        Args:
            d_model: Model dimension (optional for low-level utilities)
            n_heads: Number of attention heads (optional)
            num_heads: Alias for n_heads
            dropout: Dropout rate
            factor: Factor for selecting top-k correlations
            scale: Scaling factor for attention scores
        """
        super().__init__()
        
        # Support alias
        if n_heads is None and num_heads is not None:
            n_heads = num_heads
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        if self.d_model is not None and self.n_heads is not None:
            assert self.d_model % self.n_heads == 0
            self.d_k = self.d_model // self.n_heads
        else:
            self.d_k = None
        
        self.factor = factor
        self.scale = scale or (1.0 / np.sqrt(self.d_k) if self.d_k is not None else 1.0)
        
        # Linear projections for Q, K, V (only if full attention config is provided)
        if self.d_model is not None and self.n_heads is not None:
            self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
            self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
            self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
            self.w_o = nn.Linear(self.d_model, self.d_model)
        else:
            self.w_q = None
            self.w_k = None
            self.w_v = None
            self.w_o = None
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_top_k_autocorr(self, autocorr_values: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Utility to select top-k autocorrelation values and indices along the last dimension.
        Accepts shapes [..., L] and returns (..., k) tensors.
        """
        topk = torch.topk(autocorr_values, k=k, dim=-1, largest=True, sorted=True)
        return topk.values, topk.indices
        
    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        
        Supports both full-shape inputs [B, H, D, L] and simplified inputs [B, L, D]
        used in unit tests.
        """
        # Normalize inputs to [B, H, D, L]
        if values.dim() == 4:
            values_4 = values
        elif values.dim() == 3:  # [B, L, D]
            B, L, D = values.shape
            values_4 = values.permute(0, 2, 1).unsqueeze(1)  # [B, 1, D, L]
        else:
            raise ValueError(f"Unsupported values dim: {values.dim()}")
        
        B, H, D, L = values_4.shape
        
        # Prepare corr to [B, H, D, L]
        if corr.dim() == 4:
            corr_4 = corr
        elif corr.dim() == 3:
            # Could be [B, L, D] or [B, D, L]
            if corr.shape[-1] == L:
                # [B, D, L]
                corr_bdL = corr if corr.shape[1] == D else corr.permute(0, 2, 1)
            elif corr.shape[1] == L:
                # [B, L, D]
                corr_bdL = corr.permute(0, 2, 1)
            else:
                # Fallback: treat as [B, L, 1]
                corr_bdL = corr.permute(0, 2, 1) if corr.shape[1] != D else corr
            corr_4 = corr_bdL.unsqueeze(1)  # [B, 1, D, L]
        elif corr.dim() == 2:
            # Assume [B, L]; broadcast across D
            if corr.shape[-1] != L:
                raise ValueError("corr last dim must equal sequence length L")
            corr_4 = corr.unsqueeze(1).unsqueeze(1).repeat(1, 1, D, 1)  # [B,1,D,L]
        else:
            # As a last resort, build a uniform correlation
            corr_4 = torch.ones(B, 1, D, L, device=values_4.device, dtype=values_4.dtype)
        
        # Find the top k autocorrelations delays
        top_k = max(1, int(self.factor * np.log(max(L, 2))))
        mean_value = torch.mean(torch.mean(corr_4, dim=1), dim=0)  # [D, L]
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # [top_k]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # [D, top_k]
        # Update corr
        tmp_corr = torch.softmax(weights, dim=-1)  # [D, top_k]
        # Aggregation over delays
        tmp_values = values_4  # [B, H, D, L]
        delays_agg = torch.zeros_like(values_4).float()  # [B, H, D, L]
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), dims=-1)  # [B, H, D, L]
            weight_i = tmp_corr[:, i].view(1, 1, D, 1).to(values_4.device)  # [1,1,D,1]
            delays_agg = delays_agg + pattern * weight_i  # broadcast to [B,H,D,L]
        
        # Return to original shape
        if values.dim() == 4:
            return delays_agg
        else:  # values was [B, L, D]
            return delays_agg.squeeze(1).permute(0, 2, 1).contiguous()  # [B, L, D]
    
    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch, head, channel, length = values.shape
        # Index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # Find the top k autocorrelations delays
        top_k = max(1, int(self.factor * np.log(max(length, 2))))
        
        # Handle correlation tensor shape - ensure it matches expected dimensions
        if corr.dim() == 4:  # [B, H, D, L]
            mean_value = torch.mean(torch.mean(corr, dim=1), dim=0)  # [D, L] - same as training
        elif corr.dim() == 3:  # [B, D, L]
            mean_value = torch.mean(corr, dim=0)  # [D, L]
        else:  # [B, L] or other
            mean_value = torch.mean(corr, dim=0) if corr.dim() > 1 else corr  # [L]
            if mean_value.dim() == 1:
                mean_value = mean_value.unsqueeze(0).repeat(channel, 1)  # [D, L]
        
        # Get top-k delays - same as training method
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # [top_k]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # [D, top_k]
        
        # Update corr
        tmp_corr = torch.softmax(weights, dim=-1)  # [D, top_k]
        
        # Aggregation - use roll like training method for consistency
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), dims=-1)  # [B, H, D, L]
            weight_i = tmp_corr[:, i].view(1, 1, channel, 1).to(values.device)  # [1, 1, D, 1]
            delays_agg = delays_agg + pattern * weight_i  # broadcast to [B, H, D, L]
        
        return delays_agg
    
    def autocorrelation(self, queries: torch.Tensor, keys: torch.Tensor, 
                       values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute auto-correlation between queries and keys using FFT.
        
        Args:
            queries: Query tensor [B, H, L_q, D]
            keys: Key tensor [B, H, L_k, D] 
            values: Value tensor [B, H, L_v, D]
            
        Returns:
            Tuple of (aggregated_values, correlation_weights)
        """
        B, H, L_q, D = queries.shape
        _, _, L_k, _ = keys.shape
        _, _, L_v, _ = values.shape
        
        if L_k > L_q:
            zeros = torch.zeros_like(queries[:, :, :(L_k - L_q), :]).float()
            queries = torch.cat([queries, zeros], dim=2)
            L_q = L_k
            
        if L_q > L_k:
            zeros = torch.zeros_like(keys[:, :, :(L_q - L_k), :]).float()
            keys = torch.cat([keys, zeros], dim=2)
            values = torch.cat([values, zeros], dim=2)
            L_k = L_q
            L_v = L_q
            
        # Compute auto-correlation using FFT
        q_fft = torch.fft.rfft(queries.permute(0, 1, 3, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 1, 3, 2).contiguous(), dim=-1)
        
        # Auto-correlation in frequency domain
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        
        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 1, 3, 2).contiguous(), 
                                           corr).permute(0, 1, 3, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 1, 3, 2).contiguous(), 
                                            corr).permute(0, 1, 3, 2)
            
        return V.contiguous(), corr
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Auto-Correlation Attention.
        
        Args:
            queries: Query tensor [B, L_q, D]
            keys: Key tensor [B, L_k, D]
            values: Value tensor [B, L_v, D]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if self.w_q is None:
            raise ValueError("AutoCorrelationAttention was initialized without d_model/num_heads; forward is unavailable.")
        
        B, L_q, _ = queries.shape
        _, L_k, _ = keys.shape
        _, L_v, _ = values.shape
        
        # Linear projections
        Q = self.w_q(queries)
        K = self.w_k(keys)
        V = self.w_v(values)
        
        # Reshape for multi-head attention
        Q = Q.view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_q, D_k]
        K = K.view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_k, D_k]
        V = V.view(B, L_v, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_v, D_k]
        
        # Apply auto-correlation
        out, attn = self.autocorrelation(Q, K, V)
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.w_o(out)
        
        return self.dropout(out), attn

class AutoCorrTemporalAttention(nn.Module):
    """
    Auto-Correlation based Temporal Attention for SOTA PGAT.
    
    Replaces standard MultiheadAttention with Auto-Correlation mechanism
    for more efficient temporal pattern discovery in time series.
    """
    
    def __init__(self, d_model: int, num_heads: Optional[int] = None, n_heads: Optional[int] = None, 
                 dropout: float = 0.1, factor: int = 1, activation: str = 'gelu'):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads (alias n_heads)
            dropout: Dropout rate
            factor: Factor for auto-correlation computation
            activation: Activation function ('gelu' or 'relu')
        """
        super().__init__()
        
        heads = num_heads if num_heads is not None else n_heads
        if heads is None:
            raise ValueError("num_heads (or n_heads) must be provided")
        
        self.d_model = d_model
        self.num_heads = heads
        self.factor = factor
        
        self.autocorr_attention = AutoCorrelationAttention(
            d_model=d_model,
            n_heads=heads,
            dropout=dropout,
            factor=factor
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass compatible with tests: (queries, keys, values, history) -> output.
        
        Args:
            queries: [B, L_q, D] (query)
            keys: [B, L_k, D] (key)
            values: [B, L_v, D] (value)
            history: Optional additional context (unused, kept for API compatibility)
            
        Returns:
            output tensor [B, L_q, D]
        """
        attn_out, _attn_weights = self.autocorr_attention(
            queries=queries,
            keys=keys,
            values=values
        )
        
        # First residual connection and layer norm
        x = self.norm1(queries + self.dropout(attn_out))
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        
        # Second residual connection and layer norm
        output = self.norm2(x + ffn_out)
        
        return output
    
    def get_attention_map(self, queries: torch.Tensor,
                          keys: torch.Tensor,
                          values: torch.Tensor,
                          history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention correlation map for visualization.
        
        Args:
            queries: [B, L_q, D]
            keys: [B, L_k, D]
            values: [B, L_v, D]
            history: Optional, unused.
            
        Returns:
            Attention correlation tensor [B, H, D, L]
        """
        with torch.no_grad():
            _, attn_weights = self.autocorr_attention(
                queries=queries,
                keys=keys,
                values=values
            )
            return attn_weights