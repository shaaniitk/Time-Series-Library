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
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 factor: int = 1, scale: Optional[float] = None):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            factor: Factor for selecting top-k correlations
            scale: Scaling factor for attention scores
        """
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.scale = scale or 1.0 / np.sqrt(self.d_k)
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head, channel, length = values.shape
        # Find the top k autocorrelations delays
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=0)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # Update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # Aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(2).repeat(1, channel, length))
        return delays_agg
    
    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch, head, channel, length = values.shape
        # Index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # Find the top k autocorrelations delays
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(corr, dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # Update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # Aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, :, i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:, :, i].unsqueeze(-1))
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
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 factor: int = 1, activation: str = 'gelu'):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            factor: Factor for auto-correlation computation
            activation: Activation function ('gelu' or 'relu')
        """
        super().__init__()
        
        self.autocorr_attention = AutoCorrelationAttention(
            d_model=d_model,
            n_heads=n_heads,
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
        
    def forward(self, current_target_state: torch.Tensor, 
                historical_event_messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with existing TemporalAttention interface.
        
        Args:
            current_target_state: Current target state [B, 1, D] (query)
            historical_event_messages: Historical events [B, L, D] (key/value)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Auto-correlation attention
        attn_out, attn_weights = self.autocorr_attention(
            queries=current_target_state,
            keys=historical_event_messages,
            values=historical_event_messages
        )
        
        # First residual connection and layer norm
        x = self.norm1(current_target_state + self.dropout(attn_out))
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        
        # Second residual connection and layer norm
        output = self.norm2(x + ffn_out)
        
        return output, attn_weights
    
    def get_attention_map(self, current_target_state: torch.Tensor,
                         historical_event_messages: torch.Tensor) -> torch.Tensor:
        """
        Get attention correlation map for visualization.
        
        Args:
            current_target_state: Current target state [B, 1, D]
            historical_event_messages: Historical events [B, L, D]
            
        Returns:
            Attention correlation tensor [B, H, D, L]
        """
        with torch.no_grad():
            _, attn_weights = self.forward(current_target_state, historical_event_messages)
            return attn_weights