"""
Concrete Attention Implementations

This module provides concrete implementations of the BaseAttention interface
for different attention mechanisms used in time series forecasting.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
import math
import numpy as np

from ..base_interfaces import BaseAttention
from ..config_schemas import AttentionConfig

logger = logging.getLogger(__name__)


class MultiHeadAttention(BaseAttention):
    """
    Standard multi-head attention mechanism
    
    Implements the classical transformer attention with multiple heads
    for parallel attention computation.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.num_heads = getattr(config, 'num_heads', 8)
        self.dropout = config.dropout
        
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads
        
        # Linear projections
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"MultiHeadAttention initialized: heads={self.num_heads}, d_k={self.d_k}")
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        output, attention_weights = self._scaled_dot_product_attention(Q, K, V, attention_mask)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        output = self.dropout_layer(output)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Expand mask for multi-head attention
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get attention capabilities"""
        return {
            'type': 'multi_head_attention',
            'num_heads': self.num_heads,
            'supports_masking': True,
            'supports_self_attention': True,
            'supports_cross_attention': True
        }


class AutoCorrelationAttention(BaseAttention):
    """
    AutoCorrelation attention mechanism
    
    Uses autocorrelation to find period-based dependencies in time series.
    Particularly effective for seasonal patterns.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.factor = getattr(config, 'factor', 1)
        self.scale = getattr(config, 'scale', None)
        self.dropout = config.dropout
        
        # Linear projections
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"AutoCorrelationAttention initialized: factor={self.factor}")
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for autocorrelation attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask (optional)
            
        Returns:
            Tuple of (output, autocorrelation_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Compute autocorrelation
        autocorr_weights = self._autocorrelation(Q, K, seq_len)
        
        # Apply autocorrelation to values
        output = self._apply_autocorrelation(V, autocorr_weights)
        
        # Project output
        output = self.w_o(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        output = self.dropout_layer(output)
        
        return output, autocorr_weights
    
    def _autocorrelation(self, Q, K, seq_len):
        """Compute autocorrelation coefficients"""
        # FFT-based autocorrelation
        Q_fft = torch.fft.rfft(Q, dim=1)
        K_fft = torch.fft.rfft(K, dim=1)
        
        # Compute cross-correlation in frequency domain
        correlation = Q_fft * torch.conj(K_fft)
        autocorr = torch.fft.irfft(correlation, n=seq_len, dim=1)
        
        # Find top-k correlations
        k = min(self.factor, seq_len // 2)
        autocorr_values, autocorr_indices = torch.topk(autocorr.mean(dim=-1), k, dim=-1)
        
        # Normalize
        autocorr_weights = F.softmax(autocorr_values, dim=-1)
        
        return autocorr_weights, autocorr_indices
    
    def _apply_autocorrelation(self, V, autocorr_info):
        """Apply autocorrelation to values"""
        autocorr_weights, autocorr_indices = autocorr_info
        batch_size, seq_len, d_model = V.size()
        
        # Initialize output
        output = torch.zeros_like(V)
        
        # Apply weighted autocorrelation
        for i, (weights, indices) in enumerate(zip(autocorr_weights, autocorr_indices)):
            for weight, idx in zip(weights, indices):
                # Roll and weight
                rolled_V = torch.roll(V[i], shifts=int(idx.item()), dims=0)
                output[i] += weight.unsqueeze(-1) * rolled_V
        
        return output
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get attention capabilities"""
        return {
            'type': 'autocorrelation_attention',
            'factor': self.factor,
            'supports_seasonal_patterns': True,
            'supports_fft': True,
            'time_series_optimized': True
        }


class SparseAttention(BaseAttention):
    """
    Sparse attention mechanism for efficient long sequences
    
    Implements sparse attention patterns to reduce computational complexity
    while maintaining important dependencies.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.num_heads = getattr(config, 'num_heads', 8)
        self.sparsity_factor = getattr(config, 'sparsity_factor', 4)
        self.dropout = config.dropout
        
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads
        
        # Linear projections
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"SparseAttention initialized: heads={self.num_heads}, sparsity={self.sparsity_factor}")
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for sparse attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask (optional)
            
        Returns:
            Tuple of (output, sparse_attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create sparse attention pattern
        sparse_mask = self._create_sparse_mask(seq_len, batch_size)
        
        # Combine with provided mask
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            sparse_mask = sparse_mask & attention_mask
        
        # Sparse attention computation
        output, attention_weights = self._sparse_attention(Q, K, V, sparse_mask)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        output = self.dropout_layer(output)
        
        return output, attention_weights
    
    def _create_sparse_mask(self, seq_len, batch_size):
        """Create sparse attention mask"""
        # Local attention pattern
        local_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # Diagonal bands (local attention)
        for i in range(seq_len):
            start = max(0, i - self.sparsity_factor)
            end = min(seq_len, i + self.sparsity_factor + 1)
            local_mask[i, start:end] = True
        
        # Strided attention (global patterns)
        stride = max(1, seq_len // self.sparsity_factor)
        for i in range(0, seq_len, stride):
            local_mask[:, i] = True
            local_mask[i, :] = True
        
        # Expand for batch and heads
        device = next(self.parameters()).device
        sparse_mask = local_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        
        return sparse_mask.to(device)
    
    def _sparse_attention(self, Q, K, V, mask):
        """Compute sparse attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sparse mask
        scores = scores.masked_fill(~mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get attention capabilities"""
        return {
            'type': 'sparse_attention',
            'num_heads': self.num_heads,
            'sparsity_factor': self.sparsity_factor,
            'supports_long_sequences': True,
            'computational_complexity': 'O(n*sqrt(n))',
            'supports_masking': True
        }


class LogSparseAttention(BaseAttention):
    """
    LogSparse attention mechanism
    
    Implements logarithmic sparse attention for very long sequences
    with O(n log n) complexity.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.num_heads = getattr(config, 'num_heads', 8)
        self.dropout = config.dropout
        
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads
        
        # Linear projections
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"LogSparseAttention initialized: heads={self.num_heads}")
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for log-sparse attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create log-sparse pattern
        log_mask = self._create_log_sparse_mask(seq_len, batch_size)
        
        # Apply attention with log-sparse pattern
        output, attention_weights = self._log_sparse_attention(Q, K, V, log_mask, attention_mask)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        output = self.dropout_layer(output)
        
        return output, attention_weights
    
    def _create_log_sparse_mask(self, seq_len, batch_size):
        """Create logarithmic sparse attention mask"""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        for i in range(seq_len):
            # Local attention
            mask[i, max(0, i-1):min(seq_len, i+2)] = True
            
            # Logarithmic pattern
            for log_step in range(1, int(math.log2(seq_len)) + 1):
                step_size = 2 ** log_step
                # Forward jumps
                if i + step_size < seq_len:
                    mask[i, i + step_size] = True
                # Backward jumps
                if i - step_size >= 0:
                    mask[i, i - step_size] = True
        
        # Expand for batch and heads
        device = next(self.parameters()).device
        log_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        
        return log_mask.to(device)
    
    def _log_sparse_attention(self, Q, K, V, log_mask, attention_mask=None):
        """Compute log-sparse attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply log-sparse mask
        scores = scores.masked_fill(~log_mask, -1e9)
        
        # Apply additional mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get attention capabilities"""
        return {
            'type': 'log_sparse_attention',
            'num_heads': self.num_heads,
            'supports_very_long_sequences': True,
            'computational_complexity': 'O(n*log(n))',
            'logarithmic_pattern': True,
            'supports_masking': True
        }


class ProbSparseAttention(BaseAttention):
    """
    ProbSparse attention mechanism from Informer
    
    Uses probability-based sparse attention to focus on 
    important query-key pairs.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.num_heads = getattr(config, 'num_heads', 8)
        self.factor = getattr(config, 'factor', 5)
        self.dropout = config.dropout
        
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads
        
        # Linear projections
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"ProbSparseAttention initialized: heads={self.num_heads}, factor={self.factor}")
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for prob-sparse attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # ProbSparse attention
        output, attention_weights = self._prob_sparse_attention(Q, K, V, attention_mask)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        output = self.dropout_layer(output)
        
        return output, attention_weights
    
    def _prob_sparse_attention(self, Q, K, V, mask=None):
        """Compute probability sparse attention"""
        B, H, L, D = Q.shape
        _, _, S, _ = K.shape
        
        # Sample factor
        U_part = min(self.factor * int(np.ceil(np.log(L))), L)
        u = min(self.factor * int(np.ceil(np.log(S))), S)
        
        # Sample queries
        Q_sample = Q[:, :, torch.randperm(L)[:U_part], :]
        
        # Compute query-key scores for sampling
        K_sample = K[:, :, torch.randperm(S)[:u], :]
        scores_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) / math.sqrt(D)
        
        # Compute measurement
        M = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)
        M_top = M.topk(U_part, dim=-1)[1]
        
        # Select top queries
        Q_reduce = Q[:, :, M_top, :]  # [B, H, U_part, D]
        
        # Compute full attention for selected queries
        scores = torch.matmul(Q_reduce, K.transpose(-2, -1)) / math.sqrt(D)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Select corresponding mask
            mask_selected = mask[:, :, M_top, :]
            scores = scores.masked_fill(mask_selected == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        output_reduced = torch.matmul(attention_weights, V)  # [B, H, U_part, D]
        
        # Fill back to original size
        output = torch.zeros_like(Q)
        output[:, :, M_top, :] = output_reduced
        
        # Create full attention weights for return
        full_attention_weights = torch.zeros(B, H, L, S, device=Q.device)
        full_attention_weights[:, :, M_top, :] = attention_weights
        
        return output, full_attention_weights
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get attention capabilities"""
        return {
            'type': 'prob_sparse_attention',
            'num_heads': self.num_heads,
            'factor': self.factor,
            'supports_long_sequences': True,
            'probability_based_sampling': True,
            'from_informer': True,
            'supports_masking': True
        }