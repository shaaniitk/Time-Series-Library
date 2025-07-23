"""
CLEAN ATTENTION COMPONENTS
Self-contained attention implementations for the new modular framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseAttention
from ..config_schemas import ComponentConfig

# Import AutoCorrelationLayer from layers module for backward compatibility
try:
    from .layers import AutoCorrelationLayer
    AUTOCORRELATION_LAYER_AVAILABLE = True
except ImportError:
    AUTOCORRELATION_LAYER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultiHeadAttention(BaseAttention):
    """Standard multi-head attention mechanism"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "multihead"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply multi-head attention mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        
        Q = self.w_qs(queries).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_ks(keys).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_vs(values).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
            
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)
        
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.fc(output)
        
        return output, attn


class AutoCorrelationAttention(BaseAttention):
    """Clean implementation of AutoCorrelation attention"""
    
    def __init__(self, d_model=512, n_heads=8, factor=1, dropout=0.1):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.factor = factor
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "autocorrelation"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply autocorrelation attention mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def time_delay_agg_training(self, values, corr):
        """Aggregation in time delay"""
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        
        return delays_agg
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        
        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        
        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr)
        else:
            # simplified for inference
            V = values.permute(0, 2, 3, 1).contiguous()
        
        V = V.permute(0, 3, 1, 2).contiguous()
        return V.contiguous(), corr


class SparseAttention(BaseAttention):
    """Sparse attention with configurable sparsity pattern"""
    
    def __init__(self, d_model=512, n_heads=8, sparsity_factor=0.1, dropout=0.1):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.sparsity_factor = sparsity_factor
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity_factor = sparsity_factor
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "sparse"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply sparse attention mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        d_k = self.d_model // self.n_heads
        
        Q = self.w_qs(queries).view(B, L, self.n_heads, d_k).transpose(1, 2)
        K = self.w_ks(keys).view(B, L, self.n_heads, d_k).transpose(1, 2)
        V = self.w_vs(values).view(B, L, self.n_heads, d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply sparsity mask
        k = int(L * self.sparsity_factor)
        if k > 0:
            top_k = torch.topk(scores, k, dim=-1)[1]
            sparse_mask = torch.zeros_like(scores)
            sparse_mask.scatter_(-1, top_k, 1)
            scores = scores * sparse_mask - 1e9 * (1 - sparse_mask)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
            
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)
        
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.fc(output)
        
        return output, attn


class FourierAttention(BaseAttention):
    """Fourier-based attention for periodic patterns"""
    
    def __init__(self, d_model=512, n_heads=8, seq_len=96, modes=32, dropout=0.1):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.seq_len = seq_len
                self.modes = modes
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        # Fourier weights
        self.fourier_weight = nn.Parameter(torch.randn(d_model, modes, dtype=torch.cfloat) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "fourier"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply fourier attention mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        
        # Standard attention computation
        Q = self.w_qs(queries)
        K = self.w_ks(keys)
        V = self.w_vs(values)
        
        # Fourier transform
        x_fft = torch.fft.rfft(queries, dim=1)
        
        # Apply Fourier weights
        out_fft = torch.zeros_like(x_fft)
        modes = min(self.modes, x_fft.size(1))
        out_fft[:, :modes] = x_fft[:, :modes] * self.fourier_weight[:, :modes].unsqueeze(0)
        
        # Inverse Fourier transform
        fourier_result = torch.fft.irfft(out_fft, n=L, dim=1)
        
        # Combine with standard attention
        d_k = self.d_model // self.n_heads
        Q = Q.view(B, L, self.n_heads, d_k).transpose(1, 2)
        K = K.view(B, L, self.n_heads, d_k).transpose(1, 2)
        V = V.view(B, L, self.n_heads, d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        # Combine Fourier and attention results
        output = self.fc(attn_output + fourier_result)
        
        return output, attn


# Registry for clean attention components
ATTENTION_REGISTRY = {
    'multihead': MultiHeadAttention,
    'autocorrelation': AutoCorrelationAttention,
    'sparse': SparseAttention,
    'fourier': FourierAttention,
}

# Add AutoCorrelationLayer for backward compatibility if available
if AUTOCORRELATION_LAYER_AVAILABLE:
    ATTENTION_REGISTRY['autocorrelation_layer'] = AutoCorrelationLayer


def get_attention_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get attention component by name"""
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention component: {name}")
    
    component_class = ATTENTION_REGISTRY[name]
    
    if config is not None:
        # Use config parameters
        params = {
            'd_model': getattr(config, 'd_model', 512),
            'dropout': getattr(config, 'dropout', 0.1),
            **getattr(config, 'custom_params', {}),
            **kwargs
        }
    else:
        params = kwargs
    
    return component_class(**params)


def register_attention_components(registry):
    """Register all attention components with the registry"""
    for name, component_class in ATTENTION_REGISTRY.items():
        registry.register('attention', name, component_class)
    
    logger.info(f"Registered {len(ATTENTION_REGISTRY)} clean attention components")


def list_attention_components():
    """List all available attention components"""
    return list(ATTENTION_REGISTRY.keys())
