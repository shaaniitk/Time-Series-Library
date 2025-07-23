"""
UNIFIED ATTENTION COMPONENTS - SINGLE SOURCE OF TRUTH
All attention mechanisms consolidated into one file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseAttention
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CORE ATTENTION COMPONENTS
# =============================================================================

class MultiHeadAttention(BaseAttention):
    """Standard multi-head attention mechanism"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, **kwargs):
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
        
        self.dropout_layer = nn.Dropout(dropout)
    
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
            
        attn = self.dropout_layer(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)
        
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.fc(output)
        
        return output, attn


class AutoCorrelationAttention(BaseAttention):
    """Clean implementation of AutoCorrelation attention"""
    
    def __init__(self, d_model=512, n_heads=8, factor=1, dropout=0.1, **kwargs):
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
        
        self.dropout_layer = nn.Dropout(dropout)
    
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
    
    def __init__(self, d_model=512, n_heads=8, sparsity_factor=0.1, dropout=0.1, **kwargs):
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
        
        self.dropout_layer = nn.Dropout(dropout)
    
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
            
        attn = self.dropout_layer(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)
        
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.fc(output)
        
        return output, attn


class FourierAttention(BaseAttention):
    """Fourier-based attention for periodic patterns"""
    
    def __init__(self, d_model=512, n_heads=8, seq_len=96, modes=32, dropout=0.1, **kwargs):
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
        
        self.dropout_layer = nn.Dropout(dropout)
    
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


class WaveletAttention(BaseAttention):
    """Wavelet-based attention for multi-scale analysis"""
    
    def __init__(self, d_model=512, n_heads=8, levels=3, dropout=0.1, **kwargs):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.levels = levels
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        
        # Learnable wavelet filters
        self.wavelet_filters = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_model, 4))  # 4-tap wavelet
            for _ in range(self.levels)
        ])
        
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout_layer = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "wavelet"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply wavelet attention mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def _wavelet_transform(self, x, level):
        """Simple learnable wavelet transform"""
        B, L, D = x.shape
        filter_weights = self.wavelet_filters[level]
        
        # Apply 1D convolution as wavelet transform
        x_conv = F.conv1d(x.transpose(1, 2), filter_weights.unsqueeze(1), 
                         padding=2, groups=D)
        
        # Downsample
        x_down = x_conv[:, :, ::2]
        return x_down.transpose(1, 2)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # Multi-scale wavelet decomposition
        scale_features = []
        current_q = queries
        
        for level in range(self.levels):
            if current_q.shape[1] > 4:  # Minimum sequence length
                current_q = self._wavelet_transform(current_q, level)
                scale_features.append(current_q)
        
        # Process largest scale
        if scale_features:
            processed = scale_features[-1]
            
            qkv = self.qkv(processed).reshape(processed.shape[0], processed.shape[1], 
                                            3, self.n_heads, D // self.n_heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            
            scale = math.sqrt(D // self.n_heads)
            attn_scores = (q @ k.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)
            
            out = (attn_weights @ v).transpose(1, 2).reshape(processed.shape[0], 
                                                           processed.shape[1], D)
            
            # Interpolate back to original length
            out = F.interpolate(out.transpose(1, 2), size=L, 
                              mode='linear', align_corners=False).transpose(1, 2)
        else:
            # Fallback to standard attention
            qkv = self.qkv(queries).reshape(B, L, 3, self.n_heads, D // self.n_heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            
            scale = math.sqrt(D // self.n_heads)
            attn_scores = (q @ k.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights


class BayesianAttention(BaseAttention):
    """Bayesian attention with uncertainty quantification"""
    
    def __init__(self, d_model=512, n_heads=8, prior_std=1.0, dropout=0.1, **kwargs):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.prior_std = prior_std
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.prior_std = prior_std
        
        # Bayesian linear layers
        self.q_mean = nn.Linear(self.d_model, self.d_model)
        self.q_logvar = nn.Linear(self.d_model, self.d_model)
        self.k_mean = nn.Linear(self.d_model, self.d_model)
        self.k_logvar = nn.Linear(self.d_model, self.d_model)
        self.v_mean = nn.Linear(self.d_model, self.d_model)
        self.v_logvar = nn.Linear(self.d_model, self.d_model)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model)
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "bayesian"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply bayesian attention mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def _reparameterize(self, mean, logvar):
        """Reparameterization trick for Bayesian sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # Bayesian projections
        q_mean = self.q_mean(queries)
        q_logvar = self.q_logvar(queries)
        q = self._reparameterize(q_mean, q_logvar)
        
        k_mean = self.k_mean(keys)
        k_logvar = self.k_logvar(keys)
        k = self._reparameterize(k_mean, k_logvar)
        
        v_mean = self.v_mean(values)
        v_logvar = self.v_logvar(values)
        v = self._reparameterize(v_mean, v_logvar)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        
        # Attention computation
        scale = math.sqrt(D // self.n_heads)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights


class AdaptiveAttention(BaseAttention):
    """Adaptive attention with dynamic parameter selection"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, **kwargs):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Adaptive components (simplified - no circular imports)
        self.multihead = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single gate for now
            nn.Sigmoid()
        )
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "adaptive"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply adaptive attention mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # Compute gating weights
        gate_input = queries.mean(dim=1)  # [B, D]
        gate_weight = self.gate(gate_input)  # [B, 1]
        
        # Apply multihead attention
        out, attn = self.multihead(queries, keys, values, attn_mask)
        
        # Apply gating
        output = gate_weight.unsqueeze(1) * out + (1 - gate_weight.unsqueeze(1)) * queries
        
        return output, attn


# =============================================================================
# AUTOCORRELATION LAYER (Complete Layer Implementation)
# =============================================================================

class AutoCorrelationLayer(BaseAttention):
    """Complete AutoCorrelation layer with normalization, residuals, and feed-forward"""
    
    def __init__(self, d_model=512, n_heads=8, factor=1, d_ff=2048, dropout=0.1, activation='relu', **kwargs):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.factor = factor
                self.d_ff = d_ff
                self.dropout = dropout
                self.activation = activation
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_ff = d_ff
        
        # AutoCorrelation attention mechanism
        self.autocorr_attention = AutoCorrelationAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            factor=factor, 
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        """Return the output dimension"""
        return self.d_model
    
    def get_attention_type(self) -> str:
        """Return the attention type identifier"""
        return "autocorrelation_layer"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Apply autocorrelation layer mechanism"""
        return self.forward(queries, keys, values, attn_mask)
        
    def forward(self, x, attn_mask=None):
        """
        Forward pass for AutoCorrelation layer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attn_mask: Attention mask (optional)
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # AutoCorrelation attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # For autocorrelation, we typically use x as queries, keys, and values
        attn_output, _ = self.autocorr_attention(x, x, x, attn_mask)
        x = residual + self.dropout_layer(attn_output)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + ff_output
        
        return x


# =============================================================================
# REGISTRY AND FACTORY FUNCTIONS
# =============================================================================

# Registry for clean attention components
ATTENTION_REGISTRY = {
    'multihead': MultiHeadAttention,
    'autocorrelation': AutoCorrelationAttention,
    'sparse': SparseAttention,
    'fourier': FourierAttention,
    'wavelet': WaveletAttention,
    'bayesian': BayesianAttention,
    'adaptive': AdaptiveAttention,
    'autocorrelation_layer': AutoCorrelationLayer,
}


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


# Add compute_loss method for compatibility
def add_compute_loss_method(cls):
    """Add compute_loss method to attention classes for compatibility"""
    if not hasattr(cls, 'compute_loss'):
        def compute_loss(self, predictions, targets, mask=None):
            # For attention layers, this is just a forward pass
            output, attn_weights = self.forward(predictions, targets, targets, mask)
            # Return MSE loss between input and output for reconstruction
            return F.mse_loss(output, predictions)
        
        cls.compute_loss = compute_loss
    return cls

# Apply compatibility patches
for name, cls in ATTENTION_REGISTRY.items():
    ATTENTION_REGISTRY[name] = add_compute_loss_method(cls)
    logger.info("✅ ALL attention components consolidated in iAttention.py")
    
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
