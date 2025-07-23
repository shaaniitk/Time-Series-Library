"""
UNIFIED ATTENTION COMPONENTS
Consolidated from all attention implementations into a single file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseAttention
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CORE ATTENTION COMPONENTS
# =============================================================================

class MultiHeadAttention(BaseAttention):
    """Standard Multi-Head Attention mechanism"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return output, attn_weights


class AutoCorrelationAttention(BaseAttention):
    """AutoCorrelation attention mechanism for time series"""
    
    def __init__(self, d_model=512, n_heads=8, factor=1, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.dropout = dropout
        
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def time_delay_agg_training(self, values, corr):
        """Time delay aggregation for training"""
        head, channel, length = values.shape
        # Simple aggregation for time series
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(values, dim=1)
        index = torch.topk(torch.mean(corr, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(2).repeat(1, channel, length))
        return delays_agg
        
    def time_delay_agg_inference(self, values, corr):
        """Time delay aggregation for inference"""
        batch, head, channel, length = values.shape
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(values, dim=2)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, :, i].unsqueeze(2).unsqueeze(3).repeat(1, 1, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:, :, i].unsqueeze(2).unsqueeze(3).repeat(1, 1, channel, length))
        return delays_agg

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # AutoCorrelation computation using FFT
        q_fft = torch.fft.rfft(Q.permute(0, 1, 3, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(K.permute(0, 1, 3, 2).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        
        # Time delay aggregation
        if self.training:
            V_agg = self.time_delay_agg_training(V.permute(0, 1, 3, 2).contiguous(), corr)
        else:
            V_agg = self.time_delay_agg_inference(V.permute(0, 1, 3, 2).contiguous(), corr)
        
        V_agg = V_agg.permute(0, 1, 3, 2).contiguous()
        context = V_agg.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return output, None


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
MultiHeadAttention = add_compute_loss_method(MultiHeadAttention)
AutoCorrelationAttention = add_compute_loss_method(AutoCorrelationAttention)


# =============================================================================
# ATTENTION REGISTRY
# =============================================================================

ATTENTION_REGISTRY = {
    # Core attention mechanisms
    'multihead': MultiHeadAttention,
    'multi_head': MultiHeadAttention,
    'autocorrelation': AutoCorrelationAttention,
    'autocorrelation_layer': AutoCorrelationAttention,
}


def get_attention_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get attention component by name"""
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention component: {name}. Available: {list(ATTENTION_REGISTRY.keys())}")
    
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
    
    logger.info(f"Registered {len(ATTENTION_REGISTRY)} unified attention components")


def list_attention_components():
    """List all available attention components"""
    return list(ATTENTION_REGISTRY.keys())


# =============================================================================
# COMPATIBILITY LAYER - ALIAS FOR AutoCorrelationLayer
# =============================================================================

# Create alias for backward compatibility
AutoCorrelationLayer = AutoCorrelationAttention

logger.info("✅ Unified attention components loaded successfully")


class AutoCorrelationAttention(BaseAttention):
    """Autocorrelation-based attention for time series"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.factor = config.custom_params.get('factor', 1)
        self.dropout = config.dropout
        self.output_attention = config.custom_params.get('output_attention', False)
        
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = queries.shape
        H = self.n_heads
        E = D // H
        
        queries = self.query_projection(queries).view(B, L, H, E)
        keys = self.key_projection(keys).view(B, L, H, E)
        values = self.value_projection(values).view(B, L, H, E)
        
        # Time delay aggregation through autocorrelation
        output = self._time_delay_agg_training(queries, keys, values, L)
        
        output = output.transpose(2, 1).contiguous().view(B, L, -1)
        output = self.out_projection(output)
        
        if self.output_attention:
            return output, None
        else:
            return output, None
    
    def _time_delay_agg_training(self, queries, keys, values, L):
        # Simple autocorrelation implementation
        B, L, H, E = queries.shape
        
        # Compute autocorrelation
        queries_fft = torch.fft.rfft(queries, dim=1)
        keys_fft = torch.fft.rfft(keys, dim=1)
        
        res = queries_fft * torch.conj(keys_fft)
        corr = torch.fft.irfft(res, dim=1)
        
        # Apply to values
        output = torch.zeros_like(queries)
        for i in range(L):
            output[:, i, :, :] = values[:, i, :, :] * corr[:, i, :, :].abs()
        
        return output


# =============================================================================
# FOURIER ATTENTION COMPONENTS
# =============================================================================

class FourierAttention(BaseAttention):
    """Fourier-based attention for capturing periodic patterns"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.seq_len = config.custom_params.get('seq_len', 96)
        self.dropout = config.dropout
        self.head_dim = self.d_model // self.n_heads
        
        # Learnable frequency components
        max_freq_dim = max(self.seq_len // 2 + 1, 64)
        self.freq_weights = nn.Parameter(torch.randn(max_freq_dim))
        self.phase_weights = nn.Parameter(torch.zeros(max_freq_dim))
        
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = queries.shape
        
        # Apply FFT for frequency domain processing
        try:
            queries_fft = torch.fft.rfft(queries, dim=1)
            freq_dim = queries_fft.shape[1]
            
            # Apply frequency filtering
            freq_filter = torch.sigmoid(self.freq_weights[:freq_dim])
            queries_filtered = queries_fft * freq_filter.unsqueeze(0).unsqueeze(-1)
            
            # Transform back to time domain
            queries_filtered = torch.fft.irfft(queries_filtered, n=L, dim=1)
        except:
            # Fallback if FFT fails
            queries_filtered = queries
        
        # Standard multi-head attention on filtered signal
        qkv = self.qkv(queries_filtered).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights


class FourierBlock(BaseAttention):
    """1D Fourier block for frequency domain processing"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.seq_len = config.custom_params.get('seq_len', 96)
        self.modes = config.custom_params.get('modes', 64)
        
        # Select frequency modes
        self.index = list(range(0, min(self.modes, self.seq_len // 2)))
        
        self.scale = (1 / (self.d_model * self.d_model))
        
        # Learnable frequency domain weights
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(self.n_heads, self.d_model // self.n_heads, 
                                  self.d_model // self.n_heads, len(self.index))
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(self.n_heads, self.d_model // self.n_heads, 
                                  self.d_model // self.n_heads, len(self.index))
        )
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = queries.shape
        H = self.n_heads
        E = D // H
        
        x = queries.view(B, L, H, E).permute(0, 2, 3, 1)  # [B, H, E, L]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Apply learnable frequency filters
        out_ft = torch.zeros_like(x_ft)
        for wi, i in enumerate(self.index):
            if i < x_ft.shape[-1] and wi < len(self.index):
                weights = torch.complex(self.weights_real[:, :, :, wi], 
                                      self.weights_imag[:, :, :, wi])
                out_ft[:, :, :, wi] = x_ft[:, :, :, i] * weights
        
        # Transform back to time domain
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        output = x_out.permute(0, 3, 1, 2).reshape(B, L, -1)
        
        return output, None


# =============================================================================
# WAVELET ATTENTION COMPONENTS
# =============================================================================

class WaveletAttention(BaseAttention):
    """Wavelet-based attention for multi-scale analysis"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.levels = config.custom_params.get('levels', 3)
        self.dropout = config.dropout
        
        # Learnable wavelet filters
        self.wavelet_filters = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_model, 4))  # 4-tap wavelet
            for _ in range(self.levels)
        ])
        
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
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
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
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


# =============================================================================
# BAYESIAN ATTENTION COMPONENTS
# =============================================================================

class BayesianAttention(BaseAttention):
    """Bayesian attention with uncertainty quantification"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.dropout = config.dropout
        self.prior_std = config.custom_params.get('prior_std', 1.0)
        
        # Bayesian linear layers
        self.q_mean = nn.Linear(self.d_model, self.d_model)
        self.q_logvar = nn.Linear(self.d_model, self.d_model)
        self.k_mean = nn.Linear(self.d_model, self.d_model)
        self.k_logvar = nn.Linear(self.d_model, self.d_model)
        self.v_mean = nn.Linear(self.d_model, self.d_model)
        self.v_logvar = nn.Linear(self.d_model, self.d_model)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
    def _reparameterize(self, mean, logvar):
        """Reparameterization trick for Bayesian sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
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


# =============================================================================
# ADAPTIVE ATTENTION COMPONENTS
# =============================================================================

class AdaptiveAttention(BaseAttention):
    """Adaptive attention with dynamic parameter selection"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.custom_params.get('n_heads', 8)
        self.dropout = config.dropout
        
        # Adaptive components
        self.attention_types = nn.ModuleList([
            MultiHeadAttention(config),
            AutoCorrelationAttention(config),
            FourierAttention(config)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.attention_types)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = queries.shape
        
        # Compute gating weights
        gate_input = queries.mean(dim=1)  # [B, D]
        gate_weights = self.gate(gate_input)  # [B, num_types]
        
        # Apply each attention type
        outputs = []
        for attention_layer in self.attention_types:
            out, _ = attention_layer(queries, keys, values, attn_mask, tau, delta)
            outputs.append(out)
        
        # Weighted combination
        output = torch.zeros_like(queries)
        for i, out in enumerate(outputs):
            weight = gate_weights[:, i:i+1, None]  # [B, 1, 1]
            output += weight * out
        
        return output, None


# =============================================================================
# AUTOCORRELATION COMPONENTS (from layers/AutoCorrelation.py)
# =============================================================================

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
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
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
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
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(BaseAttention):
    """AutoCorrelation Layer wrapper for modular framework"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        
        # Extract parameters from config
        d_model = config.d_model
        n_heads = config.custom_params.get('n_heads', 8)
        factor = config.custom_params.get('factor', 1)
        attention_dropout = config.custom_params.get('attention_dropout', config.dropout)
        output_attention = config.custom_params.get('output_attention', False)
        
        d_keys = config.custom_params.get('d_keys', d_model // n_heads)
        d_values = config.custom_params.get('d_values', d_model // n_heads)

        # Create inner correlation mechanism
        self.inner_correlation = AutoCorrelation(
            mask_flag=True,
            factor=factor,
            attention_dropout=attention_dropout,
            output_attention=output_attention
        )
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        logger.info(f"AutoCorrelationLayer initialized: d_model={d_model}, n_heads={n_heads}")

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

    def apply_attention(self, queries, keys, values, attn_mask=None):
        """Implementation of abstract method from BaseAttention"""
        return self.forward(queries, keys, values, attn_mask)


# =============================================================================
# COMPONENT REGISTRY
# =============================================================================

class AttentionRegistry:
    """Registry for all attention components"""
    
    _components = {
        'multihead': MultiHeadAttention,
        'autocorrelation': AutoCorrelationAttention,
        'autocorrelation_layer': AutoCorrelationLayer,  # Added core AutoCorrelation from layers
        'fourier': FourierAttention,
        'fourier_block': FourierBlock,
        'wavelet': WaveletAttention,
        'bayesian': BayesianAttention,
        'adaptive': AdaptiveAttention,
    }
    
    @classmethod
    def register(cls, name: str, component_class):
        """Register a new attention component"""
        cls._components[name] = component_class
        logger.info(f"Registered attention component: {name}")
    
    @classmethod
    def create(cls, name: str, config: ComponentConfig):
        """Create an attention component by name"""
        if name not in cls._components:
            raise ValueError(f"Unknown attention component: {name}")
        
        component_class = cls._components[name]
        return component_class(config)
    
    @classmethod
    def list_components(cls):
        """List all available attention components"""
        return list(cls._components.keys())


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_attention_component(name: str, config: ComponentConfig = None, **kwargs):
    """Factory function to create attention components"""
    if config is None:
        # Create config from kwargs for backward compatibility
        config = ComponentConfig(
            component_name=name,
            d_model=kwargs.get('d_model', 512),
            dropout=kwargs.get('dropout', 0.1),
            custom_params=kwargs
        )
    
    return AttentionRegistry.create(name, config)


def register_attention_components(registry):
    """Register all attention components with the main registry"""
    for name, component_class in AttentionRegistry._components.items():
        registry.register('attention', name, component_class)
    
    logger.info(f"Registered {len(AttentionRegistry._components)} attention components")
