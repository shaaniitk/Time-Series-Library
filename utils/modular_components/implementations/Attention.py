class MultiWaveletCorrelation(BaseAttention):
    """
    MultiWaveletCorrelation: Multiwavelet and polynomial basis correlation (ported from layers/MultiWaveletCorrelation.py).
    """
    def __init__(self, d_model, n_heads, seq_len, basis='legendre', degree=4, dropout=0.1):
        import torch
        import torch.nn as nn
        import numpy as np
        from scipy.special import eval_legendre
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.seq_len = seq_len
                self.basis = basis
                self.degree = degree
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.basis = basis
        self.degree = degree
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // n_heads
        # Construct polynomial basis (Legendre or Chebyshev)
        self.register_buffer('basis_matrix', self._build_basis_matrix(seq_len, degree, basis))
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _build_basis_matrix(self, seq_len, degree, basis):
        # Returns [seq_len, degree] matrix
        x = np.linspace(-1, 1, seq_len)
        basis_matrix = []
        if basis == 'legendre':
            for d in range(degree):
                basis_matrix.append(eval_legendre(d, x))
        elif basis == 'chebyshev':
            for d in range(degree):
                basis_matrix.append(np.cos(d * np.arccos(x)))
        else:
            raise ValueError(f"Unknown basis: {basis}")
        basis_matrix = np.stack(basis_matrix, axis=1)  # [seq_len, degree]
        return torch.tensor(basis_matrix, dtype=torch.float32)

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        B, L, D = x.shape
        H = self.n_heads
        D_h = self.head_dim
        # Project to Q, K, V
        Q = self.proj_q(x).view(B, L, H, D_h).transpose(1, 2)  # [B, H, L, D_h]
        K = self.proj_k(x).view(B, L, H, D_h).transpose(1, 2)
        V = self.proj_v(x).view(B, L, H, D_h).transpose(1, 2)
        # Project Q, K onto polynomial basis
        # basis_matrix: [L, degree]
        basis = self.basis_matrix[:L, :].to(Q.device)  # [L, degree]
        Qb = torch.einsum('bhlm,md->bhld', Q, basis)  # [B, H, L, degree]
        Kb = torch.einsum('bhlm,md->bhld', K, basis)
        # Compute correlation in basis space
        corr = torch.einsum('bhld,bhld->bhl', Qb, Kb) / (self.degree ** 0.5)  # [B, H, L]
        attn = torch.softmax(corr, dim=-1)
        attn = self.dropout(attn)
        # Weighted sum of V
        out = torch.einsum('bhlm,bhl->bhlm', V, attn)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out, attn
class FourierAttention(BaseAttention):
    """Fourier-based attention for capturing periodic patterns (ported from layers/AdvancedComponents.py)"""
    def __init__(self, d_model, n_heads, seq_len, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.seq_len = seq_len
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.freq_weights = nn.Parameter(torch.randn(seq_len // 2 + 1, n_heads))
        self.phase_weights = nn.Parameter(torch.zeros(seq_len // 2 + 1, n_heads))
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        freq_filter = torch.complex(
            torch.cos(self.phase_weights) * self.freq_weights,
            torch.sin(self.phase_weights) * self.freq_weights
        )
        x_freq = x_freq.unsqueeze(-1) * freq_filter.unsqueeze(0).unsqueeze(2)
        x_filtered = torch.fft.irfft(x_freq.mean(-1), n=L, dim=1)
        qkv = self.qkv(x_filtered).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out, attn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

class FourierBlock(BaseAttention):
    """1D Fourier block for frequency domain representation learning (ported from layers/FourierCorrelation.py)"""
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=0, mode_select_method='random'):
        class Config:
            def __init__(self):
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.n_heads = n_heads
                self.seq_len = seq_len
                self.modes = modes
                self.mode_select_method = mode_select_method
        super().__init__(Config())
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.modes = modes
        self.mode_select_method = mode_select_method
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        B, L, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1)
        # Select frequency modes
        x_fft_modes = x_fft[:, self.index, :]
        # Linear transform in frequency domain
        x_fft_modes = self.linear(x_fft_modes)
        # Inverse FFT to time domain
        x_ifft = torch.fft.irfft(x_fft_modes, n=L, dim=1)
        return x_ifft, None
"""
ATTENTION COMPONENTS - UNIFIED IMPLEMENTATION
Single source of truth for ALL attention mechanisms in the modular framework
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

# Import helper classes
from .componentHelpers import (
    BayesianLinear, 
    WaveletDecomposition, 
    TemporalBlock, 
    Chomp1d,
    FourierModeSelector,
    ComplexMultiply1D
)

logger = logging.getLogger(__name__)


# =============================================================================
# CORE ATTENTION MECHANISMS
# =============================================================================

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


class WaveletAttention(BaseAttention):
    """Wavelet-based attention for multi-scale analysis"""
    
    def __init__(self, d_model=512, n_heads=8, levels=3, dropout=0.1):
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
        self.dropout = nn.Dropout(dropout)
    
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
            attn_weights = self.dropout(attn_weights)
            
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
    
    def __init__(self, d_model=512, n_heads=8, prior_std=1.0, dropout=0.1):
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
            attn_scores.masked_fill_(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights


class AdaptiveAttention(BaseAttention):
    """Adaptive attention with dynamic parameter selection"""
    
    def __init__(self, d_model=512, n_heads=8, adaptation_rate=0.1, dropout=0.1):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.adaptation_rate = adaptation_rate
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters
        self.gate_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_heads),
            nn.Sigmoid()
        )
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
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
        
        # Compute adaptive gates
        gates = self.gate_network(queries.mean(dim=1))  # [B, n_heads]
        
        # Standard multi-head attention
        qkv = self.qkv(queries).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        scale = math.sqrt(D // self.n_heads)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply adaptive gates
        gates = gates.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, 1, 1]
        attn_weights = attn_weights * gates
        
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights


# =============================================================================
# ADVANCED ATTENTION COMPONENTS (from attention_migrated.py)
# =============================================================================

class MetaLearningAdapter(BaseAttention):
    """MAML-style Meta-Learning Adapter with proper gradient-based fast adaptation"""
    
    def __init__(self, d_model=512, n_heads=8, adaptation_steps=5, meta_lr=0.01, inner_lr=0.1, dropout=0.1, **kwargs):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.adaptation_steps = adaptation_steps
                self.meta_lr = meta_lr
                self.inner_lr = inner_lr
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.adaptation_steps = adaptation_steps
        self.inner_lr = inner_lr
        
        # Base network for meta-learning
        self.base_network = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Meta-learning rate
        self.meta_lr = nn.Parameter(torch.tensor(meta_lr))
        
        # Task context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        self.context_projector = nn.Linear(d_model // 4, d_model)
        self.task_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "meta_learning"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        # Simplified implementation - use queries as main input
        residual = queries
        base_output = self.base_network(queries)
        final_output, attention_weights = self.task_attention(queries, base_output, base_output, attn_mask=attn_mask)
        final_output = self.dropout_layer(final_output) + residual
        return final_output, attention_weights


class AdaptiveMixture(BaseAttention):
    """Adaptive mixture of experts for different time series patterns"""
    
    def __init__(self, d_model=512, n_heads=8, mixture_components=4, gate_hidden_dim=None, dropout=0.1, temperature=1.0, **kwargs):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.mixture_components = mixture_components
                self.dropout = dropout
                self.temperature = temperature
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = mixture_components
        self.temperature = temperature
        
        if gate_hidden_dim is None:
            gate_hidden_dim = d_model // 2
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "adaptive_mixture"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        x = queries
        
        # Compute gating weights with temperature
        gate_weights = self.gate(x) / self.temperature
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        
        return self.dropout_layer(output), None


class EnhancedAutoCorrelation(BaseAttention):
    """Enhanced AutoCorrelation with adaptive window selection and multi-scale analysis"""
    
    def __init__(self, d_model=512, n_heads=8, factor=1, scale=None, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True, 
                 scales=[1, 2, 4], eps=1e-8, **kwargs):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.factor = factor
                self.dropout = attention_dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.adaptive_k = adaptive_k
        self.multi_scale = multi_scale
        self.scales = scales
        self.eps = eps
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout_layer = nn.Dropout(attention_dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "enhanced_autocorrelation"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
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
        
        # Adaptive correlation computation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        
        # Enhanced time delay aggregation
        V = values.permute(0, 2, 3, 1).contiguous()
        
        return V.permute(0, 3, 1, 2).contiguous(), corr


class CausalConvolution(BaseAttention):
    """Causal convolution attention mechanism for temporal sequence modeling"""
    
    def __init__(self, d_model=512, n_heads=8, kernel_sizes=[3, 5, 7], dilation_rates=[1, 2, 4],
                 dropout=0.1, activation='gelu', **kwargs):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        # Multi-scale causal convolutions
        self.causal_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, dilation=d, 
                     padding=(k-1)*d, groups=d_model//16 if d_model >= 16 else 1)
            for k in kernel_sizes for d in dilation_rates
        ])
        
        self.attention_proj = nn.Linear(d_model * len(kernel_sizes) * len(dilation_rates), d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "causal_convolution"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        x = queries.transpose(1, 2)  # [B, D, L]
        
        # Apply multi-scale causal convolutions
        conv_outputs = []
        for conv in self.causal_convs:
            conv_out = conv(x)
            # Remove future information (causal)
            conv_out = conv_out[:, :, :L]
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(conv_outputs, dim=1)  # [B, D*scales, L]
        multi_scale = multi_scale.transpose(1, 2)  # [B, L, D*scales]
        
        # Project to original dimension
        attention_out = self.attention_proj(multi_scale)
        attention_out = self.activation(attention_out)
        
        # Final projection
        output = self.output_proj(attention_out)
        output = self.dropout_layer(output)
        
        return output, None


class TemporalConvNet(BaseAttention):
    """Temporal Convolution Network (TCN) based attention mechanism"""
    
    def __init__(self, d_model=512, n_heads=8, num_levels=4, kernel_size=3, dropout=0.1, **kwargs):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.num_levels = num_levels
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        for i in range(num_levels):
            dilation = 2 ** i
            self.tcn_layers.append(
                nn.Conv1d(d_model, d_model, kernel_size, 
                         dilation=dilation, padding=(kernel_size-1)*dilation)
            )
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "temporal_conv_net"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        x = queries.transpose(1, 2)  # [B, D, L]
        
        # Apply TCN layers
        for tcn_layer in self.tcn_layers:
            residual = x
            x = tcn_layer(x)
            x = x[:, :, :L]  # Remove padding
            x = F.gelu(x)
            x = x + residual  # Residual connection
        
        x = x.transpose(1, 2)  # [B, L, D]
        output = self.output_proj(x)
        output = self.dropout_layer(output)
        
        return output, None


class AdaptiveWaveletAttention(BaseAttention):
    """Adaptive wavelet attention with learnable decomposition levels"""
    
    def __init__(self, d_model=512, n_heads=8, max_levels=5, **kwargs):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.max_levels = max_levels
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_levels = max_levels
        
        # Adaptive level selection
        self.level_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_levels),
            nn.Softmax(dim=-1)
        )
        
        # Wavelet decomposition for each level
        self.wavelet_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
            for _ in range(max_levels)
        ])
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(max_levels)
        ])
        
        self.fusion_proj = nn.Linear(d_model * max_levels, d_model)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "adaptive_wavelet"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # Select adaptive levels
        level_weights = self.level_selector(queries.mean(dim=1))  # [B, max_levels]
        
        # Apply wavelet decomposition at each level
        level_outputs = []
        x = queries.transpose(1, 2)  # [B, D, L]
        
        for i, (wavelet_layer, attention_layer) in enumerate(zip(self.wavelet_layers, self.attention_layers)):
            # Wavelet decomposition
            if x.shape[-1] > 1:
                x_decomp = wavelet_layer(x)
                x_decomp = x_decomp.transpose(1, 2)  # [B, L', D]
                
                # Interpolate back to original length
                if x_decomp.shape[1] != L:
                    x_decomp = F.interpolate(x_decomp.transpose(1, 2), size=L, 
                                           mode='linear', align_corners=False).transpose(1, 2)
                
                # Apply attention
                attn_out, _ = attention_layer(x_decomp, x_decomp, x_decomp, attn_mask=attn_mask)
                level_outputs.append(attn_out)
                
                # Update x for next level
                x = x_decomp.transpose(1, 2)
            else:
                level_outputs.append(queries)  # Fallback
        
        # Weighted fusion
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        
        return fused_output, None


class MultiScaleWaveletAttention(BaseAttention):
    """Multi-scale wavelet attention for capturing patterns at different time scales"""
    
    def __init__(self, d_model=512, n_heads=8, scales=[1, 2, 4, 8], **kwargs):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.scales = scales
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        
        # Scale-specific attention layers
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in scales
        ])
        
        # Scale weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.fusion_proj = nn.Linear(d_model * len(scales), d_model)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "multi_scale_wavelet"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        scale_outputs = []
        
        for scale, attention_layer in zip(self.scales, self.scale_attentions):
            # Downsample for this scale
            if scale > 1:
                downsampled_q = F.avg_pool1d(queries.transpose(1, 2), kernel_size=scale, stride=scale)
                downsampled_k = F.avg_pool1d(keys.transpose(1, 2), kernel_size=scale, stride=scale)
                downsampled_v = F.avg_pool1d(values.transpose(1, 2), kernel_size=scale, stride=scale)
                
                downsampled_q = downsampled_q.transpose(1, 2)
                downsampled_k = downsampled_k.transpose(1, 2)
                downsampled_v = downsampled_v.transpose(1, 2)
            else:
                downsampled_q, downsampled_k, downsampled_v = queries, keys, values
            
            # Apply attention at this scale
            scale_out, _ = attention_layer(downsampled_q, downsampled_k, downsampled_v, attn_mask=attn_mask)
            
            # Upsample back to original size
            if scale > 1:
                scale_out = F.interpolate(scale_out.transpose(1, 2), size=L, 
                                        mode='linear', align_corners=False).transpose(1, 2)
            
            scale_outputs.append(scale_out)
        
        # Weighted fusion
        weights = F.softmax(self.scale_weights, dim=0)
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        
        return fused_output, None


# =============================================================================
# ATTENTION REGISTRY
# =============================================================================

ATTENTION_REGISTRY = {
    'multihead': MultiHeadAttention,
    'multi_head': MultiHeadAttention,
    'autocorrelation': AutoCorrelationAttention,
    'sparse': SparseAttention,
    'fourier': FourierAttention,
    'wavelet': WaveletAttention,
    'bayesian': BayesianAttention,
    'adaptive': AdaptiveAttention,
    # Advanced attention components
    'meta_learning': MetaLearningAdapter,
    'adaptive_mixture': AdaptiveMixture,
    'enhanced_autocorrelation': EnhancedAutoCorrelation,
    'causal_convolution': CausalConvolution,
    'temporal_conv_net': TemporalConvNet,
    'adaptive_wavelet': AdaptiveWaveletAttention,
    'multi_scale_wavelet': MultiScaleWaveletAttention,
}

# Import AutoCorrelationLayer from layers module for backward compatibility
try:
    from .Layers import AutoCorrelationLayer
    ATTENTION_REGISTRY['autocorrelation_layer'] = AutoCorrelationLayer
    AUTOCORRELATION_LAYER_AVAILABLE = True
except ImportError:
    AUTOCORRELATION_LAYER_AVAILABLE = False


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
    
    logger.info(f"Registered {len(ATTENTION_REGISTRY)} attention components")


def list_attention_components():
    """List all available attention components"""
    return list(ATTENTION_REGISTRY.keys())


# =============================================================================
# ADVANCED ATTENTION COMPONENTS FROM OTHER FILES
# =============================================================================

class LogSparseAttention(BaseAttention):
    """LogSparse attention for very long sequences with O(n log n) complexity"""
    
    def __init__(self, d_model=512, n_heads=8, block_size=128, dropout=0.1):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.block_size = block_size
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "log_sparse"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
        
    def forward(self, query, key, value, attention_mask=None):
        B, L, D = query.shape
        
        # Standard projections
        Q = self.w_q(query).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Log-sparse pattern
        num_blocks = max(1, L // self.block_size)
        block_indices = torch.logspace(0, math.log2(num_blocks), steps=min(8, num_blocks), 
                                     base=2, dtype=torch.long, device=query.device)
        
        # Apply attention only to selected blocks
        attn_output = torch.zeros_like(Q)
        attn_weights = torch.zeros(B, self.n_heads, L, L, device=query.device)
        
        for block_idx in block_indices:
            start = int(block_idx * self.block_size)
            end = min(start + self.block_size, L)
            
            if start < L:
                q_block = Q[:, :, start:end]
                k_block = K[:, :, :end]  # Can attend to all previous positions
                v_block = V[:, :, :end]
                
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.d_k)
                
                if attention_mask is not None:
                    mask_block = attention_mask[:, start:end, :end]
                    scores.masked_fill_(mask_block.unsqueeze(1) == 0, -float('inf'))
                
                block_attn = F.softmax(scores, dim=-1)
                attn_output[:, :, start:end] = torch.matmul(block_attn, v_block)
                attn_weights[:, :, start:end, :end] = block_attn
        
        output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.w_o(output)
        
        return output, attn_weights


class ProbSparseAttention(BaseAttention):
    """ProbSparse attention with probabilistic sampling"""
    
    def __init__(self, d_model=512, n_heads=8, factor=5, dropout=0.1):
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
        
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "prob_sparse"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        """Probabilistic sampling of keys"""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Calculate the sample
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Find the Top_k query with sparse measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def forward(self, query, key, value, attention_mask=None):
        B, L, D = query.shape
        
        Q = self.w_q(query).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # ProbSparse sampling
        U_part = self.factor * np.ceil(np.log(L)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()
        
        U_part = U_part if U_part < L else L
        u = u if u < L else L
        
        if L > u:
            scores_top, index = self._prob_QK(Q, K, sample_k=U_part, n_top=u)
            
            # Add scale factor
            scale = 1. / math.sqrt(self.d_k)
            scores_top = scores_top * scale
            
            # Get context for all queries
            context = torch.zeros(B, self.n_heads, L, self.d_k, device=query.device)
            attn = F.softmax(scores_top, dim=-1)
            context[torch.arange(B)[:, None, None], torch.arange(self.n_heads)[None, :, None], index, :] = \
                torch.matmul(attn, V).type_as(context)
            
            # Update based on the full input
            attn_weights = torch.zeros(B, self.n_heads, L, L, device=query.device)
            attn_weights[torch.arange(B)[:, None, None], torch.arange(self.n_heads)[None, :, None], index, :] = attn
        else:
            # Standard attention for short sequences
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if attention_mask is not None:
                scores.masked_fill_(attention_mask.unsqueeze(1) == 0, -float('inf'))
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, V)
        
        output = context.transpose(1, 2).contiguous().view(B, L, D)
        output = self.w_o(output)
        
        return output, attn_weights


class CrossResolutionAttention(BaseAttention):
    """Cross-resolution attention for multi-scale features"""
    
    def __init__(self, d_model=512, n_heads=8, scales=[1, 2, 4], dropout=0.1):
        # Create a minimal config object for BaseComponent
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.scales = scales
                self.dropout = dropout
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        
        # Multi-scale projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in scales
        ])
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "cross_resolution"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
        
    def forward(self, query, key, value, attention_mask=None):
        B, L, D = query.shape
        
        # Multi-scale processing
        scale_features = []
        for i, scale in enumerate(self.scales):
            if scale == 1:
                scale_feat = query
            else:
                # Downsample
                scale_feat = F.avg_pool1d(query.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
            
            # Project
            scale_feat = self.scale_projections[i](scale_feat)
            
            # Upsample back if needed
            if scale > 1:
                scale_feat = F.interpolate(scale_feat.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            scale_features.append(scale_feat)
        
        # Combine multi-scale features
        combined_features = torch.stack(scale_features, dim=0).mean(dim=0)
        
        # Standard attention on combined features
        qkv = self.qkv(combined_features).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        scale = math.sqrt(D // self.n_heads)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attention_mask is not None:
            attn_scores.masked_fill_(attention_mask.unsqueeze(1).unsqueeze(1) == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights


# =============================================================================
# MISSING AUTOCORRELATION COMPONENTS
# =============================================================================

class AdaptiveAutoCorrelationLayer(BaseAttention):
    """Layer wrapper for Enhanced AutoCorrelation with additional processing"""
    
    def __init__(self, d_model=512, n_heads=8, factor=1, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Core autocorrelation mechanism
        self.autocorrelation = EnhancedAutoCorrelation(
            d_model=d_model, n_heads=n_heads, factor=factor,
            attention_dropout=attention_dropout, output_attention=output_attention,
            adaptive_k=adaptive_k, multi_scale=multi_scale
        )
        
        # Additional processing layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(attention_dropout)
        )
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "adaptive_autocorrelation_layer"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        # Autocorrelation with residual connection
        residual = queries
        attn_output, attn_weights = self.autocorrelation(queries, keys, values, attn_mask)
        x = self.norm1(residual + attn_output)
        
        # Feed-forward with residual connection
        residual = x
        ffn_output = self.ffn(x)
        output = self.norm2(residual + ffn_output)
        
        return output, attn_weights


class HierarchicalAutoCorrelation(BaseAttention):
    """Hierarchical autocorrelation with multiple time scales"""
    
    def __init__(self, d_model=512, n_heads=8, hierarchy_levels=[1, 4, 16], factor=1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.hierarchy_levels = hierarchy_levels
        self.factor = factor
        
        # Autocorrelation for each hierarchy level
        self.autocorrelations = nn.ModuleList([
            AutoCorrelationAttention(d_model=d_model, n_heads=n_heads, factor=factor)
            for _ in hierarchy_levels
        ])
        
        # Level weights
        self.level_weights = nn.Parameter(torch.ones(len(hierarchy_levels)) / len(hierarchy_levels))
        self.fusion_proj = nn.Linear(d_model * len(hierarchy_levels), d_model)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "hierarchical_autocorrelation"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        level_outputs = []
        level_attentions = []
        
        for level, autocorr in zip(self.hierarchy_levels, self.autocorrelations):
            if level == 1:
                # Full resolution
                level_q, level_k, level_v = queries, keys, values
            else:
                # Downsample
                level_q = F.avg_pool1d(queries.transpose(1, 2), kernel_size=level, stride=level).transpose(1, 2)
                level_k = F.avg_pool1d(keys.transpose(1, 2), kernel_size=level, stride=level).transpose(1, 2)
                level_v = F.avg_pool1d(values.transpose(1, 2), kernel_size=level, stride=level).transpose(1, 2)
            
            # Apply autocorrelation
            level_output, level_attn = autocorr(level_q, level_k, level_v, attn_mask)
            
            # Upsample back if needed
            if level > 1:
                level_output = F.interpolate(level_output.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            level_outputs.append(level_output)
            level_attentions.append(level_attn)
        
        # Weighted fusion
        weights = F.softmax(self.level_weights, dim=0)
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        
        # Average attention weights
        avg_attention = torch.stack(level_attentions, dim=0).mean(dim=0)
        
        return fused_output, avg_attention


# =============================================================================
# MISSING FOURIER COMPONENTS
# =============================================================================

class FourierBlock(BaseAttention):
    """1D Fourier block for frequency domain representation learning"""
    
    def __init__(self, d_model=512, n_heads=8, seq_len=96, modes=64, mode_select_method='random'):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.mode_select_method = mode_select_method
        
        # Complex weights for frequency domain
        self.fourier_weights = nn.Parameter(
            torch.randn(d_model, modes, dtype=torch.cfloat) * 0.02
        )
        
        # Mode selection
        if mode_select_method == 'random':
            self.register_buffer('modes_list', torch.randperm(seq_len//2 + 1)[:modes])
        else:  # 'low'
            self.register_buffer('modes_list', torch.arange(modes))
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "fourier_block"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # FFT
        x_ft = torch.fft.rfft(queries, dim=1)
        
        # Multiply with learnable weights
        out_ft = torch.zeros_like(x_ft)
        for i, mode in enumerate(self.modes_list):
            if mode < x_ft.size(1):
                out_ft[:, mode, :] = x_ft[:, mode, :] * self.fourier_weights[:, i].unsqueeze(0)
        
        # IFFT
        output = torch.fft.irfft(out_ft, n=L, dim=1)
        
        return output, None


class FourierCrossAttention(BaseAttention):
    """Fourier-enhanced cross attention for encoder-decoder architectures"""
    
    def __init__(self, d_model=512, n_heads=8, seq_len_q=96, seq_len_kv=96, modes=64, 
                 mode_select_method='random', activation='tanh'):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        
        # Fourier weights for cross attention
        self.fourier_weights_q = nn.Parameter(
            torch.randn(d_model, modes, dtype=torch.cfloat) * 0.02
        )
        self.fourier_weights_kv = nn.Parameter(
            torch.randn(d_model, modes, dtype=torch.cfloat) * 0.02
        )
        
        # Standard attention components
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Activation
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = lambda x: x
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "fourier_cross_attention"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L_q, D = queries.shape
        _, L_kv, _ = keys.shape
        
        # Standard projections
        Q = self.w_q(queries)
        K = self.w_k(keys)
        V = self.w_v(values)
        
        # Fourier processing for queries
        Q_ft = torch.fft.rfft(Q, dim=1)
        Q_out_ft = torch.zeros_like(Q_ft)
        for i in range(min(self.modes, Q_ft.size(1))):
            Q_out_ft[:, i, :] = Q_ft[:, i, :] * self.fourier_weights_q[:, i].unsqueeze(0)
        Q_fourier = torch.fft.irfft(Q_out_ft, n=L_q, dim=1)
        
        # Fourier processing for keys and values
        K_ft = torch.fft.rfft(K, dim=1)
        K_out_ft = torch.zeros_like(K_ft)
        for i in range(min(self.modes, K_ft.size(1))):
            K_out_ft[:, i, :] = K_ft[:, i, :] * self.fourier_weights_kv[:, i].unsqueeze(0)
        K_fourier = torch.fft.irfft(K_out_ft, n=L_kv, dim=1)
        
        # Cross attention computation
        scores = torch.matmul(Q_fourier.view(B, L_q, self.n_heads, -1).transpose(1, 2),
                            K_fourier.view(B, L_kv, self.n_heads, -1).transpose(1, 2).transpose(-2, -1))
        scores = scores / math.sqrt(D // self.n_heads)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values
        context = torch.matmul(attn_weights, V.view(B, L_kv, self.n_heads, -1).transpose(1, 2))
        context = context.transpose(1, 2).contiguous().view(B, L_q, D)
        
        output = self.w_o(self.activation(context))
        
        return output, attn_weights


# =============================================================================
# MISSING CONVOLUTION AND SPECIAL ATTENTION COMPONENTS
# =============================================================================

class ConvolutionalAttention(BaseAttention):
    """Convolutional attention combining spatial and temporal convolutions"""
    
    def __init__(self, d_model=512, n_heads=8, conv_kernel_size=3, pool_size=2, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.conv_kernel_size = conv_kernel_size
        
        # 2D convolutions for attention feature extraction
        self.spatial_conv = nn.Conv2d(
            1, n_heads, 
            kernel_size=(conv_kernel_size, conv_kernel_size),
            padding=(conv_kernel_size//2, conv_kernel_size//2)
        )
        
        # Temporal convolutions
        self.temporal_conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # Standard attention projections
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "convolutional"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        residual = queries
        
        # Apply temporal convolutions
        q_temp = self.temporal_conv_q(queries.transpose(1, 2)).transpose(1, 2)
        k_temp = self.temporal_conv_k(keys.transpose(1, 2)).transpose(1, 2)
        v_temp = self.temporal_conv_v(values.transpose(1, 2)).transpose(1, 2)
        
        # Apply standard projections
        q = self.w_qs(q_temp).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_ks(k_temp).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_vs(v_temp).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1) == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Apply output projection
        output = self.w_o(context)
        
        # Residual connection and normalization
        output = self.layer_norm(output + residual)
        
        return output, attn_weights


class TwoStageAttention(BaseAttention):
    """Two-Stage Attention (TSA) for segment merging and multi-variate time series"""
    
    def __init__(self, d_model=512, n_heads=8, seg_num=4, factor=5, d_ff=None, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        d_ff = d_ff or 4 * d_model
        
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.dim_sender = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.dim_receiver = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "two_stage"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        # Assume input is [batch, ts_d, seg_num, d_model]
        if queries.dim() == 3:
            # Reshape if needed
            B, L, D = queries.shape
            seg_num = int(math.sqrt(L))  # Assume square segmentation
            queries = queries.view(B, seg_num, seg_num, D)
        
        batch, ts_d, seg_num, d_model = queries.shape
        
        # Cross Time Stage
        time_in = queries.reshape(-1, seg_num, d_model)
        time_enc, _ = self.time_attention(time_in, time_in, time_in, attn_mask=attn_mask)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        
        # Cross Dimension Stage
        dim_send = dim_in.reshape(batch * seg_num, ts_d, d_model)
        batch_router = self.router.repeat(batch, 1, 1)
        dim_buffer, _ = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=attn_mask)
        dim_receive, _ = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=attn_mask)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        
        final_out = dim_enc.reshape(batch, ts_d, seg_num, d_model)
        # Flatten back to [B, L, D] format
        final_out = final_out.view(batch, ts_d * seg_num, d_model)
        
        return final_out, None


class ExponentialSmoothingAttention(BaseAttention):
    """Exponential Smoothing Attention using learnable smoothing weights"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.n_heads = n_heads
        self.d_model = d_model
        
        self._smoothing_weight = nn.Parameter(torch.randn(n_heads, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, n_heads, d_model // n_heads))
        self.dropout = nn.Dropout(dropout)
    
    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "exponential_smoothing"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def get_exponential_weight(self, T):
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))
        init_weight = self.weight ** (powers + 1)
        return init_weight.unsqueeze(0).unsqueeze(-1), weight.unsqueeze(0).unsqueeze(-1)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = values.shape
        H = self.n_heads
        d_head = D // H
        
        values = values.view(B, L, H, d_head)
        init_weight, weight = self.get_exponential_weight(L)
        
        output = F.conv1d(self.dropout(values.permute(0, 2, 3, 1)), weight.squeeze(-1), groups=H)
        output = init_weight * self.v0 + output
        output = output.permute(0, 3, 1, 2).reshape(B, L, D)
        
        return output, None


class MultiWaveletCrossAttention(BaseAttention):
    """Multi-Wavelet Cross Attention using multi-resolution wavelet transforms"""
    
    def __init__(self, d_model=512, n_heads=8, levels=3, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.levels = levels
        
        self.dropout = nn.Dropout(dropout)
        self.wavelet_decomp = WaveletDecomposition(d_model, levels)
        
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(levels + 1)
        ])
        
        self.level_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        self.fusion_proj = nn.Linear(d_model * (levels + 1), d_model)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "multi_wavelet_cross"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # Wavelet decomposition
        _, q_components = self.wavelet_decomp(queries)
        _, k_components = self.wavelet_decomp(keys)
        _, v_components = self.wavelet_decomp(values)
        
        level_outputs = []
        level_attentions = []
        
        for i, (q_comp, k_comp, v_comp, attention_layer) in enumerate(
            zip(q_components, k_components, v_components, self.cross_attentions)
        ):
            level_output, level_attn = attention_layer(q_comp, k_comp, v_comp, attn_mask=attn_mask)
            level_outputs.append(level_output)
            level_attentions.append(level_attn)
        
        # Weighted fusion
        weights = F.softmax(self.level_weights, dim=0)
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.fusion_proj(concatenated)
        
        avg_attention = torch.stack(level_attentions, dim=0).mean(dim=0)
        
        return fused_output, avg_attention


# =============================================================================
# MISSING BAYESIAN ATTENTION COMPONENTS
# =============================================================================

class BayesianMultiHeadAttention(BaseAttention):
    """Multi-head Bayesian attention with ensemble predictions"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, prior_std=1.0, n_samples=5, output_attention=False):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_samples = n_samples
        self.output_attention = output_attention
        
        # Core Bayesian attention
        self.bayesian_attention = BayesianAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, prior_std=prior_std
        )
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "bayesian_multihead"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        if self.training:
            return self.bayesian_attention(queries, keys, values, attn_mask)
        else:
            # Multiple forward passes for uncertainty estimation
            outputs = []
            for _ in range(self.n_samples):
                output, _ = self.bayesian_attention(queries, keys, values, attn_mask)
                outputs.append(output)
            
            # Mean and uncertainty
            stacked_outputs = torch.stack(outputs, dim=0)
            mean_output = stacked_outputs.mean(dim=0)
            uncertainty = stacked_outputs.var(dim=0)
            
            return mean_output, uncertainty


class VariationalAttention(BaseAttention):
    """Variational attention with learnable variance"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, learn_variance=True):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_variance = learn_variance
        
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        if learn_variance:
            self.logvar_q = nn.Linear(d_model, d_model)
            self.logvar_k = nn.Linear(d_model, d_model)
            self.logvar_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "variational"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        
        # Standard projections
        Q = self.w_qs(queries).view(B, L, self.n_heads, -1).transpose(1, 2)
        K = self.w_ks(keys).view(B, L, self.n_heads, -1).transpose(1, 2)
        V = self.w_vs(values).view(B, L, self.n_heads, -1).transpose(1, 2)
        
        if self.learn_variance and self.training:
            # Add variational sampling
            logvar_q = self.logvar_q(queries).view(B, L, self.n_heads, -1).transpose(1, 2)
            logvar_k = self.logvar_k(keys).view(B, L, self.n_heads, -1).transpose(1, 2)
            
            # Reparameterization trick
            epsilon_q = torch.randn_like(Q)
            epsilon_k = torch.randn_like(K)
            
            Q = Q + torch.exp(0.5 * logvar_q) * epsilon_q
            K = K + torch.exp(0.5 * logvar_k) * epsilon_k
        
        # Standard attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        output = self.w_o(context)
        return output, attn_weights


class BayesianCrossAttention(BaseAttention):
    """Bayesian cross attention for encoder-decoder architectures"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, prior_std=1.0):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
        
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Bayesian projections
        self.bayesian_q = BayesianLinear(d_model, d_model, prior_std)
        self.bayesian_k = BayesianLinear(d_model, d_model, prior_std)
        self.bayesian_v = BayesianLinear(d_model, d_model, prior_std)
        self.bayesian_o = BayesianLinear(d_model, d_model, prior_std)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_attention_type(self) -> str:
        return "bayesian_cross"
    
    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L_q, D = queries.shape
        _, L_kv, _ = keys.shape
        
        # Bayesian projections
        Q = self.bayesian_q(queries).view(B, L_q, self.n_heads, -1).transpose(1, 2)
        K = self.bayesian_k(keys).view(B, L_kv, self.n_heads, -1).transpose(1, 2)
        V = self.bayesian_v(values).view(B, L_kv, self.n_heads, -1).transpose(1, 2)
        
        # Cross attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L_q, D)
        
        output = self.bayesian_o(context)
        return output, attn_weights


# =============================================================================
# UPDATED REGISTRY WITH ALL COMPONENTS
# =============================================================================

ATTENTION_REGISTRY.update({
    'log_sparse': LogSparseAttention,
    'prob_sparse': ProbSparseAttention,
    'cross_resolution': CrossResolutionAttention,
    # Additional Bayesian components
    'bayesian_multihead': BayesianMultiHeadAttention,
    'variational': VariationalAttention,
    'bayesian_cross': BayesianCrossAttention,
    # AutoCorrelation components
    'adaptive_autocorrelation_layer': AdaptiveAutoCorrelationLayer,
    'hierarchical_autocorrelation': HierarchicalAutoCorrelation,
    # Fourier components
    'fourier_block': FourierBlock,
    'fourier_cross_attention': FourierCrossAttention,
    # Convolution and special components
    'convolutional': ConvolutionalAttention,
    'two_stage': TwoStageAttention,
    'exponential_smoothing': ExponentialSmoothingAttention,
    'multi_wavelet_cross': MultiWaveletCrossAttention,
})

# Export all components
__all__ = [
    'MultiHeadAttention',
    'AutoCorrelationAttention', 
    'SparseAttention',
    'FourierAttention',
    'WaveletAttention',
    'BayesianAttention',
    'AdaptiveAttention',
    'MetaLearningAdapter',
    'AdaptiveMixture',
    'EnhancedAutoCorrelation',
    'CausalConvolution',
    'TemporalConvNet',
    'AdaptiveWaveletAttention',
    'MultiScaleWaveletAttention',
    'LogSparseAttention',
    'ProbSparseAttention', 
    'CrossResolutionAttention',
    # Missing components from attention_migrated.py
    'BayesianMultiHeadAttention',
    'VariationalAttention',
    'BayesianCrossAttention',
    'AdaptiveAutoCorrelationLayer',
    'HierarchicalAutoCorrelation',
    'FourierBlock',
    'FourierCrossAttention',
    'ConvolutionalAttention',
    'TwoStageAttention',
    'ExponentialSmoothingAttention',
    'MultiWaveletCrossAttention',
    'ATTENTION_REGISTRY',
    'get_attention_component',
    'register_attention_components',
    'list_attention_components'
]

logger.info("✅ ALL Unified Attention components loaded successfully")
