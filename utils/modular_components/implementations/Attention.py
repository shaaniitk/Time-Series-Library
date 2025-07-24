
"""
ATTENTION COMPONENTS - UNIFIED IMPLEMENTATION
Single source of truth for ALL attention mechanisms in the modular framework
"""

# Imports at the top
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

# Utility functions after imports
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

# ...existing code for all attention classes and registry follows...

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


from .Attention.AutoCorrelation.autocorr import AutoCorrelationAttention


from .Attention.Sparse.sparse import SparseAttention


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


from .Attention.Wavelet.wavelet import WaveletAttention



from .Attention.Bayesian.bayesian import BayesianAttention

from .Attention.Adaptive.adaptive import AdaptiveAttention

# =============================================================================
# ADVANCED ATTENTION COMPONENTS (MetaLearningAdapter now modularized)
# =============================================================================
from .Attention.MetaLearning.meta_learning_adapter import MetaLearningAdapter



# Modular Mixture family import
from .Attention.Mixture.adaptive_mixture import AdaptiveMixture



class EnhancedAutoCorrelation(BaseAttention):
    """
    Enhanced AutoCorrelation with adaptive window selection and multi-scale analysis.
    
    Key improvements over standard AutoCorrelation:
    1. Adaptive top-k selection based on correlation energy
    2. Multi-scale correlation analysis
    3. Learnable frequency filtering
    4. Numerical stability enhancements
    """

    def __init__(self, d_model, n_heads, factor=1, scale=None, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True, 
                 scales=[1, 2, 4], eps=1e-8):
        super(EnhancedAutoCorrelation, self).__init__()
        logger.info(f"Initializing EnhancedAutoCorrelation with enhanced features: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Adaptive parameters
        self.adaptive_k = adaptive_k
        self.multi_scale = multi_scale
        self.scales = scales
        self.eps = eps
        
        # Learnable components
        if self.multi_scale:
            self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
            
        # Frequency filter for noise reduction
        self.frequency_filter = nn.Parameter(torch.ones(1))
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.output_dim_multiplier = 1

    def _select_adaptive_k(self, corr_energy, length):
        """
        Intelligently select the number of correlation peaks to use.
        """
        # Sort correlation energies in descending order
        sorted_energies, _ = torch.sort(corr_energy, dim=-1, descending=True)
        
        # Find the elbow point using second derivative
        if length > 10:  # Only for reasonable sequence lengths
            # Compute first and second derivatives
            first_diff = sorted_energies[:, :-1] - sorted_energies[:, 1:]
            second_diff = first_diff[:, :-1] - first_diff[:, 1:]
            
            # Find elbow as point of maximum curvature
            elbow_candidates = torch.argmax(second_diff, dim=-1) + 2  # +2 due to double differencing
            
            # Ensure reasonable bounds
            min_k = max(2, int(0.1 * math.log(length)))
            max_k = min(int(0.3 * length), int(self.factor * math.log(length) * 2))
            
            if min_k > max_k:
                max_k = min_k
            
            adaptive_k = torch.clamp(elbow_candidates, min_k, max_k)
            
            # Use median across batch for stability
            return int(torch.median(adaptive_k.float()).item())
        else:
            return max(2, int(self.factor * math.log(length)))

    def _multi_scale_correlation(self, queries, keys, length):
        """
        Compute correlations at multiple scales to capture different periodicities.
        """
        correlations = []
        B, H, L, E = queries.shape
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # Original scale - ensure contiguous tensors
                q_input = queries.contiguous()
                k_input = keys.contiguous()
                q_fft = torch.fft.rfft(q_input, dim=-2)  # FFT along sequence dimension
                k_fft = torch.fft.rfft(k_input, dim=-2)
                
            else:
                # Downsample for multi-scale analysis - handle 4D properly
                target_len = max(length // scale, 1)
                q_downsampled = torch.zeros(B, H, target_len, E, device=queries.device)
                k_downsampled = torch.zeros(B, H, target_len, E, device=keys.device)
                
                # Process each head separately for F.interpolate compatibility
                for h in range(H):
                    for e in range(E):
                        q_downsampled[:, h, :, e] = F.interpolate(
                            queries[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                        k_downsampled[:, h, :, e] = F.interpolate(
                            keys[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                
                q_fft = torch.fft.rfft(q_downsampled, dim=-2)
                k_fft = torch.fft.rfft(k_downsampled, dim=-2)
            
            # Compute correlation in frequency domain
            correlation = q_fft * torch.conj(k_fft)
            
            # Apply learnable frequency filtering
            if hasattr(self, 'frequency_filter'):
                correlation = correlation * self.frequency_filter
            
            # Transform back to time domain
            correlation_time = torch.fft.irfft(correlation, n=length if scale == 1 else target_len, dim=-2)
            
            # Upsample back to original length if needed
            if scale != 1:
                correlation_upsampled = torch.zeros(B, H, length, E, device=correlation_time.device)
                for h in range(H):
                    for e in range(E):
                        correlation_upsampled[:, h, :, e] = F.interpolate(
                            correlation_time[:, h, :, e].unsqueeze(1), size=length,
                            mode='linear', align_corners=False
                        ).squeeze(1)
                correlation_time = correlation_upsampled
            
            correlations.append(correlation_time)
        
        # Weighted combination of multi-scale correlations
        if self.multi_scale and len(correlations) > 1:
            weights = F.softmax(self.scale_weights, dim=0)
            correlation_combined = sum(w * corr for w, corr in zip(weights, correlations))
        else:
            correlation_combined = correlations[0]
        
        return correlation_combined

    def _correlation_based_attention(self, queries, keys, values, length):
        """
        Compute attention weights based on autocorrelation.
        """
        B, H, L, E = queries.shape
        
        # Compute multi-scale correlations
        correlation = self._multi_scale_correlation(queries, keys, length)
        
        # Compute correlation energy for adaptive k selection
        correlation_energy = torch.mean(torch.abs(correlation), dim=[1, 3])  # [B, L]
        
        # Select adaptive k
        if self.adaptive_k:
            k = self._select_adaptive_k(correlation_energy, length)
        else:
            k = int(self.factor * math.log(length))
        
        k = max(k, 1)  # Ensure k is at least 1
        
        # Find top-k correlations
        mean_correlation = torch.mean(correlation, dim=1)  # [B, L, E]
        _, top_k_indices = torch.topk(torch.mean(torch.abs(mean_correlation), dim=-1), k, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(values)
        weights = torch.zeros(B, H, L, L, device=queries.device)
        
        # Apply correlation-based attention
        for b in range(B):
            for h in range(H):
                for i, lag in enumerate(top_k_indices[b]):
                    lag = lag.item()
                    
                    # Circular shift for autocorrelation
                    if lag > 0:
                        shifted_values = torch.cat([
                            values[b, h, -lag:, :], 
                            values[b, h, :-lag, :]
                        ], dim=0)
                    else:
                        shifted_values = values[b, h, :, :]
                    
                    # Compute attention weight based on correlation strength
                    corr_strength = torch.abs(mean_correlation[b, lag, :]).mean()
                    weight = F.softmax(torch.tensor([corr_strength]), dim=0)[0]
                    
                    output[b, h, :, :] += weight * shifted_values
                    
                    # Store attention weights for visualization
                    if lag > 0:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device).roll(-lag, dims=0)
                    else:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device)
        
        return output, weights

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass of enhanced autocorrelation attention.
        """
        # Handle different input shapes
        if queries.dim() == 4:
            # Input is [B, H, L, E] - reshape to [B, L, D]
            B, H, L, E = queries.shape
            D = H * E
            queries = queries.transpose(1, 2).contiguous().view(B, L, D)
            keys = keys.transpose(1, 2).contiguous().view(B, L, D)
            values = values.transpose(1, 2).contiguous().view(B, L, D)
        else:
            # Input is [B, L, D]
            B, L, D = queries.shape
            H = self.n_heads
            E = D // H
        
        # Apply projections
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        
        # Reshape for multi-head processing
        queries = queries.view(B, L, H, E).transpose(1, 2)  # [B, H, L, E]
        keys = keys.view(B, L, H, E).transpose(1, 2)        # [B, H, L, E]
        values = values.view(B, L, H, E).transpose(1, 2)    # [B, H, L, E]
        
        # Apply correlation-based attention
        output, attn_weights = self._correlation_based_attention(queries, keys, values, L)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape back and apply output projection
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_projection(output)
        
        if self.output_attention:
            return output, attn_weights
        else:
            return output, None

    def __init__(self, d_model, n_heads, factor=1, scale=None, attention_dropout=0.1, 
                 output_attention=False, adaptive_k=True, multi_scale=True, 
                 scales=[1, 2, 4], eps=1e-8):
        super(EnhancedAutoCorrelation, self).__init__()
        logger.info(f"Initializing EnhancedAutoCorrelation with enhanced features: d_model={d_model}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Adaptive parameters
        self.adaptive_k = adaptive_k
        self.multi_scale = multi_scale
        self.scales = scales
        self.eps = eps
        
        # Learnable components
        if self.multi_scale:
            self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
            
        # Frequency filter for noise reduction
        self.frequency_filter = nn.Parameter(torch.ones(1))
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.output_dim_multiplier = 1

    def _select_adaptive_k(self, corr_energy, length):
        """
        Intelligently select the number of correlation peaks to use.
        
        Args:
            corr_energy: [B, L] correlation energy per time lag
            length: sequence length
            
        Returns:
            optimal k value
        """
        # Sort correlation energies in descending order
        sorted_energies, _ = torch.sort(corr_energy, dim=-1, descending=True)
        
        # Find the elbow point using second derivative
        if length > 10:  # Only for reasonable sequence lengths
            # Compute first and second derivatives
            first_diff = sorted_energies[:, :-1] - sorted_energies[:, 1:]
            second_diff = first_diff[:, :-1] - first_diff[:, 1:]
            
            # Find elbow as point of maximum curvature
            elbow_candidates = torch.argmax(second_diff, dim=-1) + 2  # +2 due to double differencing
            
            # Ensure reasonable bounds
            min_k = max(2, int(0.1 * math.log(length)))
            max_k = min(int(0.3 * length), int(self.factor * math.log(length) * 2))
            
            if min_k > max_k:
                max_k = min_k
            
            adaptive_k = torch.clamp(elbow_candidates, min_k, max_k)
            
            # Use median across batch for stability
            return int(torch.median(adaptive_k.float()).item())
        else:
            return max(2, int(self.factor * math.log(length)))

    def _multi_scale_correlation(self, queries, keys, length):
        """
        Compute correlations at multiple scales to capture different periodicities.
        
        Args:
            queries, keys: input tensors [B, H, L, E]
            length: sequence length
            
        Returns:
            aggregated multi-scale correlations
        """
        correlations = []
        B, H, L, E = queries.shape
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # Original scale - ensure contiguous tensors
                q_input = queries.contiguous()
                k_input = keys.contiguous()
                q_fft = torch.fft.rfft(q_input, dim=-2)  # FFT along sequence dimension
                k_fft = torch.fft.rfft(k_input, dim=-2)
                
            else:
                # Downsample for multi-scale analysis - handle 4D properly
                target_len = max(length // scale, 1)
                q_downsampled = torch.zeros(B, H, target_len, E, device=queries.device)
                k_downsampled = torch.zeros(B, H, target_len, E, device=keys.device)
                
                # Process each head separately for F.interpolate compatibility
                for h in range(H):
                    for e in range(E):
                        q_downsampled[:, h, :, e] = F.interpolate(
                            queries[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                        k_downsampled[:, h, :, e] = F.interpolate(
                            keys[:, h, :, e].unsqueeze(1), size=target_len, 
                            mode='linear', align_corners=False
                        ).squeeze(1)
                
                q_fft = torch.fft.rfft(q_downsampled, dim=-2)
                k_fft = torch.fft.rfft(k_downsampled, dim=-2)
            
            # Compute correlation in frequency domain
            correlation = q_fft * torch.conj(k_fft)
            
            # Apply learnable frequency filtering
            if hasattr(self, 'frequency_filter'):
                correlation = correlation * self.frequency_filter
            
            # Transform back to time domain
            correlation_time = torch.fft.irfft(correlation, n=length if scale == 1 else target_len, dim=-2)
            
            # Upsample back to original length if needed
            if scale != 1:
                correlation_upsampled = torch.zeros(B, H, length, E, device=correlation_time.device)
                for h in range(H):
                    for e in range(E):
                        correlation_upsampled[:, h, :, e] = F.interpolate(
                            correlation_time[:, h, :, e].unsqueeze(1), size=length,
                            mode='linear', align_corners=False
                        ).squeeze(1)
                correlation_time = correlation_upsampled
            
            correlations.append(correlation_time)
        
        # Weighted combination of multi-scale correlations
        if self.multi_scale and len(correlations) > 1:
            weights = F.softmax(self.scale_weights, dim=0)
            correlation_combined = sum(w * corr for w, corr in zip(weights, correlations))
        else:
            correlation_combined = correlations[0]
        
        return correlation_combined

    def _correlation_based_attention(self, queries, keys, values, length):
        """
        Compute attention weights based on autocorrelation.
        
        Args:
            queries: [B, H, L, E] query tensor
            keys: [B, H, L, E] key tensor  
            values: [B, H, L, E] value tensor
            length: sequence length
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, H, L, E = queries.shape
        
        # Compute multi-scale correlations
        correlation = self._multi_scale_correlation(queries, keys, length)
        
        # Compute correlation energy for adaptive k selection
        correlation_energy = torch.mean(torch.abs(correlation), dim=[1, 3])  # [B, L]
        
        # Select adaptive k
        if self.adaptive_k:
            k = self._select_adaptive_k(correlation_energy, length)
        else:
            k = int(self.factor * math.log(length))
        
        k = max(k, 1)  # Ensure k is at least 1
        
        # Find top-k correlations
        mean_correlation = torch.mean(correlation, dim=1)  # [B, L, E]
        _, top_k_indices = torch.topk(torch.mean(torch.abs(mean_correlation), dim=-1), k, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(values)
        weights = torch.zeros(B, H, L, L, device=queries.device)
        
        # Apply correlation-based attention
        for b in range(B):
            for h in range(H):
                for i, lag in enumerate(top_k_indices[b]):
                    lag = lag.item()
                    
                    # Circular shift for autocorrelation
                    if lag > 0:
                        shifted_values = torch.cat([
                            values[b, h, -lag:, :], 
                            values[b, h, :-lag, :]
                        ], dim=0)
                    else:
                        shifted_values = values[b, h, :, :]
                    
                    # Compute attention weight based on correlation strength
                    corr_strength = torch.abs(mean_correlation[b, lag, :]).mean()
                    weight = F.softmax(torch.tensor([corr_strength]), dim=0)[0]
                    
                    output[b, h, :, :] += weight * shifted_values
                    
                    # Store attention weights for visualization
                    if lag > 0:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device).roll(-lag, dims=0)
                    else:
                        weights[b, h, :, :] += weight * torch.eye(L, device=queries.device)
        
        return output, weights

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass of enhanced autocorrelation attention.
        
        Args:
            queries: [B, L, D] or [B, H, L, E] query tensor
            keys: [B, L, D] or [B, H, L, E] key tensor
            values: [B, L, D] or [B, H, L, E] value tensor
            attn_mask: Optional attention mask
            tau: Temperature parameter (optional)
            delta: Delta parameter (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        
        # Handle different input shapes
        if queries.dim() == 4:
            # Input is [B, H, L, E] - reshape to [B, L, D]
            B, H, L, E = queries.shape
            D = H * E
            queries = queries.transpose(1, 2).contiguous().view(B, L, D)
            keys = keys.transpose(1, 2).contiguous().view(B, L, D)
            values = values.transpose(1, 2).contiguous().view(B, L, D)
        else:
            # Input is [B, L, D]
            B, L, D = queries.shape
            H = self.n_heads
            E = D // H
        
        # Apply projections
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        
        # Reshape for multi-head processing
        queries = queries.view(B, L, H, E).transpose(1, 2)  # [B, H, L, E]
        keys = keys.view(B, L, H, E).transpose(1, 2)        # [B, H, L, E]
        values = values.view(B, L, H, E).transpose(1, 2)    # [B, H, L, E]
        
        # Apply correlation-based attention
        output, attn_weights = self._correlation_based_attention(queries, keys, values, L)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape back and apply output projection
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_projection(output)
        
        if self.output_attention:
            return output, attn_weights
        else:
            return output, None



# Modular Convolution family imports
from .Attention.Convolution.causal_convolution import CausalConvolution
from .Attention.Convolution.convolutional_attention import ConvolutionalAttention
        self.positional_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        self.output_dim_multiplier = 1
    
    def apply_causal_mask(self, x, kernel_size, dilation):
        """
        Apply causal masking by removing future information.
        """
        trim_amount = (kernel_size - 1) * dilation
        if trim_amount > 0:
            x = x[:, :, :-trim_amount]
        return x
    
    def multi_scale_causal_conv(self, x):
        """
        Apply multi-scale causal convolutions.
        """
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)  # [B, D, L] for convolution
        
        scale_outputs = []
        
        for kernel_idx, conv_layers in enumerate(self.causal_convs):
            kernel_size = self.kernel_sizes[kernel_idx]
            
            # Apply dilated convolutions at current kernel size
            dilated_outputs = []
            for dilation_idx, conv_layer in enumerate(conv_layers):
                dilation = self.dilation_rates[dilation_idx]
                
                # Apply convolution
                conv_out = conv_layer(x_conv)
                
                # Apply causal masking
                conv_out = self.apply_causal_mask(conv_out, kernel_size, dilation)
                
                # Pad to original length if needed
                if conv_out.size(-1) < L:
                    padding_needed = L - conv_out.size(-1)
                    conv_out = F.pad(conv_out, (0, padding_needed))
                
                # Apply activation
                conv_out = self.activation(conv_out)
                dilated_outputs.append(conv_out)
            
            # Combine dilated outputs (average)
            combined_output = torch.stack(dilated_outputs, dim=0).mean(dim=0)
            scale_outputs.append(combined_output)
        
        return scale_outputs
    
    def temporal_attention(self, queries, keys, values):
        """
        Compute temporal attention with convolution features.
        """
        B, D, L = queries.shape
        H = self.n_heads
        d_k = D // H
        
        # Reshape for multi-head processing
        q = queries.view(B, H, d_k, L).transpose(2, 3)  # [B, H, L, d_k]
        k = keys.view(B, H, d_k, L).transpose(2, 3)     # [B, H, L, d_k]
        v = values.view(B, H, d_k, L).transpose(2, 3)   # [B, H, L, d_k]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        
        # Apply causal mask (future positions set to -inf)
        causal_mask = torch.triu(torch.ones(L, L, device=queries.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, H, L, d_k]
        
        # Reshape back
        context = context.transpose(2, 3).contiguous().view(B, D, L)
        
        return context, attn_weights
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with causal convolution attention.
        """
        B, L, D = queries.shape
        
        residual = queries
        
        # Apply positional encoding
        pos_encoded = queries + self.positional_conv(queries.transpose(1, 2)).transpose(1, 2)
        
        # Multi-scale causal convolutions
        conv_outputs = self.multi_scale_causal_conv(pos_encoded)
        
        # Convert to attention format
        q_conv = self.query_conv(queries.transpose(1, 2))    # [B, D, L]
        k_conv = self.key_conv(keys.transpose(1, 2))         # [B, D, L]
        v_conv = self.value_conv(values.transpose(1, 2))     # [B, D, L]
        
        # Add convolution features to keys and values
        enhanced_keys = k_conv
        enhanced_values = v_conv
        
        for conv_out in conv_outputs:
            enhanced_keys = enhanced_keys + conv_out
            enhanced_values = enhanced_values + conv_out
        
        # Apply temporal attention
        attn_output, attn_weights = self.temporal_attention(q_conv, enhanced_keys, enhanced_values)
        
        # Combine multi-scale outputs
        combined_conv = torch.cat(conv_outputs, dim=1)  # [B, D*len(kernels), L]
        combined_conv = combined_conv.transpose(1, 2)   # [B, L, D*len(kernels)]
        
        # Project to original dimension
        conv_features = self.output_projection(combined_conv)
        
        # Combine attention and convolution features
        final_output = attn_output.transpose(1, 2) + conv_features
        
        # Residual connection and normalization
        output = self.layer_norm(final_output + residual)
        
        return output, attn_weights


class TemporalConvNet(BaseAttention):
    """
    Temporal Convolution Network (TCN) based attention mechanism.
    
    This implements a full TCN architecture for sequence modeling
    with attention-like interfaces.
    """
    def __init__(self, d_model, n_heads, num_levels=4, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        logger.info(f"Initializing TemporalConvNet: levels={num_levels}, kernel={kernel_size}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        
        # Build TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList()
        dilation_size = 1
        
        for i in range(num_levels):
            # Residual block with dilated convolution
            layer = TemporalBlock(
                d_model, d_model, kernel_size, 
                stride=1, dilation=dilation_size, 
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            )
            self.tcn_layers.append(layer)
            dilation_size *= 2  # Exponential dilation
        
        # Attention mechanism for combining temporal features
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Output processing
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through TCN with attention combination.
        """
        B, L, D = queries.shape
        
        # Use keys as primary input for TCN processing
        x = keys
        residual = x
        
        # Apply TCN layers sequentially
        temporal_features = []
        current_input = x.transpose(1, 2)  # [B, D, L] for conv
        
        for tcn_layer in self.tcn_layers:
            current_input = tcn_layer(current_input)
            # Store features from each level
            temporal_features.append(current_input.transpose(1, 2))  # Back to [B, L, D]
        
        # Final TCN output
        tcn_output = current_input.transpose(1, 2)  # [B, L, D]
        
        # Apply attention to combine with queries
        attn_output, attn_weights = self.temporal_attention(
            queries, tcn_output, values, attn_mask=attn_mask
        )
        
        # Apply output processing
        output = self.output_norm(attn_output + residual)
        output = self.output_dropout(output)
        
        return output, attn_weights
class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with residual connections and causal convolutions.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        
        # Apply causal masking by chopping off the future
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Downsample for residual connection if needed
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize convolution weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """Forward pass through temporal block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """
    Remove rightmost elements to ensure causality in convolutions.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        """Remove future information from convolution output."""
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class AdaptiveWaveletAttention(BaseAttention):


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
                scale_feat = F.avg_pool1d(query.transpose(1, 2), kernel_size=scale, stride=scale)
            
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
    """
    Convolutional attention mechanism combining spatial and temporal convolutions.
    
    This component uses 2D convolutions to capture both spatial (feature)
    and temporal relationships in attention computation.
    """
    def __init__(self, d_model, n_heads, conv_kernel_size=3, pool_size=2, dropout=0.1):
        super(ConvolutionalAttention, self).__init__()
        logger.info(f"Initializing ConvolutionalAttention: d_model={d_model}")
        
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
class TwoStageAttention(BaseAttention):
    """
    Modular implementation of Two-Stage Attention (TSA).
    This combines cross-time and cross-dimension attention for segment merging and multi-variate time series.
    Algorithm adapted from TwoStageAttentionLayer in SelfAttention_Family.py.
    """
    def __init__(self, d_model: int, n_heads: int, seg_num: int, factor: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
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

    def forward(self, x: torch.Tensor, attn_mask=None, tau=None, delta=None):
        # x: [batch, ts_d, seg_num, d_model]
        batch = x.shape[0]
        ts_d = x.shape[1]
        seg_num = x.shape[2]
        d_model = x.shape[3]
        # Cross Time Stage
        time_in = x.reshape(-1, seg_num, d_model)
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
        return final_out, None
class MultiScaleWaveletAttention(BaseAttention):
    """
    Multi-scale wavelet attention for capturing patterns at different time scales.
    
    This component applies wavelet attention at multiple predefined scales
    and combines the results for comprehensive temporal modeling.
    """
    
    def __init__(self, d_model, n_heads, scales=[1, 2, 4, 8]):
        super(MultiScaleWaveletAttention, self).__init__()
        logger.info(f"Initializing MultiScaleWaveletAttention: scales={scales}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        
        # Wavelet attention for each scale
        self.scale_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=3)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_proj = nn.Linear(d_model * len(scales), d_model)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with multi-scale wavelet processing.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        scale_outputs = []
        scale_attentions = []
        
        for scale, scale_attn in zip(self.scales, self.scale_attentions):
            if scale == 1:
                # Original scale
                q_scaled, k_scaled, v_scaled = queries, keys, values
            else:
                # Downsample for larger scales
                target_len = max(L // scale, 1)
                q_scaled = F.interpolate(queries.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                k_scaled = F.interpolate(keys.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                v_scaled = F.interpolate(values.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
            
            # Apply wavelet attention at this scale
            output, attention = scale_attn(q_scaled, k_scaled, v_scaled, attn_mask)
            
            # Upsample back to original length if needed
            if output.size(1) != L:
                output = F.interpolate(output.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            scale_outputs.append(output)
            scale_attentions.append(attention)
        
        # Weighted fusion of multi-scale outputs
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Concatenate and project
        concatenated = torch.cat(scale_outputs, dim=-1)  # [B, L, D*len(scales)]
        fused_output = self.scale_proj(concatenated)      # [B, L, D]
        
        # Average attention weights
        avg_attention = torch.stack(scale_attentions, dim=0).mean(dim=0)
        
        return fused_output, avg_attention
class AdaptiveWaveletAttention(BaseAttention):
    """
    Adaptive wavelet attention with learnable decomposition levels.
    
    This component adaptively selects the optimal number of decomposition
    levels based on the input characteristics.
    """
    
    def __init__(self, d_model, n_heads, max_levels=5):
        super(AdaptiveWaveletAttention, self).__init__()
        logger.info(f"Initializing AdaptiveWaveletAttention: d_model={d_model}, max_levels={max_levels}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_levels = max_levels
        
        # Level selection network
        self.level_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_levels),
            nn.Softmax(dim=-1)
        )
        
        # Multiple wavelet attention modules for different levels
        self.wavelet_attentions = nn.ModuleList([
            WaveletAttention(d_model, n_heads, levels=i+1)
            for i in range(max_levels)
        ])
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass with adaptive level selection.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Select optimal decomposition levels based on input characteristics
        input_summary = queries.mean(dim=1)  # [B, D]
        level_weights = self.level_selector(input_summary)  # [B, max_levels]
        
        # Compute outputs for all levels
        level_outputs = []
        level_attentions = []
        
        for i, wavelet_attn in enumerate(self.wavelet_attentions):
            output, attention = wavelet_attn(queries, keys, values, attn_mask)
            level_outputs.append(output)
            level_attentions.append(attention)
        
        # Weighted combination based on adaptive selection
        level_outputs = torch.stack(level_outputs, dim=-1)  # [B, L, D, max_levels]
        level_attentions = torch.stack(level_attentions, dim=-1)  # [B, H, L, L, max_levels]
        
        # Apply level weights
        final_output = torch.sum(level_outputs * level_weights.view(B, 1, 1, -1), dim=-1)
        final_attention = torch.sum(level_attentions * level_weights.view(B, 1, 1, 1, -1), dim=-1)
        
        return final_output, final_attention

# =============================================================================
# UNIFIED REGISTRY WITH ALL COMPONENTS
# =============================================================================
ATTENTION_REGISTRY = {
    # 'multihead': MultiHeadAttention,  # Removed, use torch.nn.MultiheadAttention directly
    # 'multi_head': MultiHeadAttention,  # Removed, use torch.nn.MultiheadAttention directly
    'autocorrelation': AutoCorrelationAttention,
    'sparse': SparseAttention,
    'fourier': FourierAttention,
    'wavelet': WaveletAttention,
    'bayesian': BayesianAttention,
    'adaptive': AdaptiveAttention,
    'meta_learning': MetaLearningAdapter,
    'adaptive_mixture': AdaptiveMixture,
    'enhanced_autocorrelation': EnhancedAutoCorrelation,
    'causal_convolution': CausalConvolution,
    'temporal_conv_net': TemporalConvNet,
    'adaptive_wavelet': AdaptiveWaveletAttention,
    'multi_scale_wavelet': MultiScaleWaveletAttention,
    'log_sparse': LogSparseAttention,
    'prob_sparse': ProbSparseAttention,
    'cross_resolution': CrossResolutionAttention,
    # Missing components from attention_migrated.py
    'bayesian_multihead': BayesianMultiHeadAttention,
    'variational': VariationalAttention,
    'bayesian_cross': BayesianCrossAttention,
    'adaptive_autocorrelation_layer': AdaptiveAutoCorrelationLayer,
    'hierarchical_autocorrelation': HierarchicalAutoCorrelation,
    'fourier_block': FourierBlock,
    'fourier_cross_attention': FourierCrossAttention,
    'convolutional': ConvolutionalAttention,
    'two_stage': TwoStageAttention,
    'exponential_smoothing': ExponentialSmoothingAttention,
    'multi_wavelet_cross': MultiWaveletCrossAttention,
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
