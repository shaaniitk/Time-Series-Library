"""
Fourier-based Attention Components for Modular Autoformer

This module implements Fourier-domain attention mechanisms that excel at
capturing periodic patterns and frequency-domain relationships in time series.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# Configure logger
logger = logging.getLogger(__name__)
import numpy as np
from .base import BaseAttention
from utils.logger import logger


class FourierAttention(BaseAttention):
    """
    Fourier-based attention for capturing periodic patterns in time series.
    
    This component uses frequency domain analysis to identify and emphasize
    periodic patterns, making it particularly effective for seasonal forecasting.
    """
    
    def __init__(self, d_model, n_heads, seq_len=96, frequency_selection='adaptive', 
                 dropout=0.1, temperature=1.0, learnable_filter=True):
        super(FourierAttention, self).__init__()
        logger.info(f"Initializing FourierAttention: d_model={d_model}, n_heads={n_heads}, seq_len={seq_len}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature
        self.learnable_filter = learnable_filter
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Learnable frequency components - fix initialization
        max_freq_dim = max(seq_len // 2 + 1, 64)  # Ensure minimum size
        self.freq_weights = nn.Parameter(torch.randn(max_freq_dim))
        self.phase_weights = nn.Parameter(torch.zeros(max_freq_dim))
        
        # Standard attention components
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Frequency selection strategy
        self.frequency_selection = frequency_selection
        if frequency_selection == 'adaptive':
            self.freq_selector = nn.Linear(d_model, max_freq_dim)
        
        self.output_dim_multiplier = 1
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass using Fourier-domain attention with complex filtering.
        
        Args:
            queries: [B, L, D] query tensor
            keys: [B, L, D] key tensor (same as queries for self-attention)
            values: [B, L, D] value tensor
            attn_mask: Optional attention mask
            tau: Temperature parameter (optional)
            delta: Delta parameter (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = queries.shape
        
        # Ensure input is contiguous for FFT operations
        queries = queries.contiguous()
        
        try:
            # Use alternative FFT approach to avoid Intel MKL issues
            queries = queries.contiguous()
            
            # Check if sequence length is compatible with FFT
            if L < 2:
                raise RuntimeError("Sequence length too short for FFT")
            
            # Try multiple FFT approaches
            queries_freq = None
            approaches = [
                lambda x: torch.fft.rfft(x, dim=1, norm='ortho'),  # Orthogonal normalization
                lambda x: torch.fft.fft(x, dim=1)[:, :L//2+1],    # Manual truncation
                lambda x: self._manual_dft(x)                      # Manual DFT implementation
            ]
            
            for i, approach in enumerate(approaches):
                try:
                    queries_freq = approach(queries)
                    freq_dim = queries_freq.shape[1]
                    logger.debug(f"FFT approach {i+1} succeeded with freq_dim={freq_dim}")
                    break
                except Exception as e:
                    logger.debug(f"FFT approach {i+1} failed: {e}")
                    continue
            
            if queries_freq is None:
                raise RuntimeError("All FFT approaches failed")
                
            # Validate frequency dimensions
            if freq_dim <= 0:
                raise RuntimeError("Invalid frequency dimension")
                
        except (RuntimeError, Exception) as e:
            logger.warning(f"All FFT operations failed: {e}, using fallback frequency simulation")
            # Fallback: simulate frequency filtering using conv1d
            queries_filtered = self._frequency_fallback(queries)
            return self._standard_attention(queries_filtered, attn_mask)
        
        try:
            
            # Adaptive frequency selection if enabled
            if self.frequency_selection == 'adaptive':
                freq_weights_input = queries.mean(dim=1)  # [B, D]
                
                # Ensure freq_selector output matches frequency dimensions
                if hasattr(self, 'freq_selector'):
                    freq_logits = self.freq_selector(freq_weights_input)  # [B, max_freq_dim]
                    # Trim to actual frequency dimension
                    freq_logits = freq_logits[:, :freq_dim]
                    freq_selection_weights = F.softmax(freq_logits, dim=-1)  # [B, freq_dim]
                else:
                    # Fallback: uniform weights
                    freq_selection_weights = torch.ones(B, freq_dim, device=queries.device) / freq_dim
                
                # Create frequency filter with proper dimensions
                phase_weights = self.phase_weights[:freq_dim].unsqueeze(0)  # [1, freq_dim]
                freq_weights = self.freq_weights[:freq_dim].unsqueeze(0)    # [1, freq_dim]
                
                # Complex frequency filter with adaptive selection
                freq_filter = torch.complex(
                    torch.cos(phase_weights) * freq_weights * freq_selection_weights,
                    torch.sin(phase_weights) * freq_weights * freq_selection_weights
                )  # [B, freq_dim]
            else:
                # Create frequency filter with proper dimensions
                phase_weights = self.phase_weights[:freq_dim]  # [freq_dim]
                freq_weights = self.freq_weights[:freq_dim]    # [freq_dim]
                
                freq_filter = torch.complex(
                    torch.cos(phase_weights) * freq_weights,
                    torch.sin(phase_weights) * freq_weights
                )  # [freq_dim]
                freq_filter = freq_filter.unsqueeze(0)  # [1, freq_dim]
            
            # Apply complex frequency filtering with correct broadcasting
            freq_filter = freq_filter.unsqueeze(-1)  # [B, freq_dim, 1] or [1, freq_dim, 1]
            queries_freq_filtered = queries_freq * freq_filter  # [B, freq_dim, D]
            
            # Transform back to time domain with error handling and multiple approaches
            try:
                inverse_approaches = [
                    lambda x: torch.fft.irfft(x, n=L, dim=1, norm='ortho'),
                    lambda x: torch.real(torch.fft.ifft(F.pad(x, (0, 0, 0, L-x.shape[1])), dim=1)),
                    lambda x: self._manual_idft(x, L)
                ]
                
                for i, inv_approach in enumerate(inverse_approaches):
                    try:
                        queries_filtered = inv_approach(queries_freq_filtered)
                        logger.debug(f"Inverse FFT approach {i+1} succeeded")
                        break
                    except Exception as e:
                        logger.debug(f"Inverse FFT approach {i+1} failed: {e}")
                        continue
                        
                if 'queries_filtered' not in locals():
                    raise RuntimeError("All inverse FFT approaches failed")
                    
            except Exception as e:
                logger.warning(f"Inverse FFT failed: {e}, using frequency fallback")
                queries_filtered = self._frequency_fallback(queries)
                return self._standard_attention(queries_filtered, attn_mask)
            
            # Ensure correct output length
            if queries_filtered.shape[1] != L:
                queries_filtered = F.interpolate(
                    queries_filtered.transpose(1, 2), 
                    size=L, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
                
        except Exception as e:
            logger.warning(f"FFT operation failed: {e}, falling back to standard attention")
            # Fallback to standard processing without FFT
            queries_filtered = queries
        
        # Standard multi-head attention on filtered signal
        qkv = self.qkv(queries_filtered).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D/H]
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights

    def _manual_dft(self, x):
        """
        Manual DFT implementation as fallback for problematic FFT configurations.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            x_freq: [B, L//2+1, D] frequency domain representation
        """
        B, L, D = x.shape
        
        # Create frequency indices
        freqs = torch.arange(0, L//2 + 1, device=x.device, dtype=torch.float32)
        times = torch.arange(0, L, device=x.device, dtype=torch.float32)
        
        # Compute DFT manually (simplified version)
        x_freq = torch.zeros(B, L//2 + 1, D, dtype=torch.complex64, device=x.device)
        
        for k in range(L//2 + 1):
            # e^(-2πi * k * n / L)
            phase = -2 * math.pi * k * times / L
            kernel_real = torch.cos(phase)
            kernel_imag = torch.sin(phase)
            
            # Apply DFT kernel
            real_part = torch.sum(x * kernel_real.view(1, -1, 1), dim=1)
            imag_part = torch.sum(x * kernel_imag.view(1, -1, 1), dim=1)
            
            x_freq[:, k, :] = torch.complex(real_part, imag_part)
        
        return x_freq

    def _manual_idft(self, x_freq, target_length):
        """
        Manual inverse DFT implementation.
        
        Args:
            x_freq: [B, F, D] frequency domain tensor
            target_length: Target time series length
            
        Returns:
            x_time: [B, target_length, D] time domain representation
        """
        B, F, D = x_freq.shape
        L = target_length
        
        # Create time and frequency indices
        times = torch.arange(0, L, device=x_freq.device, dtype=torch.float32)
        freqs = torch.arange(0, F, device=x_freq.device, dtype=torch.float32)
        
        # Initialize output
        x_time = torch.zeros(B, L, D, device=x_freq.device)
        
        for n in range(L):
            # e^(2πi * k * n / L)
            phase = 2 * math.pi * freqs * n / L
            kernel_real = torch.cos(phase)
            kernel_imag = torch.sin(phase)
            
            # Apply inverse DFT
            real_contrib = torch.sum(
                torch.real(x_freq) * kernel_real.view(1, -1, 1) - 
                torch.imag(x_freq) * kernel_imag.view(1, -1, 1), 
                dim=1
            )
            
            x_time[:, n, :] = real_contrib / L
        
        return x_time

    def _frequency_fallback(self, x):
        """
        Fallback frequency filtering using 1D convolution when FFT fails.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            filtered: [B, L, D] frequency-filtered output
        """
        B, L, D = x.shape
        
        # Transpose for conv1d: [B, D, L]
        x_conv = x.transpose(1, 2)
        
        # Apply learnable frequency filters (simulate band-pass filtering)
        freq_kernels = self.freq_weights[:min(3, len(self.freq_weights))].view(1, 1, -1)
        
        # Pad for convolution
        padding = freq_kernels.shape[-1] // 2
        x_padded = F.pad(x_conv, (padding, padding), mode='reflect')
        
        # Apply convolution as frequency filter
        filtered = F.conv1d(x_padded, freq_kernels.expand(D, 1, -1), groups=D)
        
        # Ensure correct output length and transpose back
        if filtered.shape[-1] != L:
            filtered = F.interpolate(filtered, size=L, mode='linear', align_corners=False)
        
        return filtered.transpose(1, 2)  # [B, L, D]

    def _standard_attention(self, x, attn_mask=None):
        """
        Standard multi-head attention fallback.
        
        Args:
            x: [B, L, D] input tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, D = x.shape
        
        # Standard multi-head attention
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D/H]
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn_weights


class FourierBlock(BaseAttention):
    """
    1D Fourier block for frequency domain representation learning.
    
    Performs FFT, learnable linear transformation in frequency domain,
    and inverse FFT to return to time domain.
    """
    
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=64, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        logger.info(f"Initializing FourierBlock: {in_channels} -> {out_channels}, modes={modes}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.seq_len = seq_len
        
        # Get frequency modes
        self.index = self._get_frequency_modes(seq_len, modes, mode_select_method)
        logger.info(f"Selected frequency modes: {self.index}")
        
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable frequency domain weights (complex)
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index))
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index))
        )
        
        self.output_dim_multiplier = out_channels / in_channels
    
    def _get_frequency_modes(self, seq_len, modes=64, mode_select_method='random'):
        """Select frequency modes for processing."""
        modes = min(modes, seq_len // 2)
        if mode_select_method == 'random':
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index
    
    def _complex_mul1d(self, x, weights_real, weights_imag):
        """Complex multiplication in frequency domain."""
        # Convert to complex if needed
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        
        weights = torch.complex(weights_real, weights_imag)
        
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        return torch.einsum("bhio,bhio->bho", x, weights)
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through Fourier block.
        
        Args:
            queries: [B, L, H, E] input tensor (reshaped from [B, L, D])
            keys: Not used in this implementation
            values: Not used in this implementation
            
        Returns:
            Tuple of (output, None)
        """
        # Assume queries is [B, L, D] and reshape to [B, H, E, L]
        B, L, D = queries.shape
        H = self.n_heads
        E = D // H
        
        x = queries.view(B, L, H, E).permute(0, 2, 3, 1)  # [B, H, E, L]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= len(self.index):
                continue
            out_ft[:, :, :, wi] = self._complex_mul1d(
                x_ft[:, :, :, i:i+1], 
                self.weights_real[:, :, :, wi:wi+1],
                self.weights_imag[:, :, :, wi:wi+1]
            ).squeeze(-1)
        
        # Return to time domain
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        
        # Reshape back to [B, L, D]
        output = x_out.permute(0, 3, 1, 2).reshape(B, L, -1)
        
        return output, None


class FourierCrossAttention(BaseAttention):
    """
    Fourier-enhanced cross attention for encoder-decoder architectures.
    
    Performs cross-attention in frequency domain with different modes
    for queries and key-value pairs.
    """
    
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, 
                 mode_select_method='random', activation='tanh', num_heads=8):
        super(FourierCrossAttention, self).__init__()
        logger.info(f"Initializing FourierCrossAttention: {in_channels} -> {out_channels}")
        
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        # Get frequency modes for queries and keys/values
        self.index_q = self._get_frequency_modes(seq_len_q, modes, mode_select_method)
        self.index_kv = self._get_frequency_modes(seq_len_kv, modes, mode_select_method)
        
        logger.info(f"Query modes: {len(self.index_q)}, KV modes: {len(self.index_kv)}")
        
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable weights for cross-attention in frequency domain
        self.weights_q = nn.Parameter(self.scale * torch.rand(num_heads, in_channels, len(self.index_q), dtype=torch.cfloat))
        self.weights_kv = nn.Parameter(self.scale * torch.rand(num_heads, in_channels, len(self.index_kv), dtype=torch.cfloat))
        
        self.output_dim_multiplier = out_channels / in_channels
    
    def _get_frequency_modes(self, seq_len, modes=64, mode_select_method='random'):
        """Select frequency modes for processing."""
        modes = min(modes, seq_len // 2)
        if mode_select_method == 'random':
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass for cross-attention in frequency domain.
        
        Args:
            queries: [B, L_q, D] query tensor
            keys: [B, L_kv, D] key tensor
            values: [B, L_kv, D] value tensor
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L_q, D = queries.shape
        _, L_kv, _ = keys.shape
        
        # Transform to frequency domain
        q_ft = torch.fft.rfft(queries, dim=1)  # [B, L_q//2+1, D]
        k_ft = torch.fft.rfft(keys, dim=1)     # [B, L_kv//2+1, D]
        v_ft = torch.fft.rfft(values, dim=1)   # [B, L_kv//2+1, D]
        
        # Apply frequency domain transformations
        q_filtered = torch.zeros_like(q_ft)
        kv_filtered = torch.zeros_like(k_ft)
        
        # Process selected frequency modes
        for i, freq_idx in enumerate(self.index_q):
            if freq_idx < q_ft.shape[1]:
                q_filtered[:, i, :] = q_ft[:, freq_idx, :] * self.weights_q[0, :, i]
        
        for i, freq_idx in enumerate(self.index_kv):
            if freq_idx < k_ft.shape[1]:
                kv_filtered[:, i, :] = k_ft[:, freq_idx, :] * self.weights_kv[0, :, i]
        
        # Convert back to time domain for attention computation
        q_processed = torch.fft.irfft(q_filtered, n=L_q, dim=1)
        k_processed = torch.fft.irfft(kv_filtered, n=L_kv, dim=1)
        v_processed = torch.fft.irfft(kv_filtered, n=L_kv, dim=1)
        
        # Standard cross-attention
        scale = math.sqrt(D)
        attn_scores = torch.bmm(q_processed, k_processed.transpose(1, 2)) / scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.bmm(attn_weights, v_processed)
        
        return output, attn_weights
