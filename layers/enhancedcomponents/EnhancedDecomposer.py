"""
Enhanced Wavelet Hierarchical Decomposer for HierarchicalEnhancedAutoformer

This module provides a modular, reusable implementation of the hierarchical wavelet decomposer logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse
from utils.logger import logger

class WaveletHierarchicalDecomposer(nn.Module):
    """Hierarchical decomposition using existing DWT infrastructure."""
    def __init__(self, seq_len, d_model, wavelet_type='db4', levels=3, use_learnable_weights=True):
        super(WaveletHierarchicalDecomposer, self).__init__()
        logger.info(f"Initializing WaveletHierarchicalDecomposer with {levels} levels")
        self.seq_len, self.d_model, self.levels, self.wavelet_type = seq_len, d_model, levels, wavelet_type
        self.dwt_forward = DWT1DForward(J=levels, wave=wavelet_type, mode='symmetric')
        self.dwt_inverse = DWT1DInverse(wave=wavelet_type, mode='symmetric')
        if use_learnable_weights:
            self.scale_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        else:
            self.register_buffer('scale_weights', torch.ones(levels + 1) / (levels + 1))
        self.scale_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(levels + 1)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        min_len = 2 ** self.levels
        padding = 0
        if seq_len < min_len:
            logger.warning(f"Sequence length {seq_len} is shorter than DWT minimum of {min_len}. Padding to {min_len}.")
            padding = min_len - seq_len
            x = F.pad(x, (0, 0, 0, padding), 'replicate')
        
        try:
            low_freq, high_freqs = self.dwt_forward(x.transpose(1, 2))
        except (ValueError, RuntimeError) as e:
            logger.error(f"DWT failed: {e}", exc_info=True)
            return self._fallback_decomposition(x[:, :-padding, :] if padding > 0 else x)

        decomposed_scales = []
        low_freq = self._resize_to_target_length(low_freq.transpose(1, 2), seq_len // (2 ** self.levels))
        decomposed_scales.append(self.scale_projections[0](low_freq))
        
        for i, high_freq in enumerate(high_freqs):
            target_length = seq_len // (2 ** (self.levels - i))
            high_freq = self._resize_to_target_length(high_freq.transpose(1, 2), target_length)
            decomposed_scales.append(self.scale_projections[i + 1](high_freq))
        return decomposed_scales

    def _resize_to_target_length(self, x, target_length):
        if x.size(1) == target_length: return x
        mode = 'linear' if x.size(1) < target_length else 'area'
        return F.interpolate(x.transpose(1, 2), size=target_length, mode=mode).transpose(1, 2)

    def _fallback_decomposition(self, x):
        logger.info("Using fallback multi-scale decomposition")
        scales = [x]
        for _ in range(self.levels):
            x = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            scales.append(x)
        return scales
