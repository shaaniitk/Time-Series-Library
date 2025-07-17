import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDecomposition
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse
from utils.logger import logger

class WaveletHierarchicalDecomposition(BaseDecomposition):
    """
    Hierarchical decomposition using existing DWT infrastructure.
    
    Integrates with existing DWT_Decomposition.py to create multi-resolution
    time series representations.
    """
    
    def __init__(self, seq_len, d_model, wavelet_type='db4', levels=3, 
                 use_learnable_weights=True):
        super(WaveletHierarchicalDecomposition, self).__init__()
        logger.info(f"Initializing WaveletHierarchicalDecomposer with {levels} levels")
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.levels = levels
        self.wavelet_type = wavelet_type
        
        self.dwt_forward = DWT1DForward(J=levels, wave=wavelet_type, mode='symmetric')
        self.dwt_inverse = DWT1DInverse(wave=wavelet_type, mode='symmetric')
        
        if use_learnable_weights:
            self.scale_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        else:
            self.register_buffer('scale_weights', torch.ones(levels + 1) / (levels + 1))
        
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(levels + 1)
        ])
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        if seq_len < 2 ** self.levels:
            logger.warning(
                f"Sequence length {seq_len} is too short for {self.levels} DWT levels. "
                f"Required minimum length is {2 ** self.levels}. Using fallback decomposition."
            )
            return self._fallback_decomposition(x)

        x_dwt = x.transpose(1, 2)
        
        try:
            low_freq, high_freqs = self.dwt_forward(x_dwt)
        except (ValueError, RuntimeError) as e:
            logger.error(f"DWT failed with a critical error: {e}", exc_info=True)
            return self._fallback_decomposition(x)
        
        decomposed_scales = []
        
        low_freq = low_freq.transpose(1, 2)
        low_freq = self._resize_to_target_length(low_freq, seq_len // (2 ** self.levels))
        low_freq = self.scale_projections[0](low_freq)
        decomposed_scales.append(low_freq)
        
        for i, high_freq in enumerate(high_freqs):
            high_freq = high_freq.transpose(1, 2)
            target_length = seq_len // (2 ** (self.levels - i))
            high_freq = self._resize_to_target_length(high_freq, target_length)
            high_freq = self.scale_projections[i + 1](high_freq)
            decomposed_scales.append(high_freq)
        
        # For now, we must return a single seasonal and trend component.
        # We will reconstruct a single "seasonal" and "trend" from the multi-resolution components.
        # A simple approach is to treat the low_freq as trend and sum of high_freqs as seasonal.
        
        # Upsample all components to the original sequence length to combine them
        reconstructed_components = []
        for i, scale in enumerate(decomposed_scales):
            target_len = seq_len
            reconstructed_components.append(self._resize_to_target_length(scale, target_len))

        trend = reconstructed_components[0] # Coarsest level is the trend
        seasonal = sum(reconstructed_components[1:]) # Sum of details is the seasonal part

        return seasonal, trend

    def _resize_to_target_length(self, x, target_length):
        current_length = x.size(1)
        if current_length == target_length:
            return x
        elif current_length < target_length:
            x_resized = F.interpolate(
                x.transpose(1, 2), 
                size=target_length, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        else:
            x_resized = F.adaptive_avg_pool1d(
                x.transpose(1, 2), 
                target_length
            ).transpose(1, 2)
        
        return x_resized
    
    def _fallback_decomposition(self, x):
        logger.info("Using fallback multi-scale decomposition")
        scales = []
        current = x
        
        for i in range(self.levels + 1):
            if i == 0:
                scales.append(current)
            else:
                pooled = F.avg_pool1d(
                    current.transpose(1, 2), 
                    kernel_size=2, 
                    stride=2
                ).transpose(1, 2)
                scales.append(pooled)
                current = pooled
        
        trend = scales[-1]
        seasonal = x - self._resize_to_target_length(trend, x.size(1))
        return seasonal, trend