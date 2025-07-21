"""
Migrated Decomposition Components
Auto-migrated from layers/modular/decomposition to utils.modular_components.implementations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent, BaseLoss, BaseAttention
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)

# Migrated imports
    import inspect
from abc import ABC, abstractmethod
from utils.logger import logger
import torch.nn as nn
import torch
        from utils.logger import logger
import torch.nn.functional as F
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse

# Migrated Classes
class BaseDecomposition(nn.Module, ABC):
    """
    Abstract base class for all series decomposition components.
    """
    def __init__(self):
        super(BaseDecomposition, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Decomposes the input time series.

        Args:
            x (torch.Tensor): The input time series of shape [Batch, Seq_Len, Dims].

        Returns:
            tuple: A tuple containing two tensors:
                   - seasonal (torch.Tensor): The seasonal component.
                   - trend (torch.Tensor): The trend component.
        """
        pass

class LearnableSeriesDecomposition(BaseDecomposition):
    """
    Enhanced series decomposition with learnable parameters.
    
    Improvements over standard moving average:
    1. Learnable trend extraction weights
    2. Adaptive kernel size selection
    3. Feature-specific decomposition parameters
    """
    
    def __init__(self, input_dim, init_kernel_size=25, max_kernel_size=50):
        super(LearnableSeriesDecomposition, self).__init__()
        logger.info("Initializing LearnableSeriesDecomp")
        
        self.input_dim = input_dim
        self.max_kernel_size = max_kernel_size
        
        # Learnable trend extraction weights: one set of weights per input feature/channel
        self.feature_specific_trend_weights = nn.Parameter(torch.randn(input_dim, 1, max_kernel_size))
        
        # Adaptive kernel size predictor
        self.kernel_predictor = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 4), 1),
            nn.Sigmoid()
        )
        
        self.init_kernel_size = init_kernel_size
        if self.init_kernel_size > 0 and self.init_kernel_size <= self.max_kernel_size:
            nn.init.constant_(self.feature_specific_trend_weights[:, :, :self.init_kernel_size], 1.0 / self.init_kernel_size)
        
    def forward(self, x):
        B, L, D = x.shape
        assert D == self.input_dim, f"Input feature dimension mismatch. Expected {self.input_dim}, got {D}"

        x_global = x.mean(dim=1)
        kernel_logits = self.kernel_predictor(x_global)
        predicted_k_float = (kernel_logits * (self.max_kernel_size - 5) + 5)

        k_float = predicted_k_float.squeeze(-1)
        kernel_sizes = torch.round(k_float).int()
        kernel_sizes = torch.clamp(kernel_sizes, 3, min(self.max_kernel_size, L // 2))
        kernel_sizes = kernel_sizes - (kernel_sizes % 2 == 0).int()
        kernel_sizes = torch.clamp(kernel_sizes, min=3)

        trend = torch.zeros_like(x)
        
        unique_kernels = torch.unique(kernel_sizes)

        for k_val in unique_kernels:
            k = k_val.item()
            indices = torch.where(kernel_sizes == k)[0]
            batch_subset = x[indices]
            
            padding = k // 2
            subset_permuted = batch_subset.transpose(1, 2)
            subset_padded = F.pad(subset_permuted, (padding, padding), mode='replicate')
            
            current_weights_unnormalized = self.feature_specific_trend_weights[:, :, :k]
            normalized_weights = F.softmax(current_weights_unnormalized, dim=2)
            
            trend_conv = F.conv1d(
                subset_padded,
                normalized_weights,
                groups=self.input_dim,
                padding=0
            )
            
            trend[indices] = trend_conv.transpose(1, 2)

        seasonal = x - trend
        return seasonal, trend

class DecompositionRegistry:
    """
    A registry for all available series decomposition components.
    """
    _registry = {
        "series_decomp": SeriesDecomposition,
        "stable_decomp": StableSeriesDecomposition,
        "learnable_decomp": LearnableSeriesDecomposition,
        "wavelet_decomp": WaveletHierarchicalDecomposition,
    }

    @classmethod
    def register(cls, name, component_class):
        """
        Register a new decomposition component.
        """
        if name in cls._registry:
            logger.warning(f"Component '{name}' is already registered and will be overwritten.")
        cls._registry[name] = component_class
        logger.info(f"Registered decomposition component: {name}")

    @classmethod
    def get(cls, name):
        """
        Get a decomposition component by name.
        """
        component = cls._registry.get(name)
        if component is None:
            logger.error(f"Decomposition component '{name}' not found.")
            raise ValueError(f"Decomposition component '{name}' not found.")
        return component

    @classmethod
    def list_components(cls):
        """
        List all registered decomposition components.
        """
        return list(cls._registry.keys())

def get_decomposition_component(name, **kwargs):
    """
    Factory function to get an instance of a decomposition component.
    Automatically filters parameters based on component requirements.
    """
    component_class = DecompositionRegistry.get(name)
    
    # Filter parameters based on component's __init__ signature
    signature = inspect.signature(component_class.__init__)
    valid_params = set(signature.parameters.keys()) - {'self'}
    
    # Filter kwargs to only include valid parameters for this component
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # Log filtered parameters for debugging
    if len(filtered_kwargs) != len(kwargs):
        removed_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
        logger.debug(f"Component '{name}' filtered out parameters: {removed_params}")
        logger.debug(f"Component '{name}' using parameters: {list(filtered_kwargs.keys())}")
    
    return component_class(**filtered_kwargs)

class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(BaseDecomposition):
    """
    Standard series decomposition block using a moving average.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        """
        Decomposes the input time series into seasonal and trend components.
        """
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean

class StableSeriesDecomposition(BaseDecomposition):
    """
    Series decomposition with a stability fix to ensure the kernel size is always odd.
    """
    def __init__(self, kernel_size):
        super(StableSeriesDecomposition, self).__init__()
        # Fix: Ensure odd kernel size for stability
        stable_kernel_size = kernel_size + (1 - kernel_size % 2)
        self.moving_avg = MovingAverage(stable_kernel_size, stride=1)

    def forward(self, x):
        """
        Decomposes the input time series into seasonal and trend components.
        """
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean

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

# Migrated Functions  


# Registry function for decomposition components
def get_decomposition_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get decomposition component by name"""
    # This will be implemented based on the migrated components
    pass

def register_decomposition_components(registry):
    """Register all decomposition components with the registry"""
    # This will be implemented to register all migrated components
    pass
