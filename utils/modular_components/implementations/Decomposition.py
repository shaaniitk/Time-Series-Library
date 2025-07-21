"""


Unified Decomposition Components

This module consolidates all series decomposition implementations:
- SeriesDecomposition
- StableSeriesDecomposition
- LearnableSeriesDecomposition
- WaveletHierarchicalDecomposition

All classes implement the BaseDecomposition interface and are registered in DECOMPOSITION_REGISTRY.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod

from ..base_interfaces import BaseComponent


from ..config_schemas import ComponentConfig
from utils.logger import logger
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse

logger = logging.getLogger(__name__)

# ==============================================================================
# BASE DECOMPOSITION INTERFACE
# ==============================================================================

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
            tuple: (seasonal, trend)
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
        self.feature_specific_trend_weights = nn.Parameter(torch.randn(input_dim, 1, max_kernel_size))
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
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean

class StableSeriesDecomposition(BaseDecomposition):
    """
    Series decomposition with a stability fix to ensure the kernel size is always odd.
    """
    def __init__(self, kernel_size):
        super(StableSeriesDecomposition, self).__init__()
        stable_kernel_size = kernel_size + (1 - kernel_size % 2)
        self.moving_avg = MovingAverage(stable_kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean

class WaveletHierarchicalDecomposition(BaseDecomposition):
    """
    Hierarchical decomposition using existing DWT infrastructure.
    Integrates with existing DWT_Decomposition.py to create multi-resolution time series representations.
    """
    def __init__(self, seq_len, d_model, wavelet_type='db4', levels=3, use_learnable_weights=True):
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
        # NOTE: Implementation of forward method should be completed as per DWT logic
        # For now, placeholder
    def forward(self, x):
        # TODO: Implement full wavelet decomposition logic
        raise NotImplementedError("WaveletHierarchicalDecomposition forward not implemented.")

# Unified registry for decomposition components
DECOMPOSITION_REGISTRY = {
    "series_decomp": SeriesDecomposition,
    "stable_decomp": StableSeriesDecomposition,
    "learnable_decomp": LearnableSeriesDecomposition,
    "wavelet_decomp": WaveletHierarchicalDecomposition,
}

import inspect

def get_decomposition_method(name, **kwargs):
    """
    Factory function to get an instance of a decomposition component.
    Automatically filters parameters based on component requirements.
    """
    component_class = DECOMPOSITION_REGISTRY.get(name)
    if component_class is None:
        logger.error(f"Decomposition component '{name}' not found.")
        raise ValueError(f"Decomposition component '{name}' not found.")
    signature = inspect.signature(component_class.__init__)
    valid_params = set(signature.parameters.keys()) - {'self'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    if len(filtered_kwargs) != len(kwargs):
        removed_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
        logger.debug(f"Component '{name}' filtered out parameters: {removed_params}")
        logger.debug(f"Component '{name}' using parameters: {list(filtered_kwargs.keys())}")
    return component_class(**filtered_kwargs)
