"""
UNIFIED DECOMPOSITION COMPONENTS
All decomposition mechanisms in one place - no duplicates, clean modular structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# BASE DECOMPOSITION
# =============================================================================

class BaseDecomposition(BaseComponent):
    """Base class for all decomposition components"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.config = config
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (trend, seasonal) components"""
        raise NotImplementedError


# =============================================================================
# MOVING AVERAGE DECOMPOSITION
# =============================================================================

class MovingAverageDecomposition(BaseDecomposition):
    """Moving average based decomposition"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.kernel_size = config.custom_params.get('kernel_size', 25)
        
        # Ensure odd kernel size
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            
        self.padding = self.kernel_size // 2
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, seq_len, features]
        Returns: (trend, seasonal)
        """
        batch_size, seq_len, features = x.shape
        
        # Moving average for trend extraction
        x_permuted = x.permute(0, 2, 1)  # [batch, features, seq_len]
        
        # Apply moving average
        trend = F.avg_pool1d(
            F.pad(x_permuted, (self.padding, self.padding), mode='replicate'),
            kernel_size=self.kernel_size,
            stride=1
        )
        
        trend = trend.permute(0, 2, 1)  # [batch, seq_len, features]
        seasonal = x - trend
        
        return trend, seasonal


class SeriesDecomposition(BaseDecomposition):
    """Series decomposition used in Autoformer"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.kernel_size = config.custom_params.get('kernel_size', 25)
        
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            
        self.moving_avg = MovingAverageDecomposition(config)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns decomposed trend and seasonal components
        """
        return self.moving_avg(x)


# =============================================================================
# LEARNABLE DECOMPOSITION
# =============================================================================

class LearnableDecomposition(BaseDecomposition):
    """Learnable decomposition with neural networks"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.hidden_dim = config.custom_params.get('hidden_dim', 64)
        
        # Trend extraction network
        self.trend_net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.d_model)
        )
        
        # Seasonal extraction network
        self.seasonal_net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.d_model)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, seq_len, features]
        """
        trend = self.trend_net(x)
        seasonal = self.seasonal_net(x)
        
        # Ensure decomposition constraint: trend + seasonal ≈ x
        residual = x - (trend + seasonal)
        trend = trend + 0.5 * residual
        seasonal = seasonal + 0.5 * residual
        
        return trend, seasonal


# =============================================================================
# FOURIER DECOMPOSITION
# =============================================================================

class FourierDecomposition(BaseDecomposition):
    """Fourier-based decomposition for periodic components"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.top_k = config.custom_params.get('top_k', 10)
        self.low_freq_threshold = config.custom_params.get('low_freq_threshold', 0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, seq_len, features]
        """
        batch_size, seq_len, features = x.shape
        
        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=1)
        freq_dim = x_fft.shape[1]
        
        # Create frequency mask for trend (low frequencies)
        freq_indices = torch.arange(freq_dim, device=x.device)
        normalized_freq = freq_indices.float() / freq_dim
        
        trend_mask = (normalized_freq <= self.low_freq_threshold).float()
        seasonal_mask = 1.0 - trend_mask
        
        # Apply masks
        trend_fft = x_fft * trend_mask.unsqueeze(0).unsqueeze(-1)
        seasonal_fft = x_fft * seasonal_mask.unsqueeze(0).unsqueeze(-1)
        
        # Transform back to time domain
        trend = torch.fft.irfft(trend_fft, n=seq_len, dim=1)
        seasonal = torch.fft.irfft(seasonal_fft, n=seq_len, dim=1)
        
        return trend, seasonal


# =============================================================================
# WAVELET DECOMPOSITION
# =============================================================================

class WaveletDecomposition(BaseDecomposition):
    """Wavelet-based decomposition for multi-scale analysis"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.levels = config.custom_params.get('levels', 3)
        self.d_model = config.d_model
        
        # Learnable wavelet filters
        self.wavelet_filters = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_model, 4))  # 4-tap filters
            for _ in range(self.levels)
        ])
        
    def _wavelet_transform(self, x: torch.Tensor, level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single level wavelet transform"""
        batch_size, seq_len, features = x.shape
        filter_weights = self.wavelet_filters[level]
        
        # Apply convolution
        x_conv = F.conv1d(
            x.transpose(1, 2), 
            filter_weights.unsqueeze(1), 
            padding=2, 
            groups=features
        )
        
        # Split into approximation and detail
        approx = x_conv[:, :, ::2]  # Downsample by 2
        detail = x_conv[:, :, 1::2]
        
        return approx.transpose(1, 2), detail.transpose(1, 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, seq_len, features]
        """
        current = x
        details = []
        
        # Multi-level decomposition
        for level in range(self.levels):
            if current.shape[1] > 4:  # Minimum size
                approx, detail = self._wavelet_transform(current, level)
                details.append(detail)
                current = approx
        
        # Trend is the final approximation
        trend = current
        
        # Seasonal is sum of all detail coefficients
        if details:
            # Interpolate details back to original length
            seasonal = torch.zeros_like(x)
            for detail in details:
                detail_interp = F.interpolate(
                    detail.transpose(1, 2), 
                    size=x.shape[1], 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
                seasonal = seasonal + detail_interp
        else:
            seasonal = torch.zeros_like(x)
            
        # Interpolate trend back to original length
        if trend.shape[1] != x.shape[1]:
            trend = F.interpolate(
                trend.transpose(1, 2),
                size=x.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        return trend, seasonal


# =============================================================================
# ADAPTIVE DECOMPOSITION
# =============================================================================

class AdaptiveDecomposition(BaseDecomposition):
    """Adaptive decomposition that selects best method"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.d_model = config.d_model
        
        # Available decomposition methods
        self.decomposition_methods = nn.ModuleList([
            MovingAverageDecomposition(config),
            LearnableDecomposition(config),
            FourierDecomposition(config)
        ])
        
        # Method selection network
        self.method_selector = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.decomposition_methods)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, seq_len, features]
        """
        batch_size = x.shape[0]
        
        # Select decomposition method based on input characteristics
        x_summary = x.mean(dim=1)  # [batch_size, features]
        method_weights = self.method_selector(x_summary)  # [batch_size, num_methods]
        
        # Apply each method and combine
        trend_total = torch.zeros_like(x)
        seasonal_total = torch.zeros_like(x)
        
        for i, decomp_method in enumerate(self.decomposition_methods):
            trend_i, seasonal_i = decomp_method(x)
            
            # Weight by method selection
            weight = method_weights[:, i:i+1, None]  # [batch_size, 1, 1]
            trend_total += weight * trend_i
            seasonal_total += weight * seasonal_i
        
        return trend_total, seasonal_total


# =============================================================================
# RESIDUAL DECOMPOSITION
# =============================================================================

class ResidualDecomposition(BaseDecomposition):
    """Decomposition with explicit residual component"""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.base_decomp = SeriesDecomposition(config)
        self.d_model = config.d_model
        
        # Residual processing
        self.residual_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (trend, seasonal, residual)
        """
        # Base decomposition
        trend, seasonal = self.base_decomp(x)
        
        # Compute residual
        residual = x - trend - seasonal
        residual = self.residual_net(residual)
        
        # Adjust trend and seasonal to account for processed residual
        adjusted_trend = trend + 0.5 * residual
        adjusted_seasonal = seasonal + 0.5 * residual
        
        return adjusted_trend, adjusted_seasonal, residual


# =============================================================================
# COMPONENT REGISTRY
# =============================================================================

class DecompositionRegistry:
    """Registry for all decomposition components"""
    
    _components = {
        'moving_average': MovingAverageDecomposition,
        'series': SeriesDecomposition,
        'learnable': LearnableDecomposition,
        'fourier': FourierDecomposition,
        'wavelet': WaveletDecomposition,
        'adaptive': AdaptiveDecomposition,
        'residual': ResidualDecomposition,
    }
    
    @classmethod
    def register(cls, name: str, component_class):
        """Register a new decomposition component"""
        cls._components[name] = component_class
        logger.info(f"Registered decomposition component: {name}")
    
    @classmethod
    def create(cls, name: str, config: ComponentConfig):
        """Create a decomposition component by name"""
        if name not in cls._components:
            raise ValueError(f"Unknown decomposition component: {name}")
        
        component_class = cls._components[name]
        return component_class(config)
    
    @classmethod
    def list_components(cls):
        """List all available decomposition components"""
        return list(cls._components.keys())


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_decomposition_component(name: str, config: ComponentConfig = None, **kwargs):
    """Factory function to create decomposition components"""
    if config is None:
        # Create config from kwargs for backward compatibility
        config = ComponentConfig(
            component_name=name,
            d_model=kwargs.get('d_model', 512),
            custom_params=kwargs
        )
    
    return DecompositionRegistry.create(name, config)


def register_decomposition_components(registry):
    """Register all decomposition components with the main registry"""
    for name, component_class in DecompositionRegistry._components.items():
        registry.register('decomposition', name, component_class)
    
    logger.info(f"Registered {len(DecompositionRegistry._components)} decomposition components")
