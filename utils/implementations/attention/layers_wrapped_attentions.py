"""
Adapters for legacy/specialized attention models from layers.modular into the utils registry.

Each adapter wraps an existing attention implementation to the utils BaseAttention
interface and registers it into the global ComponentRegistry via a helper.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from ...modular_components.base_interfaces import BaseAttention
from ...modular_components.config_schemas import AttentionConfig
from ...modular_components.registry import register_component

# Import legacy implementations from layers folder
from layers.modular.attention.wavelet_attention import (
    WaveletAttention as LegacyWaveletAttention,
    AdaptiveWaveletAttention as LegacyAdaptiveWaveletAttention,
    MultiScaleWaveletAttention as LegacyMultiScaleWaveletAttention,
)
from layers.modular.attention.bayesian_attention import (
    BayesianAttention as LegacyBayesianAttention,
    VariationalAttention as LegacyVariationalAttention,
    BayesianCrossAttention as LegacyBayesianCrossAttention,
)
from layers.modular.attention.fourier_attention import (
    FourierCrossAttention as LegacyFourierCrossAttention,
)
from layers.modular.attention.temporal_conv_attention import (
    ConvolutionalAttention as LegacyConvolutionalAttention,
)
from layers.modular.attention.cross_resolution_attention import (
    CrossResolutionAttention as LegacyCrossResolutionAttention,
)


# ---------------------- Configs ----------------------

@dataclass
class WaveletAttentionConfig(AttentionConfig):
    levels: int = 3
    n_levels: Optional[int] = None
    wavelet_type: str = "learnable"


@dataclass
class AdaptiveWaveletAttentionConfig(AttentionConfig):
    max_levels: int = 5


@dataclass
class MultiScaleWaveletAttentionConfig(AttentionConfig):
    scales: List[int] = (1, 2, 4, 8)


@dataclass
class BayesianAttentionConfig(AttentionConfig):
    prior_std: float = 1.0
    temperature: float = 1.0
    output_attention: bool = False


@dataclass
class VariationalAttentionConfig(AttentionConfig):
    learn_variance: bool = True


@dataclass
class BayesianCrossAttentionConfig(AttentionConfig):
    prior_std: float = 1.0


@dataclass
class FourierCrossAttentionConfig(AttentionConfig):
    temperature: float = 1.0


@dataclass
class ConvolutionalAttentionConfig(AttentionConfig):
    conv_kernel_size: int = 3
    pool_size: int = 2
    activation: str = "gelu"


@dataclass
class CrossResolutionAttentionConfig(AttentionConfig):
    n_levels: int = 3


# ---------------------- Adapters ----------------------

class _LegacyAttentionAdapter(BaseAttention):
    """Generic adapter that wraps a legacy attention module with forward(q,k,v)."""

    def __init__(self, config: AttentionConfig, impl: nn.Module, attention_type: str):
        super().__init__(config)
        self.impl = impl
        self._d_model = config.d_model
        self._attention_type = attention_type

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Many legacy modules accept (q, k, v, mask) or (q, k, v)
        try:
            return self.impl(q, k, v, attn_mask)
        except TypeError:
            out = self.impl(q, k, v)
            return out if isinstance(out, tuple) else (out, None)

    def apply_attention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                        attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, attn = self.forward(queries, keys, values, attn_mask)
        return out, attn

    def get_output_dim(self) -> int:
        return self._d_model

    def get_attention_type(self) -> str:
        return self._attention_type


class WrappedWaveletAttention(_LegacyAttentionAdapter):
    def __init__(self, config: WaveletAttentionConfig):
        impl = LegacyWaveletAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            levels=config.levels,
            n_levels=config.n_levels,
            dropout=config.dropout,
        )
        super().__init__(config, impl, "layers_wavelet")


class WrappedAdaptiveWaveletAttention(_LegacyAttentionAdapter):
    def __init__(self, config: AdaptiveWaveletAttentionConfig):
        impl = LegacyAdaptiveWaveletAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            max_levels=config.max_levels,
        )
        super().__init__(config, impl, "layers_adaptive_wavelet")


class WrappedMultiScaleWaveletAttention(_LegacyAttentionAdapter):
    def __init__(self, config: MultiScaleWaveletAttentionConfig):
        impl = LegacyMultiScaleWaveletAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            scales=list(config.scales) if isinstance(config.scales, tuple) else config.scales,
            dropout=config.dropout,
        )
        super().__init__(config, impl, "layers_multiscale_wavelet")


class WrappedBayesianAttention(_LegacyAttentionAdapter):
    def __init__(self, config: BayesianAttentionConfig):
        impl = LegacyBayesianAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            dropout=config.dropout,
            prior_std=config.prior_std,
            temperature=config.temperature,
            output_attention=config.output_attention,
        )
        super().__init__(config, impl, "layers_bayesian")


class WrappedVariationalAttention(_LegacyAttentionAdapter):
    def __init__(self, config: VariationalAttentionConfig):
        impl = LegacyVariationalAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            dropout=config.dropout,
            learn_variance=config.learn_variance,
        )
        super().__init__(config, impl, "layers_variational")


class WrappedBayesianCrossAttention(_LegacyAttentionAdapter):
    def __init__(self, config: BayesianCrossAttentionConfig):
        impl = LegacyBayesianCrossAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            dropout=config.dropout,
            prior_std=config.prior_std,
        )
        super().__init__(config, impl, "layers_bayesian_cross")


class WrappedFourierCrossAttention(_LegacyAttentionAdapter):
    def __init__(self, config: FourierCrossAttentionConfig):
        impl = LegacyFourierCrossAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            dropout=config.dropout,
            temperature=config.temperature,
        )
        super().__init__(config, impl, "layers_fourier_cross")


class WrappedConvolutionalAttention(_LegacyAttentionAdapter):
    def __init__(self, config: ConvolutionalAttentionConfig):
        impl = LegacyConvolutionalAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            conv_kernel_size=config.conv_kernel_size,
            pool_size=config.pool_size,
            dropout=config.dropout,
            activation=config.activation,
        )
        super().__init__(config, impl, "layers_convolutional")


class WrappedCrossResolutionAttention(_LegacyAttentionAdapter):
    def __init__(self, config: CrossResolutionAttentionConfig):
        impl = LegacyCrossResolutionAttention(
            d_model=config.d_model,
            n_levels=config.n_levels,
            n_heads=config.num_heads,
        )
        super().__init__(config, impl, "layers_cross_resolution")


# ---------------------- Registration ----------------------

def register_layers_attentions():
    """Register wrapped legacy attention components into utils registry."""
    # Wavelet family
    register_component(
        'attention',
        'layers_wavelet_attention',
        WrappedWaveletAttention,
        metadata={
            'source': 'layers.modular.attention.wavelet_attention',
            'features': ['multi_resolution', 'wavelet_decomposition', 'fusion'],
            'sophistication_level': 'high'
        }
    )
    register_component(
        'attention',
        'layers_adaptive_wavelet_attention',
        WrappedAdaptiveWaveletAttention,
        metadata={
            'source': 'layers.modular.attention.wavelet_attention',
            'features': ['adaptive_levels', 'multi_resolution'],
            'sophistication_level': 'high'
        }
    )
    register_component(
        'attention',
        'layers_multiscale_wavelet_attention',
        WrappedMultiScaleWaveletAttention,
        metadata={
            'source': 'layers.modular.attention.wavelet_attention',
            'features': ['multiscale', 'wavelet'],
            'sophistication_level': 'high'
        }
    )

    # Bayesian family
    register_component(
        'attention',
        'layers_bayesian_attention',
        WrappedBayesianAttention,
        metadata={
            'source': 'layers.modular.attention.bayesian_attention',
            'features': ['bayesian', 'uncertainty', 'prior_std'],
            'sophistication_level': 'high'
        }
    )
    register_component(
        'attention',
        'layers_variational_attention',
        WrappedVariationalAttention,
        metadata={
            'source': 'layers.modular.attention.bayesian_attention',
            'features': ['variational', 'learn_variance'],
            'sophistication_level': 'high'
        }
    )
    register_component(
        'attention',
        'layers_bayesian_cross_attention',
        WrappedBayesianCrossAttention,
        metadata={
            'source': 'layers.modular.attention.bayesian_attention',
            'features': ['cross_attention', 'bayesian'],
            'sophistication_level': 'high'
        }
    )

    # Fourier cross-attention
    register_component(
        'attention',
        'layers_fourier_cross_attention',
        WrappedFourierCrossAttention,
        metadata={
            'source': 'layers.modular.attention.fourier_attention',
            'features': ['fourier', 'cross_attention'],
            'sophistication_level': 'high'
        }
    )

    # Convolutional attention
    register_component(
        'attention',
        'layers_convolutional_attention',
        WrappedConvolutionalAttention,
        metadata={
            'source': 'layers.modular.attention.temporal_conv_attention',
            'features': ['temporal_conv', 'multi_kernel', 'pooling'],
            'sophistication_level': 'medium'
        }
    )

    # Cross-resolution attention
    register_component(
        'attention',
        'layers_cross_resolution_attention',
        WrappedCrossResolutionAttention,
        metadata={
            'source': 'layers.modular.attention.cross_resolution_attention',
            'features': ['multi_resolution', 'cross_scale'],
            'sophistication_level': 'high'
        }
    )
