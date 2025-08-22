"""Initial canonical component registrations for unified registry.

This performs a *shadow registration* so existing code paths continue to work
while new factory adoption proceeds.
"""
from .registry import unified_registry, ComponentFamily
from configs.schemas import ComponentType

# Import canonical modular implementations
# (Prefer modular versions over config-defined duplicates)
from layers.modular.attention.enhanced_autocorrelation import (
    EnhancedAutoCorrelation,
    AdaptiveAutoCorrelationLayer as NewAdaptiveAutoCorrelationLayer,
    HierarchicalAutoCorrelation,
)
from layers.modular.attention.cross_resolution_attention import CrossResolutionAttention
from layers.modular.attention.fourier_attention import FourierAttention, FourierBlock, FourierCrossAttention
from layers.modular.attention.wavelet_attention import (
    WaveletAttention,
    WaveletDecomposition,
    AdaptiveWaveletAttention,
    MultiScaleWaveletAttention,
)
from layers.modular.attention.bayesian_attention import (
    BayesianAttention,
    BayesianMultiHeadAttention,
    VariationalAttention,
    BayesianCrossAttention,
)
from layers.modular.attention.adaptive_components import MetaLearningAdapter, AdaptiveMixture
from layers.modular.attention.temporal_conv_attention import (
    CausalConvolution,
    TemporalConvNet,
    ConvolutionalAttention,
)
from layers.modular.attention.graph_attention import GraphAttentionLayer, MultiGraphAttention
from layers.modular.encoder.standard_encoder import StandardEncoder
from layers.modular.encoder.enhanced_encoder import EnhancedEncoder
from layers.modular.encoder.hierarchical_encoder import HierarchicalEncoder
from layers.modular.encoder.graph_encoder import GraphTimeSeriesEncoder, HybridGraphEncoder, AdaptiveGraphEncoder
from layers.modular.encoder.stable_encoder import StableEncoder
from layers.modular.decoder.standard_decoder import StandardDecoder
from layers.modular.decoder.enhanced_decoder import EnhancedDecoder
from layers.modular.decoder.stable_decoder import StableDecoder
from layers.modular.decomposition.series_decomposition import SeriesDecomposition as MovingAvgDecomposition
from layers.modular.decomposition.learnable_decomposition import LearnableSeriesDecomposition
from layers.modular.decomposition.stable_decomposition import StableSeriesDecomposition
from layers.modular.decomposition.wavelet_decomposition import WaveletHierarchicalDecomposition
from layers.LearnableWaveletDecomposition import LearnableWaveletDecomposition
from layers.modular.sampling.deterministic_sampling import DeterministicSampling
from layers.modular.sampling.bayesian_sampling import BayesianSampling
from layers.modular.sampling.monte_carlo_sampling import MonteCarloSampling
from layers.modular.sampling.dropout_sampling import DropoutSampling
from layers.modular.output_heads.standard_output_head import StandardOutputHead
from layers.modular.output_heads.quantile_output_head import QuantileOutputHead
from layers.modular.fusion.hierarchical_fusion import HierarchicalFusion
from layers.modular.backbone import ChronosBackboneWrapper
from layers.modular.embedding import TemporalEmbeddingWrapper
from layers.modular.processor import TimeDomainProcessorWrapper
from layers.modular.feedforward import PositionwiseFeedForwardWrapper
from layers.modular.output import LinearOutputWrapper

# Loss implementations in current codebase:
# - StandardLossWrapper wraps torch losses (MSE, MAE)
# - QuantileLoss and BayesianQuantileLoss in adaptive_bayesian_losses
# There is no explicit MSELoss/MAELoss class, so we create wrappers here.
import torch.nn as nn
from layers.modular.losses.standard_losses import StandardLossWrapper
from layers.modular.losses.adaptive_bayesian_losses import (
    BayesianQuantileLoss,
    QuantileLoss,
    AdaptiveAutoformerLoss,
    FrequencyAwareLoss,
    UncertaintyCalibrationLoss,
)
from layers.modular.losses.advanced_losses import (
    MAPELoss,
    SMAPELoss,
    MASELoss,
    PSLoss,
    FocalLoss,
)

class MSELoss(StandardLossWrapper):  # type: ignore
    """Wrapper providing a named MSE loss component for registry."""
    def __init__(self):
        super().__init__(nn.MSELoss)

class MAELoss(StandardLossWrapper):  # type: ignore
    """Wrapper providing a named MAE loss component for registry."""
    def __init__(self):
        super().__init__(nn.L1Loss)

# Bayesian MSE implemented as a simple extension placeholder (can be replaced later)
class BayesianMSELoss(MSELoss):  # type: ignore
    """Placeholder Bayesian MSE; extend with uncertainty terms in future."""
    pass

class HuberLoss(StandardLossWrapper):  # type: ignore
    """Wrapper providing a named Huber loss component for registry."""
    def __init__(self):
        super().__init__(nn.HuberLoss)

"""ATTENTION family registrations (comprehensive set mirroring legacy registry)."""
# Core / enhanced autocorrelation variants
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.AUTOCORRELATION.value,
    EnhancedAutoCorrelation,
    component_type=ComponentType.AUTOCORRELATION,
    aliases=['enhanced_autocorrelation', 'autocorrelation_layer']
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.ADAPTIVE_AUTOCORRELATION.value,
    NewAdaptiveAutoCorrelationLayer,
    component_type=ComponentType.ADAPTIVE_AUTOCORRELATION,
    aliases=['adaptive_autocorrelation_layer','adaptive_autocorrelation']
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    'hierarchical_autocorrelation',
    HierarchicalAutoCorrelation,
    component_type=None,
    metadata={'description': 'Hierarchical multi-resolution autocorrelation'}
)

# Cross-resolution
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.CROSS_RESOLUTION.value,
    CrossResolutionAttention,
    component_type=ComponentType.CROSS_RESOLUTION,
)

# Fourier variants
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.FOURIER_ATTENTION.value,
    FourierAttention,
    component_type=ComponentType.FOURIER_ATTENTION,
    aliases=['fourier']
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.FOURIER_BLOCK.value,
    FourierBlock,
    component_type=ComponentType.FOURIER_BLOCK,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.FOURIER_CROSS_ATTENTION.value,
    FourierCrossAttention,
    component_type=ComponentType.FOURIER_CROSS_ATTENTION,
)

# Wavelet / multi-scale
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.WAVELET_ATTENTION.value,
    WaveletAttention,
    component_type=ComponentType.WAVELET_ATTENTION,
    aliases=['wavelet']
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.WAVELET_DECOMPOSITION.value,
    WaveletDecomposition,
    component_type=ComponentType.WAVELET_DECOMPOSITION,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.ADAPTIVE_WAVELET_ATTENTION.value,
    AdaptiveWaveletAttention,
    component_type=ComponentType.ADAPTIVE_WAVELET_ATTENTION,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.MULTI_SCALE_WAVELET_ATTENTION.value,
    MultiScaleWaveletAttention,
    component_type=ComponentType.MULTI_SCALE_WAVELET_ATTENTION,
)

# Bayesian / probabilistic attn variants
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.BAYESIAN_ATTENTION.value,
    BayesianAttention,
    component_type=ComponentType.BAYESIAN_ATTENTION,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.BAYESIAN_MULTI_HEAD_ATTENTION.value,
    BayesianMultiHeadAttention,
    component_type=ComponentType.BAYESIAN_MULTI_HEAD_ATTENTION,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.VARIATIONAL_ATTENTION.value,
    VariationalAttention,
    component_type=ComponentType.VARIATIONAL_ATTENTION,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.BAYESIAN_CROSS_ATTENTION.value,
    BayesianCrossAttention,
    component_type=ComponentType.BAYESIAN_CROSS_ATTENTION,
)

# Meta / adaptive adapters
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.META_LEARNING_ADAPTER.value,
    MetaLearningAdapter,
    component_type=ComponentType.META_LEARNING_ADAPTER,
    aliases=['meta_learning_adapter','meta_adapter']
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.ADAPTIVE_MIXTURE_ATTN.value,
    AdaptiveMixture,
    component_type=ComponentType.ADAPTIVE_MIXTURE_ATTN,
    aliases=['adaptive_mixture']
)

# Temporal convolution hybrids
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.CAUSAL_CONVOLUTION.value,
    CausalConvolution,
    component_type=ComponentType.CAUSAL_CONVOLUTION,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.TEMPORAL_CONV_NET.value,
    TemporalConvNet,
    component_type=ComponentType.TEMPORAL_CONV_NET,
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    ComponentType.CONVOLUTIONAL_ATTENTION.value,
    ConvolutionalAttention,
    component_type=ComponentType.CONVOLUTIONAL_ATTENTION,
)

# Graph attention
unified_registry.register(
    ComponentFamily.ATTENTION,
    'graph_attention_layer',
    GraphAttentionLayer,
    component_type=None,
    metadata={'description': 'Single graph attention layer'}
)
unified_registry.register(
    ComponentFamily.ATTENTION,
    'multi_graph_attention',
    MultiGraphAttention,
    component_type=None,
    metadata={'description': 'Multi-graph attention wrapper'}
)

# DECOMPOSITION
unified_registry.register(
    ComponentFamily.DECOMPOSITION,
    ComponentType.MOVING_AVG.value,
    MovingAvgDecomposition,
    component_type=ComponentType.MOVING_AVG,
    aliases=['series_decomp']
)
unified_registry.register(
    ComponentFamily.DECOMPOSITION,
    ComponentType.LEARNABLE_DECOMP.value,
    LearnableSeriesDecomposition,
    component_type=ComponentType.LEARNABLE_DECOMP,
)
unified_registry.register(
    ComponentFamily.DECOMPOSITION,
    ComponentType.STABLE_DECOMP.value,
    StableSeriesDecomposition,
    component_type=ComponentType.STABLE_DECOMP,
    aliases=['stable_decomp']
)
unified_registry.register(
    ComponentFamily.DECOMPOSITION,
    'wavelet_decomp',
    WaveletHierarchicalDecomposition,
    component_type=ComponentType.WAVELET_DECOMP,
    aliases=[ComponentType.WAVELET_DECOMP.value]
)
unified_registry.register(
    ComponentFamily.DECOMPOSITION,
    'learnable_wavelet_decomp',
    LearnableWaveletDecomposition,
    component_type=ComponentType.ADVANCED_WAVELET,
    aliases=[ComponentType.ADVANCED_WAVELET.value]
)

"""ENCODER family registrations (standard + advanced graph/stable variants)."""
unified_registry.register(
    ComponentFamily.ENCODER,
    ComponentType.STANDARD_ENCODER.value,
    StandardEncoder,
    component_type=ComponentType.STANDARD_ENCODER,
)
unified_registry.register(
    ComponentFamily.ENCODER,
    ComponentType.ENHANCED_ENCODER.value,
    EnhancedEncoder,
    component_type=ComponentType.ENHANCED_ENCODER,
    aliases=['enhanced_encoder']
)
unified_registry.register(
    ComponentFamily.ENCODER,
    ComponentType.HIERARCHICAL_ENCODER.value,
    HierarchicalEncoder,
    component_type=ComponentType.HIERARCHICAL_ENCODER,
    aliases=['hier_encoder']
)
unified_registry.register(
    ComponentFamily.ENCODER,
    'graph_encoder',
    GraphTimeSeriesEncoder,
    component_type=None,
    metadata={'description': 'Graph time series encoder'}
)
unified_registry.register(
    ComponentFamily.ENCODER,
    'hybrid_graph_encoder',
    HybridGraphEncoder,
    component_type=None,
    metadata={'description': 'Hybrid graph encoder combining GAT + temporal'}
)
unified_registry.register(
    ComponentFamily.ENCODER,
    'adaptive_graph_encoder',
    AdaptiveGraphEncoder,
    component_type=None,
    metadata={'description': 'Adaptive graph encoder selecting dynamic topology'}
)
unified_registry.register(
    ComponentFamily.ENCODER,
    'stable_encoder',
    StableEncoder,
    component_type=None,
    metadata={'description': 'Stability-enhanced encoder variant'}
)

"""DECODER family registrations (standard + stable variants)."""
unified_registry.register(
    ComponentFamily.DECODER,
    ComponentType.STANDARD_DECODER.value,
    StandardDecoder,
    component_type=ComponentType.STANDARD_DECODER,
)
unified_registry.register(
    ComponentFamily.DECODER,
    ComponentType.ENHANCED_DECODER.value,
    EnhancedDecoder,
    component_type=ComponentType.ENHANCED_DECODER,
    aliases=['enhanced_decoder']
)
unified_registry.register(
    ComponentFamily.DECODER,
    'stable_decoder',
    StableDecoder,
    component_type=None,
    metadata={'description': 'Stability-enhanced decoder variant'}
)

"""SAMPLING family registrations (extended)."""
unified_registry.register(
    ComponentFamily.SAMPLING,
    ComponentType.DETERMINISTIC.value,
    DeterministicSampling,
    component_type=ComponentType.DETERMINISTIC,
)
unified_registry.register(
    ComponentFamily.SAMPLING,
    ComponentType.BAYESIAN.value,
    BayesianSampling,
    component_type=ComponentType.BAYESIAN,
)
unified_registry.register(
    ComponentFamily.SAMPLING,
    ComponentType.MONTE_CARLO.value,
    MonteCarloSampling,
    component_type=ComponentType.MONTE_CARLO,
    aliases=['monte_carlo_sampling']
)
unified_registry.register(
    ComponentFamily.SAMPLING,
    'dropout',
    DropoutSampling,
    component_type=None,
    metadata={'description': 'MC Dropout sampling strategy'}
)

# OUTPUT HEAD
unified_registry.register(ComponentFamily.OUTPUT_HEAD, ComponentType.STANDARD_HEAD.value, StandardOutputHead,
                          component_type=ComponentType.STANDARD_HEAD)
unified_registry.register(ComponentFamily.OUTPUT_HEAD, ComponentType.QUANTILE.value, QuantileOutputHead,
                          component_type=ComponentType.QUANTILE)
unified_registry.register(ComponentFamily.FUSION, 'hierarchical_fusion', HierarchicalFusion,
                          component_type=None, metadata={'description': 'Hierarchical multi-resolution fusion'})

"""LOSS family registrations (standard + advanced)."""
unified_registry.register(
    ComponentFamily.LOSS, ComponentType.MSE.value, MSELoss, component_type=ComponentType.MSE
)
unified_registry.register(
    ComponentFamily.LOSS, ComponentType.MAE.value, MAELoss, component_type=ComponentType.MAE
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.QUANTILE_LOSS.value,
    QuantileLoss,
    component_type=ComponentType.QUANTILE_LOSS,
    aliases=['pinball_loss','pinball','quantile','multi_quantile']
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.BAYESIAN_MSE.value,
    BayesianMSELoss,
    component_type=ComponentType.BAYESIAN_MSE,
    aliases=['bayesian']
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.BAYESIAN_QUANTILE.value,
    BayesianQuantileLoss,
    component_type=ComponentType.BAYESIAN_QUANTILE,
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.ADAPTIVE_AUTOFORMER_LOSS.value,
    AdaptiveAutoformerLoss,
    component_type=ComponentType.ADAPTIVE_AUTOFORMER_LOSS,
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.FREQUENCY_AWARE_LOSS.value,
    FrequencyAwareLoss,
    component_type=ComponentType.FREQUENCY_AWARE_LOSS,
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.UNCERTAINTY_CALIBRATION_LOSS.value,
    UncertaintyCalibrationLoss,
    component_type=ComponentType.UNCERTAINTY_CALIBRATION_LOSS,
    aliases=['uncertainty_calibration']
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.MAPE_LOSS.value,
    MAPELoss,
    component_type=ComponentType.MAPE_LOSS,
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.SMAPE_LOSS.value,
    SMAPELoss,
    component_type=ComponentType.SMAPE_LOSS,
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.MASE_LOSS.value,
    MASELoss,
    component_type=ComponentType.MASE_LOSS,
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.PS_LOSS.value,
    PSLoss,
    component_type=ComponentType.PS_LOSS,
)
unified_registry.register(
    ComponentFamily.LOSS,
    ComponentType.FOCAL_LOSS.value,
    FocalLoss,
    component_type=ComponentType.FOCAL_LOSS,
    aliases=['focal']
)
unified_registry.register(
    ComponentFamily.LOSS,
    'huber',
    HuberLoss,
    component_type=None,
)

"""BACKBONE / EMBEDDING / PROCESSOR / FEEDFORWARD / OUTPUT generic family registrations."""
unified_registry.register(
    ComponentFamily.BACKBONE,
    'chronos_backbone',
    ChronosBackboneWrapper,
    component_type=None,
    aliases=['chronos']
)
unified_registry.register(
    ComponentFamily.EMBEDDING,
    'temporal_embedding',
    TemporalEmbeddingWrapper,
    component_type=None,
    aliases=['temporal']
)
unified_registry.register(
    ComponentFamily.PROCESSOR,
    'time_domain_processor',
    TimeDomainProcessorWrapper,
    component_type=None,
    aliases=['time_domain']
)
unified_registry.register(
    ComponentFamily.FEEDFORWARD,
    'positionwise_feedforward',
    PositionwiseFeedForwardWrapper,
    component_type=None,
    aliases=['ff']
)
unified_registry.register(
    ComponentFamily.OUTPUT,
    'linear_output',
    LinearOutputWrapper,
    component_type=None,
    aliases=['linear']
)
