"""Component registration for the unified registry.

This module registers all available components with the unified registry.
Import this module to populate the registry with all components.
"""

from .registry import component_registry, ComponentFamily
from ..fusion.hierarchical_fusion import HierarchicalFusion
from ..loss.quantile_loss import PinballLoss
from ..loss.standard_losses import StandardLossWrapper
from ..loss.advanced_losses import (
    MAPELoss, SMAPELoss, MASELoss, PSLoss, FocalLoss
)
from ..loss.adaptive_bayesian_losses import (
    AdaptiveAutoformerLoss, FrequencyAwareLoss, BayesianLoss,
    BayesianQuantileLoss, QuantileLoss, UncertaintyCalibrationLoss
)
from ..backbone.backbones import ChronosBackbone, T5Backbone, BERTBackbone, SimpleTransformerBackbone
from ..backbone.simple_backbones import VariationalLSTMBackbone
from ..backbone.crossformer_backbone import CrossformerBackboneWrapper
from ..feedforward.feedforwards import StandardFFN, GatedFFN, MoEFFN, ConvFFN
from ..output.linear_output import LinearOutput, LinearOutputConfig
from ..output.outputs import ForecastingHead, RegressionHead, OutputConfig
from ..normalization.registry import (
    LayerNormWrapper, RMSNormWrapper, TSNormalizerWrapper, NormalizationProcessorWrapper
)
from ..output_heads.level_output_head import LevelOutputHead
from ..encoder.crossformer_encoder import CrossformerEncoder
from ..encoder.variational_lstm_encoder import VariationalLSTMEncoder
from ..encoder.enhanced_encoder import EnhancedEncoder
from ..decoder.enhanced_decoder import EnhancedDecoder
from ..attention.registry import AttentionRegistry
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from ..attention.enhanced_autocorrelation import EnhancedAutoCorrelation
from ..attention.cross_resolution_attention import CrossResolutionAttention
from ..attention.fourier.fourier_attention import FourierAttention
from ..attention.timeseries.autocorrelation import AutoCorrelationAttention
from ..attention.wavelet.wavelet_attention import WaveletAttention
from ..attention.wavelet.adaptive_wavelet_attention import AdaptiveWaveletAttention
from ..attention.wavelet.multiscale_wavelet_attention import MultiScaleWaveletAttention
from ..attention.wavelet.wavelet_decomposition import WaveletDecomposition
from ..attention.temporal_conv.causal_convolution import CausalConvolution
from ..attention.temporal_conv.temporal_conv_net import TemporalConvNet
from ..attention.temporal_conv.convolutional_attention import ConvolutionalAttention
from ..attention.adaptive.meta_learning_adapter import MetaLearningAdapter
from ..attention.adaptive.adaptive_mixture import AdaptiveMixture
from ..attention.bayesian.bayesian_multi_head_attention import BayesianMultiHeadAttention
from ..attention.bayesian.variational_attention import VariationalAttention
from ..attention.bayesian.bayesian_cross_attention import BayesianCrossAttention
from ..attention.bayesian.bayesian_attention import BayesianAttention
from ..attention.multihead_graph_attention import MultiHeadGraphAttention, GraphTransformerLayer
from ..embedding.graph_positional_encoding import GraphAwarePositionalEncoding, HierarchicalGraphPositionalEncoding
from ..encoder.spatiotemporal_encoding import JointSpatioTemporalEncoding, AdaptiveSpatioTemporalEncoder
from ..graph.dynamic_graph import DynamicGraphConstructor, AdaptiveGraphStructure
import torch.nn as nn
from ...Autoformer_EncDec import moving_avg as MovingAverage

# Register Fusion Components
component_registry.register(
    name="hierarchical_fusion",
    component_class=HierarchicalFusion,
    component_type=ComponentFamily.FUSION,
    test_config={
        "d_model": 512,
        "n_levels": 3,
        "fusion_strategy": "weighted_concat"
    }
)

# Register Loss Components
# Standard losses
component_registry.register(
    name="quantile_loss",
    component_class=PinballLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

component_registry.register(
    name="pinball_loss",  # Alias for quantile
    component_class=PinballLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

# Advanced metric losses
component_registry.register(
    name="mape_loss",
    component_class=MAPELoss,
    component_type=ComponentFamily.LOSS,
    test_config={}
)

component_registry.register(
    name="smape_loss",
    component_class=SMAPELoss,
    component_type=ComponentFamily.LOSS,
    test_config={}
)

component_registry.register(
    name="mase_loss",
    component_class=MASELoss,
    component_type=ComponentFamily.LOSS,
    test_config={"seasonal_periods": 1}
)

component_registry.register(
    name="ps_loss",
    component_class=PSLoss,
    component_type=ComponentFamily.LOSS,
    test_config={}
)

component_registry.register(
    name="focal_loss",
    component_class=FocalLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"alpha": 1.0, "gamma": 2.0}
)

# Adaptive losses
component_registry.register(
    name="adaptive_autoformer_loss",
    component_class=AdaptiveAutoformerLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"base_loss": "mse"}
)

component_registry.register(
    name="frequency_aware_loss",
    component_class=FrequencyAwareLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"base_loss": "mse"}
)

component_registry.register(
    name="multi_quantile_loss",
    component_class=QuantileLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

# Bayesian losses
component_registry.register(
    name="bayesian_quantile_loss",
    component_class=BayesianQuantileLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

component_registry.register(
    name="uncertainty_calibration_loss",
    component_class=UncertaintyCalibrationLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"base_loss": "mse"}
)

# Additional standard loss aliases for smoke tests
component_registry.register(
    name="mse",
    component_class=StandardLossWrapper,
    component_type=ComponentFamily.LOSS,
    test_config={"loss_class": nn.MSELoss}
)

from ..loss.adaptive_bayesian_losses import BayesianMSELoss  # noqa: E402
component_registry.register(
    name="bayesian_mse_loss",
    component_class=BayesianMSELoss,
    component_type=ComponentFamily.LOSS,
    test_config={"config": {}}
)

# Register Backbone Components
component_registry.register(
    name="chronos_backbone",
    component_class=ChronosBackbone,
    component_type=ComponentFamily.BACKBONE,
    test_config={"d_model": 512, "n_layers": 6}
)

component_registry.register(
    name="t5_backbone",
    component_class=T5Backbone,
    component_type=ComponentFamily.BACKBONE,
    test_config={"d_model": 512, "n_layers": 6}
)

component_registry.register(
    name="bert_backbone",
    component_class=BERTBackbone,
    component_type=ComponentFamily.BACKBONE,
    test_config={"d_model": 512, "n_layers": 6}
)

component_registry.register(
    name="simple_transformer_backbone",
    component_class=SimpleTransformerBackbone,
    component_type=ComponentFamily.BACKBONE,
    test_config={"d_model": 512, "n_layers": 6, "n_heads": 8}
)

# Variational LSTM Backbone
component_registry.register(
    name="variational_lstm_backbone",
    component_class=VariationalLSTMBackbone,
    component_type=ComponentFamily.BACKBONE,
    test_config={"d_model": 256, "n_layers": 2, "hidden_size": 256, "dropout": 0.1, "input_dim": 256}
)

# Crossformer Backbone
component_registry.register(
    name="crossformer_backbone",
    component_class=CrossformerBackboneWrapper,
    component_type=ComponentFamily.BACKBONE,
    test_config={
        "d_model": 512,
        "dropout": 0.1,
        "enc_in": 7,
        "seq_len": 96,
        "n_heads": 8,
        "d_ff": 2048,
        "e_layers": 2,
        "factor": 3
    }
)

# Register Graph Attention Components
component_registry.register(
    name="multihead_graph_attention",
    component_class=MultiHeadGraphAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "d_model": 512,
        "n_heads": 8,
        "dropout": 0.1,
        "num_node_types": 3
    }
)

component_registry.register(
    name="graph_transformer_layer",
    component_class=GraphTransformerLayer,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1
    }
)

# Register Graph Positional Encoding Components
component_registry.register(
    name="graph_aware_positional",
    component_class=GraphAwarePositionalEncoding,
    component_type=ComponentFamily.EMBEDDING,
    test_config={
        "d_model": 512,
        "max_len": 1000,
        "num_nodes": 100
    }
)

component_registry.register(
    name="hierarchical_graph_positional",
    component_class=HierarchicalGraphPositionalEncoding,
    component_type=ComponentFamily.EMBEDDING,
    test_config={
        "d_model": 512,
        "max_len": 1000,
        "num_nodes": 100,
        "num_levels": 3
    }
)

# Register Spatiotemporal Encoding Components
component_registry.register(
    name="joint_spatiotemporal",
    component_class=JointSpatioTemporalEncoding,
    component_type=ComponentFamily.ENCODER,
    test_config={
        "d_model": 512,
        "num_nodes": 100,
        "temporal_kernel_sizes": [3, 5, 7],
        "spatial_hops": [1, 2, 3]
    }
)

component_registry.register(
    name="adaptive_spatiotemporal",
    component_class=AdaptiveSpatioTemporalEncoder,
    component_type=ComponentFamily.ENCODER,
    test_config={
        "d_model": 512,
        "num_nodes": 100,
        "num_layers": 4,
        "dropout": 0.1
    }
)

# Register Dynamic Graph Components
component_registry.register(
    name="dynamic_graph_constructor",
    component_class=DynamicGraphConstructor,
    component_type=ComponentFamily.PROCESSOR,
    test_config={
        "num_nodes": 100,
        "k_nearest": 10,
        "temperature": 1.0
    }
)

component_registry.register(
    name="adaptive_graph_structure",
    component_class=AdaptiveGraphStructure,
    component_type=ComponentFamily.PROCESSOR,
    test_config={
        "num_nodes": 100,
        "d_model": 512,
        "num_transitions": 5
    }
)

# Register Feedforward Components
component_registry.register(
    name="standard_ffn",
    component_class=StandardFFN,
    component_type=ComponentFamily.FEEDFORWARD,
    test_config={"d_model": 512, "d_ff": 2048}
)

component_registry.register(
    name="gated_ffn",
    component_class=GatedFFN,
    component_type=ComponentFamily.FEEDFORWARD,
    test_config={"d_model": 512, "d_ff": 2048}
)

# Register Attention Components
component_registry.register(
    name="enhanced_autocorrelation",
    component_class=EnhancedAutoCorrelation,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8}
)

def _factory_adaptive_autocorr(**kwargs):
    d_model = kwargs.get("d_model", 512)
    n_heads = kwargs.get("n_heads", 8)
    factor = kwargs.get("factor", 1)
    dropout = kwargs.get("dropout", 0.1)
    # Use AdaptiveAutoCorrelation that operates on [B, L, H, E] after projections inside the layer
    inner = AdaptiveAutoCorrelation(
        factor=factor,
        adaptive_k=True,
        multi_scale=True,
        scales=[1, 2, 4],
        attention_dropout=dropout,
        output_attention=False,
    )
    layer = AdaptiveAutoCorrelationLayer(inner, d_model, n_heads)

    # Wrap to provide a flexible forward signature compatible with smoke tests
    import torch.nn as _nn  # type: ignore
    class _Wrapper(_nn.Module):
        def __init__(self, inner_layer):
            super().__init__()
            self.inner = inner_layer
        def forward(self, queries, keys=None, values=None, attn_mask=None):
            if keys is None:
                keys = queries
            if values is None:
                values = queries
            return self.inner(queries, keys, values, attn_mask)
    return _Wrapper(layer)

component_registry.register(
    name="adaptive_autocorrelation_layer",
    component_class=_factory_adaptive_autocorr,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "factor": 1, "dropout": 0.1}
)

component_registry.register(
    name="cross_resolution_attention",
    component_class=CrossResolutionAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8}
)

component_registry.register(
    name="fourier_attention",
    component_class=FourierAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8}
)

component_registry.register(
    name="bayesian_attention",
    component_class=BayesianAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8}
)

# --- Additional Attention Aliases (snake_case) for unified registry discoverability ---
component_registry.register(
    name="autocorrelation_layer",
    component_class=AutoCorrelationAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "factor": 1}
)

component_registry.register(
    name="autocorrelation_attention",
    component_class=AutoCorrelationAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "factor": 1}
)

component_registry.register(
    name="autocorrelation",
    component_class=AutoCorrelationAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "factor": 1}
)

component_registry.register(
    name="wavelet_attention",
    component_class=WaveletAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "levels": 3}
)

component_registry.register(
    name="adaptive_wavelet_attention",
    component_class=AdaptiveWaveletAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "max_levels": 3}
)

component_registry.register(
    name="multi_scale_wavelet_attention",
    component_class=MultiScaleWaveletAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "scales": [1, 2, 4]}
)

component_registry.register(
    name="wavelet_decomposition",
    component_class=WaveletDecomposition,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "n_levels": 3}
)

component_registry.register(
    name="causal_convolution",
    component_class=CausalConvolution,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "kernel_sizes": [3,5], "dilation_rates": [1,2]}
)

component_registry.register(
    name="temporal_conv_net",
    component_class=TemporalConvNet,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "num_levels": 2, "kernel_size": 3}
)

component_registry.register(
    name="convolutional_attention",
    component_class=ConvolutionalAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "conv_kernel_size": 3, "pool_size": 2}
)

# Note: graph_attention_layer module lacks a .py extension in this repo; skip alias registration to avoid import errors.

component_registry.register(
    name="meta_learning_adapter",
    component_class=MetaLearningAdapter,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "adaptation_steps": 1, "meta_lr": 0.01, "inner_lr": 0.1}
)

component_registry.register(
    name="adaptive_mixture",
    component_class=AdaptiveMixture,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "mixture_components": 3}
)

component_registry.register(
    name="bayesian_multi_head_attention",
    component_class=BayesianMultiHeadAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "prior_std": 1.0, "n_samples": 2}
)

component_registry.register(
    name="variational_attention",
    component_class=VariationalAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "dropout": 0.1, "learn_variance": True}
)

component_registry.register(
    name="bayesian_cross_attention",
    component_class=BayesianCrossAttention,
    component_type=ComponentFamily.ATTENTION,
    test_config={"d_model": 512, "n_heads": 8, "prior_std": 1.0}
)

component_registry.register(
    name="moe_ffn",
    component_class=MoEFFN,
    component_type=ComponentFamily.FEEDFORWARD,
    test_config={"d_model": 512, "d_ff": 2048, "num_experts": 4}
)

component_registry.register(
    name="conv_ffn",
    component_class=ConvFFN,
    component_type=ComponentFamily.FEEDFORWARD,
    test_config={"d_model": 512, "d_ff": 2048, "kernel_size": 3}
)

# Register Output Components
component_registry.register(
    name="linear_output",
    component_class=LinearOutput,
    component_type=ComponentFamily.OUTPUT,
    test_config={"d_model": 512, "output_dim": 1}
)

component_registry.register(
    name="forecasting_head",
    component_class=ForecastingHead,
    component_type=ComponentFamily.OUTPUT,
    test_config={"d_model": 512, "output_dim": 1, "horizon": 1}
)

component_registry.register(
    name="regression_head",
    component_class=RegressionHead,
    component_type=ComponentFamily.OUTPUT,
    test_config={"d_model": 512, "output_dim": 1}
)

# Register Normalization Components
component_registry.register(
    name="layer_norm",
    component_class=LayerNormWrapper,
    component_type=ComponentFamily.NORMALIZATION,
    test_config={"d_model": 512}
)

component_registry.register(
    name="rms_norm",
    component_class=RMSNormWrapper,
    component_type=ComponentFamily.NORMALIZATION,
    test_config={"d_model": 512}
)

component_registry.register(
    name="ts_normalizer",
    component_class=TSNormalizerWrapper,
    component_type=ComponentFamily.NORMALIZATION,
    test_config={"mode": "standard"}
)

component_registry.register(
    name="normalization_processor",
    component_class=NormalizationProcessorWrapper,
    component_type=ComponentFamily.NORMALIZATION,
    test_config={"normalization_type": "standard", "feature_wise": True}
)

# Register Encoder Components
component_registry.register(
    name="crossformer_encoder",
    component_class=CrossformerEncoder,
    component_type=ComponentFamily.ENCODER,
    test_config={
        "enc_in": 7,
        "seq_len": 96,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "e_layers": 2,
        "dropout": 0.1,
        "factor": 3
    }
)

component_registry.register(
    name="variational_lstm_encoder",
    component_class=VariationalLSTMEncoder,
    component_type=ComponentFamily.ENCODER,
    test_config={
        "input_dim": 7,
        "hidden_size": 256,
        "n_layers": 2,
        "dropout": 0.1,
        "d_model": 256
    }
)

# Enhanced modular encoder (Autoformer-like)
component_registry.register(
    name="enhanced_encoder",
    component_class=EnhancedEncoder,
    component_type=ComponentFamily.ENCODER,
    test_config={
        "num_encoder_layers": 1,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "activation": "gelu",
        # attention_comp and decomp_comp are provided by tests via overrides
    }
)

# Enhanced modular decoder (Autoformer-like)
def _factory_enhanced_decoder(**kwargs):
    required = [
        "d_layers",
        "d_model",
        "c_out",
        "n_heads",
        "d_ff",
        "dropout",
        "activation",
        "self_attention_comp",
        "decomp_comp",
    ]
    missing = [k for k in required if k not in kwargs]
    if missing:
        raise ValueError(
            f"enhanced_decoder requires the following parameters: {missing}. "
            "Please provide attention and decomposition components explicitly."
        )

    cross_attn = kwargs.get("cross_attention_comp", kwargs["self_attention_comp"])

    return EnhancedDecoder(
        d_layers=kwargs["d_layers"],
        d_model=kwargs["d_model"],
        c_out=kwargs["c_out"],
        n_heads=kwargs["n_heads"],
        d_ff=kwargs["d_ff"],
        dropout=kwargs["dropout"],
        activation=kwargs["activation"],
        self_attention_comp=kwargs["self_attention_comp"],
        cross_attention_comp=cross_attn,
        decomp_comp=kwargs["decomp_comp"],
        norm_layer=kwargs.get("norm_layer"),
        projection=kwargs.get("projection"),
    )

component_registry.register(
    name="enhanced_decoder",
    component_class=_factory_enhanced_decoder,
    component_type=ComponentFamily.DECODER,
    test_config={
        "d_layers": 1,
        "d_model": 512,
        "c_out": 1,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "activation": "gelu",
        # self_attention_comp, decomp_comp (and optionally cross_attention_comp) must be provided by tests
    }
)

# Register Output Head Components
component_registry.register(
    name="level",
    component_class=LevelOutputHead,
    component_type=ComponentFamily.OUTPUT_HEAD,
    test_config={"d_model": 512, "c_out": 1}
)

# Register minimal Decomposition components required by smoke tests
component_registry.register(
    name="moving_avg",
    component_class=MovingAverage,
    component_type=ComponentFamily.DECOMPOSITION,
    test_config={"kernel_size": 3, "stride": 1}
)