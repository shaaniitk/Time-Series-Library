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
import torch.nn as nn

# Register Fusion Components
component_registry.register(
    name="hierarchical_fusion",
    component_class=HierarchicalFusion,
    component_type=ComponentFamily.PROCESSOR,
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
    component_type=ComponentFamily.PROCESSOR,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

component_registry.register(
    name="pinball_loss",  # Alias for quantile
    component_class=PinballLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

# Advanced metric losses
component_registry.register(
    name="mape_loss",
    component_class=MAPELoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={}
)

component_registry.register(
    name="smape_loss",
    component_class=SMAPELoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={}
)

component_registry.register(
    name="mase_loss",
    component_class=MASELoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"seasonal_periods": 1}
)

component_registry.register(
    name="ps_loss",
    component_class=PSLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={}
)

component_registry.register(
    name="focal_loss",
    component_class=FocalLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"alpha": 1.0, "gamma": 2.0}
)

# Adaptive losses
component_registry.register(
    name="adaptive_autoformer_loss",
    component_class=AdaptiveAutoformerLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"base_loss": "mse"}
)

component_registry.register(
    name="frequency_aware_loss",
    component_class=FrequencyAwareLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"base_loss": "mse"}
)

component_registry.register(
    name="multi_quantile_loss",
    component_class=QuantileLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

# Bayesian losses
component_registry.register(
    name="bayesian_quantile_loss",
    component_class=BayesianQuantileLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

component_registry.register(
    name="uncertainty_calibration_loss",
    component_class=UncertaintyCalibrationLoss,
    component_type=ComponentFamily.PROCESSOR,
    test_config={"base_loss": "mse"}
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

# Register Output Head Components
component_registry.register(
    name="level",
    component_class=LevelOutputHead,
    component_type=ComponentFamily.OUTPUT_HEAD,
    test_config={"d_model": 512, "c_out": 1}
)