"""Structured configuration schemas for the Modular Autoformer (clean, deduplicated).

This module defines a single authoritative set of Pydantic v2 models and
helper factory functions. It keeps backward compatibility with legacy flat
arguments via aliases and the `from_legacy` constructor.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator


# ---------------------------------------------------------------------------
# Component Enumeration
# ---------------------------------------------------------------------------


class ComponentType(str, Enum):
    """Enumeration of registry keys used across components and configs."""

    # Core / baseline attention
    MULTI_HEAD = "multi_head"
    AUTOCORRELATION = "autocorrelation"
    AUTOCORRELATION_LAYER = "autocorrelation_layer"
    ADAPTIVE_AUTOCORRELATION = "adaptive_autocorrelation_layer"
    ADAPTIVE_AUTOCORRELATION_LAYER = "adaptive_autocorrelation_layer"  # alias

    # Fourier attention
    FOURIER_ATTENTION = "fourier_attention"
    FOURIER_BLOCK = "fourier_block"
    FOURIER_CROSS_ATTENTION = "fourier_cross_attention"

    # Wavelet / multi-scale
    WAVELET_ATTENTION = "wavelet_attention"
    WAVELET_DECOMPOSITION = "wavelet_decomposition"
    ADAPTIVE_WAVELET_ATTENTION = "adaptive_wavelet_attention"
    MULTI_SCALE_WAVELET_ATTENTION = "multi_scale_wavelet_attention"

    # Cross resolution & hybrid
    CROSS_RESOLUTION = "cross_resolution_attention"

    # Sparse families
    SPARSE = "sparse"
    LOG_SPARSE = "log_sparse"
    PROB_SPARSE = "prob_sparse"

    # Temporal convolution / hybrids
    CAUSAL_CONVOLUTION = "causal_convolution"
    TEMPORAL_CONV_NET = "temporal_conv_net"
    CONVOLUTIONAL_ATTENTION = "convolutional_attention"
    TEMPORAL_CONV_ENCODER = "temporal_conv_encoder"

    # Decomposition
    MOVING_AVG = "moving_avg"
    SERIES_DECOMP = "series_decomp"
    STABLE_DECOMP = "stable_decomp"
    LEARNABLE_DECOMP = "learnable_decomp"
    WAVELET_DECOMP = "wavelet_hierarchical_decomp"
    ADVANCED_WAVELET = "advanced_wavelet_decomp"

    # Encoder / decoder variants
    STANDARD_ENCODER = "standard_encoder"
    ENHANCED_ENCODER = "enhanced_encoder"
    HIERARCHICAL_ENCODER = "hierarchical_encoder"
    STANDARD_DECODER = "standard_decoder"
    ENHANCED_DECODER = "enhanced_decoder"
    HIERARCHICAL_DECODER = "hierarchical_decoder"
    META_LEARNING_ADAPTER = "meta_learning_adapter"

    # Sampling strategies
    DETERMINISTIC = "deterministic"
    BAYESIAN = "bayesian"
    MONTE_CARLO = "monte_carlo"
    ADAPTIVE_MIXTURE = "adaptive_mixture"
    # Attention-side alias for adaptive mixture (used by registry)
    ADAPTIVE_MIXTURE_ATTN = "adaptive_mixture_attn"

    # Output heads
    STANDARD_HEAD = "standard"
    QUANTILE = "quantile"
    BAYESIAN_HEAD = "bayesian_head"

    # Loss functions
    MSE = "mse"
    MAE = "mae"
    QUANTILE_LOSS = "quantile_loss"
    BAYESIAN_MSE = "bayesian_mse_loss"
    BAYESIAN_QUANTILE = "bayesian_quantile_loss"
    FOCAL_LOSS = "focal_loss"
    ADAPTIVE_AUTOFORMER_LOSS = "adaptive_autoformer_loss"
    ADAPTIVE_LOSS_WEIGHTING = "adaptive_loss_weighting"
    MAPE_LOSS = "mape_loss"
    SMAPE_LOSS = "smape_loss"
    MASE_LOSS = "mase_loss"
    PS_LOSS = "ps_loss"
    FREQUENCY_AWARE_LOSS = "frequency_aware_loss"
    MULTI_QUANTILE_LOSS = "multi_quantile_loss"
    UNCERTAINTY_CALIBRATION_LOSS = "uncertainty_calibration_loss"

    # Bayesian building blocks
    BAYESIAN_ATTENTION = "bayesian_attention"
    BAYESIAN_MULTI_HEAD_ATTENTION = "bayesian_multi_head_attention"
    VARIATIONAL_ATTENTION = "variational_attention"
    BAYESIAN_CROSS_ATTENTION = "bayesian_cross_attention"

    # Backbone integration
    CHRONOS = "chronos"
    CHRONOS_X = "chronos_x"

    # Normalization components
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

    # Embedding components
    POSITIONAL_EMBEDDING = "positional_embedding"
    TOKEN_EMBEDDING = "token_embedding"
    FIXED_EMBEDDING = "fixed_embedding"
    TEMPORAL_EMBEDDING = "temporal_embedding"
    TIME_FEATURE_EMBEDDING = "time_feature_embedding"
    DATA_EMBEDDING = "data_embedding"
    DATA_EMBEDDING_INVERTED = "data_embedding_inverted"
    DATA_EMBEDDING_WO_POS = "data_embedding_wo_pos"
    PATCH_EMBEDDING = "patch_embedding"

class BaseModelConfig(BaseModel):
    """Universal base configuration schema for *legacy* backbone style models.

    This is intentionally minimal – it validates the most common architectural
    hyper‑parameters while allowing extra fields so legacy config objects do not
    raise validation errors (Crossformer, FEDformer, etc.).
    """

    model_type: Literal['crossformer', 'fedformer', 'etsformer', 'timesnet']
    d_model: PositiveInt = 512
    n_heads: PositiveInt = 8
    e_layers: NonNegativeInt = 2
    d_ff: PositiveInt = 2048
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "swish"] = "gelu"

    # Common optional fields
    d_layers: Optional[NonNegativeInt] = None
    moving_avg: Optional[int] = None
    factor: Optional[int] = None
    enc_in: Optional[int] = None
    dec_in: Optional[int] = None
    c_out: Optional[int] = None
    seq_len: Optional[int] = None
    label_len: Optional[int] = None
    pred_len: Optional[int] = None
    task_name: Optional[str] = None
    embed: Optional[str] = None
    freq: Optional[str] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _validate(self):  # type: ignore[override]
        if self.d_ff is not None and self.d_model is not None and self.d_ff < 4 * self.d_model:
            raise ValueError("d_ff should be at least 4*d_model for effective expansion")
        if self.n_heads is not None and self.d_model is not None and self.n_heads > self.d_model:
            raise ValueError("n_heads cannot exceed d_model dimensionality")
        if not (0.0 <= self.dropout <= 0.9):
            raise ValueError("dropout must be between 0.0 and 0.9")
        return self


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


class AttentionConfig(BaseModel):
    type: ComponentType
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    num_heads: Optional[int] = None  # compatibility mirror
    head_dim: Optional[int] = None
    dropout: float = 0.1
    factor: int = 1
    output_attention: bool = False
    seq_len: Optional[int] = None
    local_window: Optional[int] = None
    global_window: Optional[int] = None

    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after")
    def _normalize_heads(self):  # type: ignore[override]
        if self.n_heads is None and self.num_heads is not None:
            self.n_heads = self.num_heads
        elif self.num_heads is None and self.n_heads is not None:
            self.num_heads = self.n_heads
        if self.n_heads is not None and self.num_heads is not None and self.n_heads != self.num_heads:
            raise ValueError("n_heads and num_heads must be equal if both are provided")
        if self.d_model is not None and self.n_heads is not None and self.head_dim is None:
            if self.d_model % self.n_heads == 0:
                self.head_dim = self.d_model // self.n_heads
        return self


class DecompositionConfig(BaseModel):
    type: ComponentType
    kernel_size: int = 25
    input_dim: Optional[int] = None
    wavelet_type: Optional[str] = "db4"
    levels: Optional[int] = 3

    model_config = {"extra": "allow"}


class HierarchicalConfig(BaseModel):
    n_levels: int = 3
    level_configs: Optional[List[Dict[str, Any]]] = None


class EncoderConfig(BaseModel):
    type: ComponentType
    num_encoder_layers: int = Field(..., alias="e_layers")
    d_ff: int
    dropout: float = 0.1
    activation: str = "gelu"
    d_model: Optional[int] = None
    n_heads: Optional[int] = None

    attention_comp: Optional[AttentionConfig] = None
    decomp_comp: Optional[DecompositionConfig] = None
    hierarchical: Optional[HierarchicalConfig] = None

    model_config = {"extra": "allow", "populate_by_name": True}

    # Legacy compatibility: allow accessing encoder layers via `e_layers`
    @property
    def e_layers(self) -> int:  # pragma: no cover - simple alias
        return self.num_encoder_layers

    @e_layers.setter
    def e_layers(self, value: int) -> None:  # pragma: no cover - simple alias
        self.num_encoder_layers = value


class DecoderConfig(BaseModel):
    type: ComponentType
    num_decoder_layers: int = Field(..., alias="d_layers")
    d_ff: int
    dropout: float = 0.1
    activation: str = "gelu"
    c_out: int
    d_model: Optional[int] = None
    n_heads: Optional[int] = None

    attention_comp: Optional[AttentionConfig] = None
    decomp_comp: Optional[DecompositionConfig] = None

    model_config = {"extra": "allow", "populate_by_name": True}

    # Legacy compatibility: allow accessing decoder layers via `d_layers`
    @property
    def d_layers(self) -> int:  # pragma: no cover - simple alias
        return self.num_decoder_layers

    @d_layers.setter
    def d_layers(self, value: int) -> None:  # pragma: no cover - simple alias
        self.num_decoder_layers = value


class SamplingConfig(BaseModel):
    type: ComponentType
    n_samples: int = 50
    quantile_levels: Optional[List[float]] = None
    dropout_rate: Optional[float] = 0.1
    temperature: Optional[float] = 1.0
    kl_weight: Optional[float] = 1e-5

    model_config = {"extra": "allow"}


class OutputHeadConfig(BaseModel):
    type: ComponentType
    d_model: int
    c_out: int
    num_quantiles: Optional[int] = None

    model_config = {"extra": "allow"}


class LossConfig(BaseModel):
    type: ComponentType
    quantiles: Optional[List[float]] = None
    prior_scale: Optional[float] = 1.0
    kl_weight: Optional[float] = 1.0

    model_config = {"extra": "allow"}


class BayesianConfig(BaseModel):
    enabled: bool = False
    layers_to_convert: List[str] = Field(default_factory=lambda: ["projection"])
    prior_scale: float = 1.0
    posterior_scale_init: float = -3.0
    kl_weight: float = 1.0


class BackboneConfig(BaseModel):
    type: Optional[ComponentType] = None
    model_name: Optional[str] = None
    use_backbone: bool = False
    prediction_length: Optional[int] = None
    context_length: Optional[int] = None

class EmbeddingConfig(BaseModel):
    type: ComponentType
    c_in: int
    d_model: int
    max_len: int = 5000
    dropout: float = 0.1
    time_features: bool = True
    freq: str = "h"
    model_config = {"extra": "allow"}

class NormalizationConfig(BaseModel):
    type: ComponentType
    normalized_shape: int
    eps: float = 1e-5
    affine: bool = True
    model_config = {"extra": "allow"}

class ModularAutoformerConfig(BaseModel):
    """Authoritative structured configuration for the Modular Autoformer.

    Backward compatibility:
    - Accepts legacy encoder/decoder layer keys via aliases.
    - `to_namespace()` exposes the legacy flat attributes used by older
        training / model assembly code.
    - Additional optional fields (`embed`, `freq`, `dropout`) retained.
    """
    task_name: str = "long_term_forecast"
    seq_len: int
    pred_len: int
    label_len: int
    enc_in: int
    dec_in: int
    c_out: int
    c_out_evaluation: int
    d_model: int = 512
    attention: AttentionConfig
    decomposition: DecompositionConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    sampling: SamplingConfig
    output_head: OutputHeadConfig
    loss: LossConfig
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    embedding: Optional[EmbeddingConfig] = None
    normalization: Optional[NormalizationConfig] = None
    quantile_levels: Optional[List[float]] = None
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.1

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


def create_enhanced_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Canonical enhanced configuration (baseline + adaptive autocorrelation)."""
    seq_len = kwargs.get('seq_len', 96)
    pred_len = kwargs.get('pred_len', 24)
    label_len = kwargs.get('label_len', 48)
    d_model = kwargs.get('d_model', 512)
    n_heads = kwargs.get('n_heads', 8)
    e_layers = kwargs.get('e_layers', kwargs.get('num_encoder_layers', 2))
    d_layers = kwargs.get('d_layers', kwargs.get('num_decoder_layers', 1))
    attn_type = kwargs.get('attention_type', ComponentType.ADAPTIVE_AUTOCORRELATION)
    decomp_type = kwargs.get('decomposition_type', ComponentType.LEARNABLE_DECOMP)

    return ModularAutoformerConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=num_targets + num_covariates,
        dec_in=num_targets + num_covariates,
        c_out=num_targets,
        c_out_evaluation=num_targets,
        d_model=d_model,
        attention=AttentionConfig(
            type=ComponentType.ADAPTIVE_AUTOCORRELATION,
            dropout=0.1,
            factor=1,
            output_attention=False,
            d_model=d_model,
            num_heads=num_heads,
            seq_len=seq_len,
        ),
        decomposition=DecompositionConfig(
            type=decomp_type,
            kernel_size=25,
            input_dim=d_model,
            type=ComponentType.LEARNABLE_DECOMP,
            kernel_size=25
        ),
        encoder=EncoderConfig(
            type=ComponentType.ENHANCED_ENCODER,
            e_layers=e_layers,  # alias
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=2,
            e_layers=2,
            d_ff=2048,
            dropout=0.1,
            activation="gelu",
            d_model=d_model,
            n_heads=heads,
        ),
        decoder=DecoderConfig(
            type=ComponentType.ENHANCED_DECODER,
            d_layers=d_layers,  # alias
            d_model=d_model,
            n_heads=n_heads,
            d_ff=2048,
            dropout=0.1,
            activation="gelu",
            c_out=num_targets,
            d_model=d_model,
            n_heads=heads,
        ),
        sampling=SamplingConfig(type=ComponentType.DETERMINISTIC),
        output_head=OutputHeadConfig(
            type=ComponentType.STANDARD_HEAD,
            d_model=d_model,
            c_out=num_targets,
        ),
        loss=LossConfig(type=ComponentType.MSE),
    )


def create_bayesian_enhanced_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Enhanced config + Bayesian sampling and Bayesian MSE loss."""
    config = create_enhanced_config(num_targets, num_covariates, **kwargs)
    config.sampling.type = ComponentType.BAYESIAN
    config.sampling.n_samples = kwargs.get('n_samples', 50)
    config.loss.type = ComponentType.BAYESIAN_MSE
    config.bayesian.enabled = True
    config.bayesian.layers_to_convert = kwargs.get('bayesian_layers', ['projection'])
    return config


def create_quantile_bayesian_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Bayesian + quantile outputs (quantile head, Bayesian quantile loss)."""
    quantile_levels = kwargs.get('quantile_levels', [0.1, 0.5, 0.9])
    config = create_bayesian_enhanced_config(num_targets, num_covariates, **kwargs)
    config.quantile_levels = quantile_levels
    config.c_out = num_targets * len(quantile_levels)
    config.sampling.quantile_levels = quantile_levels
    config.output_head.type = ComponentType.QUANTILE
    config.output_head.num_quantiles = len(quantile_levels)
    config.output_head.c_out = num_targets
    config.loss.type = ComponentType.BAYESIAN_QUANTILE
    config.loss.quantiles = quantile_levels
    return config


def create_hierarchical_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Hierarchical multi-resolution configuration (wavelet + cross-resolution)."""
    n_levels = kwargs.get('n_levels', 3)
    config = create_enhanced_config(num_targets, num_covariates, **kwargs)
    config.attention.type = ComponentType.CROSS_RESOLUTION
    config.decomposition.type = ComponentType.WAVELET_DECOMP
    config.decomposition.levels = n_levels
    config.encoder.type = ComponentType.HIERARCHICAL_ENCODER
    return config


# ---------------------------------------------------------------------------

# Merged from layers/modular/core/config_schemas.py: Unique dataclass configurations
# Note: Overlapping classes (e.g., BackboneConfig, EmbeddingConfig, AttentionConfig, LossConfig, OutputConfig) 
# are not added as they exist as Pydantic models here. Adding base ComponentConfig, ProcessorConfig, 
# FeedForwardConfig, and the overarching ModularModelConfig.

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from abc import ABC

@dataclass
class ComponentConfig(ABC):
    """Base configuration class for all components"""
    component_name: str = ""
    d_model: int = 256
    dropout: float = 0.1
    device: str = "auto"  # auto, cpu, cuda
    dtype: str = "float32"  # float32, float16, bfloat16
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.component_name:
            self.component_name = self.__class__.__name__.replace("Config", "")

@dataclass
class ProcessorConfig(ComponentConfig):
    """Configuration for processing strategies"""
    processor_type: str = "seq2seq"  # seq2seq, encoder_only, hierarchical, autoregressive
    seq_len: int = 96
    pred_len: int = 24
    label_len: int = 48
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cross_scale_attention: bool = True
    use_decoder: bool = True
    decoder_strategy: str = "teacher_forcing"  # teacher_forcing, autoregressive
    pooling_method: str = "adaptive"  # adaptive, average, max, attention

@dataclass
class FeedForwardConfig(ComponentConfig):
    """Configuration for feed-forward networks"""
    ffn_type: str = "standard"  # standard, mixture_experts, adaptive, gated
    d_ff: int = 1024
    activation: str = "relu"  # relu, gelu, swish, mish
    use_bias: bool = True
    num_experts: int = 4
    expert_dropout: float = 0.1
    gate_type: str = "top_k"  # top_k, softmax, learned
    top_k: int = 2
    adaptive_method: str = "linear"  # linear, attention, meta
    adaptation_dim: int = 64

@dataclass
class ModularModelConfig:
    """Complete configuration for modular time series models"""
    model_name: str = "ModularHFAutoformer"
    model_version: str = "1.0"
    seq_len: int = 96
    pred_len: int = 24
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    d_model: int = 256
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    feedforward: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    validate_config: bool = True
    strict_compatibility: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and synchronization"""
        components = [self.backbone, self.embedding, self.attention, 
                      self.processor, self.feedforward, self.loss, self.output]
        for component in components:
            if hasattr(component, 'd_model'):
                component.d_model = self.d_model
        self.embedding.input_dim = self.enc_in
        self.embedding.output_dim = self.d_model
        self.output.output_dim = self.c_out
        self.processor.seq_len = self.seq_len
        self.processor.pred_len = self.pred_len
        if self.validate_config:
            self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration consistency"""
        assert self.embedding.output_dim == self.d_model, \
            f"Embedding output dim {self.embedding.output_dim} != d_model {self.d_model}"
        assert self.processor.seq_len == self.seq_len, \
            f"Processor seq_len {self.processor.seq_len} != global seq_len {self.seq_len}"
        if self.attention.head_dim is None:
            assert self.d_model % self.attention.num_heads == 0, \
                f"d_model {self.d_model} not divisible by num_heads {self.attention.num_heads}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModularModelConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
# Additional configuration factory helpers (extending coverage)
# ---------------------------------------------------------------------------


class ModularAutoformerConfig(BaseModel):
    task_name: str = "long_term_forecast"
    seq_len: int
    pred_len: int
    label_len: int
    enc_in: int
    dec_in: int
    c_out: int
    c_out_evaluation: int
    d_model: int = 512

    attention: AttentionConfig
    decomposition: DecompositionConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    sampling: SamplingConfig
    output_head: OutputHeadConfig
    loss: LossConfig

    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)

    quantile_levels: Optional[List[float]] = None
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.1

    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after")
    def _validate_consistency(self):  # type: ignore[override]
        if self.quantile_levels:
            expected_c_out = self.c_out_evaluation * len(self.quantile_levels)
            if self.c_out != expected_c_out:
                raise ValueError(
                    f"c_out ({self.c_out}) should equal c_out_evaluation ({self.c_out_evaluation}) * num_quantiles ({len(self.quantile_levels)})"
                )
        if self.encoder and self.encoder.d_model is not None and self.encoder.d_model != self.d_model:
            raise ValueError("Encoder d_model must match top-level d_model")
        if self.decoder and self.decoder.d_model is not None and self.decoder.d_model != self.d_model:
            raise ValueError("Decoder d_model must match top-level d_model")
        return self

    def to_namespace(self):
        from argparse import Namespace

        flat: Dict[str, Any] = {
            "task_name": self.task_name,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "enc_in": self.enc_in,
            "dec_in": self.dec_in,
            "c_out": self.c_out,
            "c_out_evaluation": self.c_out_evaluation,
            "d_model": self.d_model,
            "embed": self.embed,
            "freq": self.freq,
            "dropout": self.dropout,
            # Component types
            "attention_type": self.attention.type.value,
            "decomposition_type": self.decomposition.type.value,
            "encoder_type": self.encoder.type.value,
            "decoder_type": self.decoder.type.value,
            "sampling_type": self.sampling.type.value,
            "output_head_type": self.output_head.type.value,
            "loss_function_type": self.loss.type.value,
            # Component params
            "attention_params": self.attention.model_dump(exclude={"type"}),
            "decomposition_params": self.decomposition.model_dump(exclude={"type"}),
            "encoder_params": self.encoder.model_dump(exclude={"type"}),
            "decoder_params": self.decoder.model_dump(exclude={"type"}),
            "sampling_params": self.sampling.model_dump(exclude={"type"}),
            "output_head_params": self.output_head.model_dump(exclude={"type"}),
            "loss_params": self.loss.model_dump(exclude={"type"}),
            # Specials
            "quantile_levels": self.quantile_levels,
            "bayesian_layers": self.bayesian.layers_to_convert if self.bayesian.enabled else [],
            "use_backbone_component": self.backbone.use_backbone,
            "backbone_type": self.backbone.type.value if self.backbone.type else None,
        }
        return Namespace(**flat)

    @classmethod
    def from_legacy(cls, legacy_config):
        """Construct from a legacy Namespace or dict with best-effort mappings."""
        cfg = legacy_config if isinstance(legacy_config, dict) else vars(legacy_config)

        def _map_type(val: Optional[str], default: ComponentType, domain: str) -> ComponentType:
            if val is None:
                return default
            v = str(val).lower()
            if domain == "encoder":
                if v in {"enhanced", "enhanced_encoder"}:
                    return ComponentType.ENHANCED_ENCODER
                if v in {"standard", "standard_encoder"}:
                    return ComponentType.STANDARD_ENCODER
                if v in {"hierarchical", "hierarchical_encoder"}:
                    return ComponentType.HIERARCHICAL_ENCODER
            if domain == "decoder":
                if v in {"enhanced", "enhanced_decoder"}:
                    return ComponentType.ENHANCED_DECODER
                if v in {"standard", "standard_decoder"}:
                    return ComponentType.STANDARD_DECODER
                if v in {"hierarchical", "hierarchical_decoder"}:
                    return ComponentType.HIERARCHICAL_DECODER
            if domain == "attention":
                if v in {"adaptive_autocorrelation", "adaptive_autocorrelation_layer"}:
                    return ComponentType.ADAPTIVE_AUTOCORRELATION
                if v in {"autocorrelation", "autocorrelation_layer"}:
                    return ComponentType.AUTOCORRELATION_LAYER
            if domain == "decomposition":
                if v in {"learnable", "learnable_decomp"}:
                    return ComponentType.LEARNABLE_DECOMP
            if domain == "sampling":
                if v in {"deterministic"}:
                    return ComponentType.DETERMINISTIC
                if v in {"bayesian"}:
                    return ComponentType.BAYESIAN
            if domain == "head":
                if v in {"standard", "linear", "head"}:
                    return ComponentType.STANDARD_HEAD
                if v in {"quantile", "quantile_head"}:
                    return ComponentType.QUANTILE
            if domain == "loss":
                if v in {"mse"}:
                    return ComponentType.MSE
                if v in {"quantile", "quantile_loss"}:
                    return ComponentType.QUANTILE_LOSS
                if v in {"bayesian_mse", "bayesian_mse_loss"}:
                    return ComponentType.BAYESIAN_MSE
                if v in {"bayesian_quantile", "bayesian_quantile_loss"}:
                    return ComponentType.BAYESIAN_QUANTILE
            return ComponentType(v)

        def _clean_params(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not isinstance(d, dict):
                return {}
            cleaned = dict(d)
            cleaned.pop("type", None)
            return cleaned

        base_params = {
            "task_name": cfg.get("task_name", "long_term_forecast"),
            "seq_len": cfg["seq_len"],
            "pred_len": cfg["pred_len"],
            "label_len": cfg["label_len"],
            "enc_in": cfg["enc_in"],
            "dec_in": cfg["dec_in"],
            "c_out": cfg["c_out"],
            "c_out_evaluation": cfg.get("c_out_evaluation", cfg["c_out"]),
            "d_model": cfg.get("d_model", 512),
            "embed": cfg.get("embed", "timeF"),
            "freq": cfg.get("freq", "h"),
            "dropout": cfg.get("dropout", 0.1),
            "quantile_levels": cfg.get("quantile_levels"),
        }

        attention = AttentionConfig(
            type=_map_type(cfg.get("attention_type"), ComponentType.ADAPTIVE_AUTOCORRELATION, "attention"),
            **_clean_params(cfg.get("attention_params")),
        )
        decomposition = DecompositionConfig(
            type=_map_type(cfg.get("decomposition_type"), ComponentType.LEARNABLE_DECOMP, "decomposition"),
            **_clean_params(cfg.get("decomposition_params")),
        )
        encoder = EncoderConfig(
            type=_map_type(cfg.get("encoder_type"), ComponentType.ENHANCED_ENCODER, "encoder"),
            **_clean_params(cfg.get("encoder_params")),
        )
        decoder = DecoderConfig(
            type=_map_type(cfg.get("decoder_type"), ComponentType.ENHANCED_DECODER, "decoder"),
            **_clean_params(cfg.get("decoder_params")),
        )
        sampling = SamplingConfig(
            type=_map_type(cfg.get("sampling_type"), ComponentType.DETERMINISTIC, "sampling"),
            **_clean_params(cfg.get("sampling_params")),
        )
        output_head = OutputHeadConfig(
            type=_map_type(cfg.get("output_head_type"), ComponentType.STANDARD_HEAD, "head"),
            **_clean_params(cfg.get("output_head_params")),
        )
        loss = LossConfig(
            type=_map_type(cfg.get("loss_function_type"), ComponentType.MSE, "loss"),
            **_clean_params(cfg.get("loss_params")),
        )

        bayesian = BayesianConfig(
            enabled=bool(cfg.get("bayesian_layers")),
            layers_to_convert=cfg.get("bayesian_layers", []),
        )
        backbone = BackboneConfig(
            use_backbone=cfg.get("use_backbone_component", False),
            type=ComponentType(cfg.get("backbone_type")) if cfg.get("backbone_type") else None,
        )

        return cls(
            **base_params,
            attention=attention,
            decomposition=decomposition,
            encoder=encoder,
            decoder=decoder,
            sampling=sampling,
            output_head=output_head,
            loss=loss,
            bayesian=bayesian,
            backbone=backbone,
        )


# ---------------------------------------------------------------------------
# Helper factory functions
# ---------------------------------------------------------------------------


def create_enhanced_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    seq_len = kwargs.get("seq_len", 96)
    pred_len = kwargs.get("pred_len", 24)
    label_len = kwargs.get("label_len", 48)
    d_model = kwargs.get("d_model", 512)
    heads = kwargs.get("n_heads", kwargs.get("num_heads", None))

    return ModularAutoformerConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=num_targets + num_covariates,
        dec_in=num_targets + num_covariates,
        c_out=num_targets,
        c_out_evaluation=num_targets,
        d_model=d_model,
        attention=AttentionConfig(
            type=ComponentType.ADAPTIVE_AUTOCORRELATION,
            dropout=0.1,
            factor=1,
            output_attention=False,
            d_model=d_model,
            n_heads=heads,
            num_heads=heads,
            seq_len=seq_len,
        ),
    decomposition=DecompositionConfig(**{"type": ComponentType.LEARNABLE_DECOMP, "kernel_size": 25}),
        encoder=EncoderConfig(
            type=ComponentType.ENHANCED_ENCODER,
            num_encoder_layers=2,
            d_ff=2048,
            dropout=0.1,
            activation="gelu",
            d_model=d_model,
            n_heads=heads,
        ),
        decoder=DecoderConfig(
            type=ComponentType.ENHANCED_DECODER,
            num_decoder_layers=1,
            d_ff=2048,
            dropout=0.1,
            activation="gelu",
            c_out=num_targets,
            d_model=d_model,
            n_heads=heads,
        ),
        sampling=SamplingConfig(type=ComponentType.DETERMINISTIC),
        output_head=OutputHeadConfig(type=ComponentType.STANDARD_HEAD, d_model=d_model, c_out=num_targets),
        loss=LossConfig(type=ComponentType.MSE),
    )


def create_bayesian_enhanced_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    cfg = create_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.sampling.type = ComponentType.BAYESIAN
    cfg.sampling.n_samples = kwargs.get("n_samples", 50)
    cfg.loss.type = ComponentType.BAYESIAN_MSE
    cfg.bayesian.enabled = True
    cfg.bayesian.layers_to_convert = kwargs.get("bayesian_layers", ["projection"])
    return cfg


def create_quantile_bayesian_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    quantile_levels = kwargs.get("quantile_levels", [0.1, 0.5, 0.9])
    cfg = create_bayesian_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.quantile_levels = quantile_levels
    cfg.c_out = num_targets * len(quantile_levels)
    cfg.sampling.quantile_levels = quantile_levels
    cfg.output_head.type = ComponentType.QUANTILE
    cfg.output_head.num_quantiles = len(quantile_levels)
    cfg.output_head.c_out = num_targets
    cfg.loss.type = ComponentType.BAYESIAN_QUANTILE
    cfg.loss.quantiles = quantile_levels
    return cfg


def create_hierarchical_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    n_levels = kwargs.get("n_levels", 3)
    cfg = create_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.attention.type = ComponentType.CROSS_RESOLUTION
    cfg.decomposition.type = ComponentType.WAVELET_DECOMP
    cfg.attention.levels = n_levels  # stored via extra
    cfg.decomposition.levels = n_levels
    cfg.encoder.type = ComponentType.HIERARCHICAL_ENCODER
    cfg.encoder.hierarchical = HierarchicalConfig(n_levels=n_levels)
    cfg.decoder.type = ComponentType.ENHANCED_DECODER
    return cfg


def create_quantile_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    quantile_levels = kwargs.get("quantile_levels", [0.1, 0.5, 0.9])
    cfg = create_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.quantile_levels = quantile_levels
    cfg.c_out = num_targets * len(quantile_levels)
    cfg.output_head.type = ComponentType.QUANTILE
    cfg.output_head.num_quantiles = len(quantile_levels)
    cfg.output_head.c_out = num_targets
    cfg.sampling.type = ComponentType.DETERMINISTIC
    cfg.sampling.quantile_levels = quantile_levels
    cfg.loss.type = ComponentType.QUANTILE_LOSS
    cfg.loss.quantiles = quantile_levels
    return cfg


def create_adaptive_mixture_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    n_samples = kwargs.get("n_samples", 32)
    cfg = create_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.sampling.type = ComponentType.ADAPTIVE_MIXTURE
    cfg.sampling.n_samples = n_samples
    return cfg


def create_advanced_wavelet_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    levels = kwargs.get("levels", 3)
    cfg = create_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.decomposition.type = ComponentType.ADVANCED_WAVELET
    cfg.decomposition.levels = levels
    cfg.encoder.type = ComponentType.HIERARCHICAL_ENCODER
    cfg.encoder.hierarchical = HierarchicalConfig(n_levels=levels)
    return cfg


def create_temporal_conv_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    cfg = create_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.encoder.type = ComponentType.TEMPORAL_CONV_ENCODER
    return cfg


def create_meta_learning_adapter_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    cfg = create_enhanced_config(num_targets, num_covariates, **kwargs)
    cfg.encoder.type = ComponentType.META_LEARNING_ADAPTER
    return cfg
