"""
Structured Configuration Schemas for Modular Autoformer

This module implements the GCLI recommendations for replacing the flat Namespace
configuration with structured, validated configuration objects using Pydantic.
"""

from pydantic import (
    BaseModel,
    Field,
    validator,
    PositiveInt,
    NonNegativeInt,
    confloat,
    model_validator,
)
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum


###############################################################################
# Component Enumeration
#
# NOTE:
#   Previously most component identifiers were (incorrectly) declared as class
#   attributes inside BaseModelConfig instead of the dedicated ComponentType
#   Enum. Only MULTI_HEAD existed in the Enum which broke every reference like
#   ComponentType.AUTOCORRELATION across the codebase. This refactor consolidates
#   ALL component identifiers into ComponentType to restore consistency and
#   make IDE / static tooling happy.
###############################################################################


class ComponentType(str, Enum):
    """Enumeration of all available component and logical configuration types.

    The string values are intentionally stable – they are used as registry keys
    throughout the modular framework (component_registry, tests, HF migration
    helpers). Adding a new component only requires extending this Enum and the
    concrete factory/registry implementation; existing configs remain valid.
    """

    # Core / baseline attention components
    MULTI_HEAD = "multi_head"
    AUTOCORRELATION = "autocorrelation"
    AUTOCORRELATION_LAYER = "autocorrelation_layer"  # explicit layer variant
    ADAPTIVE_AUTOCORRELATION = "adaptive_autocorrelation_layer"
    ADAPTIVE_AUTOCORRELATION_LAYER = "adaptive_autocorrelation_layer"  # alias

    # Extended / Fourier based attention
    FOURIER_ATTENTION = "fourier_attention"
    FOURIER_BLOCK = "fourier_block"
    FOURIER_CROSS_ATTENTION = "fourier_cross_attention"

    # Wavelet / multi-scale attention
    WAVELET_ATTENTION = "wavelet_attention"
    WAVELET_DECOMPOSITION = "wavelet_decomposition"
    ADAPTIVE_WAVELET_ATTENTION = "adaptive_wavelet_attention"
    MULTI_SCALE_WAVELET_ATTENTION = "multi_scale_wavelet_attention"

    # Cross resolution & hybrid attention
    CROSS_RESOLUTION = "cross_resolution_attention"

    # Sparse attention families
    SPARSE = "sparse"
    LOG_SPARSE = "log_sparse"
    PROB_SPARSE = "prob_sparse"

    # Temporal convolution / hybrid encoder blocks
    CAUSAL_CONVOLUTION = "causal_convolution"
    TEMPORAL_CONV_NET = "temporal_conv_net"
    CONVOLUTIONAL_ATTENTION = "convolutional_attention"
    TEMPORAL_CONV_ENCODER = "temporal_conv_encoder"

    # Decomposition components
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

    # Bayesian / probabilistic building blocks
    BAYESIAN_ATTENTION = "bayesian_attention"
    BAYESIAN_MULTI_HEAD_ATTENTION = "bayesian_multi_head_attention"
    VARIATIONAL_ATTENTION = "variational_attention"
    BAYESIAN_CROSS_ATTENTION = "bayesian_cross_attention"

    # Meta / adaptive higher-level constructs
    META_LEARNING_ADAPTER_ATTN = "meta_learning_adapter"
    ADAPTIVE_MIXTURE_ATTN = "adaptive_mixture"

    # Backbone integration identifiers
    CHRONOS = "chronos"
    CHRONOS_X = "chronos_x"


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
    dropout: confloat(ge=0.0, le=0.9) = 0.1
    activation: Literal['gelu', 'relu', 'swish'] = 'gelu'

    # Frequently present in legacy configs (optional here for broad compatibility)
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

    # Allow unknown / forward‑compatible parameters
    model_config = {"extra": "allow"}

    @model_validator(mode='after')
    def validate_architecture(self):  # type: ignore[override]
        # Expansion ratio heuristic (skip if some fields missing in partial configs)
        if self.d_ff is not None and self.d_model is not None and self.d_ff < 4 * self.d_model:
            raise ValueError('d_ff should be at least 4*d_model for effective expansion')
        if self.n_heads is not None and self.d_model is not None and self.n_heads > self.d_model:
            raise ValueError('n_heads cannot exceed d_model dimensionality')
        return self


class AttentionConfig(BaseModel):
    """Configuration for attention mechanisms"""
    type: ComponentType
    d_model: int
    n_heads: int
    dropout: float = 0.1
    factor: int = 1
    output_attention: bool = False
    
    # Hierarchical-specific
    n_levels: Optional[int] = None
    
    # Sparse attention specific
    local_window: Optional[int] = None
    global_window: Optional[int] = None


class DecompositionConfig(BaseModel):
    """Configuration for series decomposition"""
    type: ComponentType
    kernel_size: int = 25
    
    # Learnable decomposition
    input_dim: Optional[int] = None
    
    # Wavelet decomposition
    wavelet_type: Optional[str] = "db4"
    levels: Optional[int] = 3


class EncoderConfig(BaseModel):
    """Configuration for encoder components"""
    type: ComponentType
    e_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Component configurations
    attention_comp: Optional[AttentionConfig] = None
    decomp_comp: Optional[DecompositionConfig] = None
    
    # Hierarchical-specific
    n_levels: Optional[int] = None
    level_configs: Optional[List[Dict[str, Any]]] = None


class DecoderConfig(BaseModel):
    """Configuration for decoder components"""
    type: ComponentType
    d_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1
    activation: str = "gelu"
    c_out: int
    
    # Component configurations
    attention_comp: Optional[AttentionConfig] = None
    decomp_comp: Optional[DecompositionConfig] = None


class SamplingConfig(BaseModel):
    """Configuration for sampling mechanisms"""
    type: ComponentType
    n_samples: int = 50
    quantile_levels: Optional[List[float]] = None
    
    # Bayesian-specific
    dropout_rate: Optional[float] = 0.1
    temperature: Optional[float] = 1.0
    # Some sampling components access kl_weight; provide lightweight default
    kl_weight: Optional[float] = 1e-5


class OutputHeadConfig(BaseModel):
    """Configuration for output heads"""
    type: ComponentType
    d_model: int
    c_out: int
    
    # Quantile-specific
    num_quantiles: Optional[int] = None


class LossConfig(BaseModel):
    """Configuration for loss functions"""
    type: ComponentType
    
    # Quantile-specific
    quantiles: Optional[List[float]] = None
    
    # Bayesian-specific
    prior_scale: Optional[float] = 1.0
    kl_weight: Optional[float] = 1.0


class BayesianConfig(BaseModel):
    """Configuration for Bayesian components"""
    enabled: bool = False
    layers_to_convert: List[str] = Field(default_factory=lambda: ["projection"])
    prior_scale: float = 1.0
    posterior_scale_init: float = -3.0
    kl_weight: float = 1.0


class BackboneConfig(BaseModel):
    """Configuration for backbone models (HF integration)"""
    type: Optional[ComponentType] = None
    model_name: Optional[str] = None
    use_backbone: bool = False
    
    # ChronosX-specific
    prediction_length: Optional[int] = None
    context_length: Optional[int] = None


class ModularAutoformerConfig(BaseModel):
    """
    Complete structured configuration for ModularAutoformer
    
    This replaces the flat Namespace configuration with a validated,
    type-safe configuration object as recommended by GCLI.
    """
    
    # Basic model parameters
    task_name: str = "long_term_forecast"
    seq_len: int
    pred_len: int
    label_len: int
    enc_in: int
    dec_in: int
    c_out: int
    c_out_evaluation: int
    d_model: int = 512
    
    # Component configurations
    attention: AttentionConfig
    decomposition: DecompositionConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    sampling: SamplingConfig
    output_head: OutputHeadConfig
    loss: LossConfig
    
    # Special configurations
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    
    # Additional parameters
    quantile_levels: Optional[List[float]] = None
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.1
    
    @validator('c_out_evaluation')
    def validate_c_out_evaluation(cls, v, values):
        """Ensure c_out_evaluation is consistent with quantile setup"""
        if 'quantile_levels' in values and values['quantile_levels']:
            # For quantile models, c_out should be c_out_evaluation * num_quantiles
            expected_c_out = v * len(values['quantile_levels'])
            if 'c_out' in values and values['c_out'] != expected_c_out:
                raise ValueError(f"c_out ({values['c_out']}) should equal c_out_evaluation ({v}) * num_quantiles ({len(values['quantile_levels'])})")
        return v
    
    @validator('encoder', 'decoder')
    def validate_component_configs(cls, v, values):
        """Ensure component configurations are consistent with parent config"""
        if hasattr(v, 'd_model') and 'd_model' in values:
            if v.d_model != values['d_model']:
                raise ValueError(f"Component d_model ({v.d_model}) must match parent d_model ({values['d_model']})")
        return v
    
    def to_namespace(self):
        """Convert to legacy Namespace format for backward compatibility"""
        from argparse import Namespace
        
        # Create flat dictionary for backward compatibility
        flat_dict = {
            'task_name': self.task_name,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'label_len': self.label_len,
            'enc_in': self.enc_in,
            'dec_in': self.dec_in,
            'c_out': self.c_out,
            'c_out_evaluation': self.c_out_evaluation,
            'd_model': self.d_model,
            'embed': self.embed,
            'freq': self.freq,
            'dropout': self.dropout,
            
            # Component types
            'attention_type': self.attention.type.value,
            'decomposition_type': self.decomposition.type.value,
            'encoder_type': self.encoder.type.value,
            'decoder_type': self.decoder.type.value,
            'sampling_type': self.sampling.type.value,
            'output_head_type': self.output_head.type.value,
            'loss_function_type': self.loss.type.value,
            
            # Component parameters
            'attention_params': self.attention.dict(exclude={'type'}),
            'decomposition_params': self.decomposition.dict(exclude={'type'}),
            'encoder_params': self.encoder.dict(exclude={'type'}),
            'decoder_params': self.decoder.dict(exclude={'type'}),
            'sampling_params': self.sampling.dict(exclude={'type'}),
            'output_head_params': self.output_head.dict(exclude={'type'}),
            'loss_params': self.loss.dict(exclude={'type'}),
            
            # Special configurations
            'quantile_levels': self.quantile_levels,
            'bayesian_layers': self.bayesian.layers_to_convert if self.bayesian.enabled else [],
            
            # Backbone configuration
            'use_backbone_component': self.backbone.use_backbone,
            'backbone_type': self.backbone.type.value if self.backbone.type else None,
        }
        
        return Namespace(**flat_dict)


def create_enhanced_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """
    Create a structured configuration for enhanced autoformer
    
    This replaces the flat configuration functions with structured Pydantic models.
    """
    
    # Extract basic parameters
    seq_len = kwargs.get('seq_len', 96)
    pred_len = kwargs.get('pred_len', 24)
    label_len = kwargs.get('label_len', 48)
    d_model = kwargs.get('d_model', 512)
    
    config = ModularAutoformerConfig(
        # Basic parameters
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=num_targets + num_covariates,
        dec_in=num_targets + num_covariates,
        c_out=num_targets,
        c_out_evaluation=num_targets,
        d_model=d_model,
        
        # Component configurations
        attention=AttentionConfig(
            type=ComponentType.ADAPTIVE_AUTOCORRELATION,
            d_model=d_model,
            n_heads=8,
            dropout=0.1,
            factor=1,
            output_attention=False
        ),
        
        decomposition=DecompositionConfig(
            type=ComponentType.LEARNABLE_DECOMP,
            kernel_size=25,
            input_dim=d_model
        ),
        
        encoder=EncoderConfig(
            type=ComponentType.ENHANCED_ENCODER,
            e_layers=2,
            d_model=d_model,
            n_heads=8,
            d_ff=2048,
            dropout=0.1,
            activation="gelu"
        ),
        
        decoder=DecoderConfig(
            type=ComponentType.ENHANCED_DECODER,
            d_layers=1,
            d_model=d_model,
            n_heads=8,
            d_ff=2048,
            dropout=0.1,
            activation="gelu",
            c_out=num_targets
        ),
        
        sampling=SamplingConfig(
            type=ComponentType.DETERMINISTIC
        ),
        
        output_head=OutputHeadConfig(
            type=ComponentType.STANDARD_HEAD,
            d_model=d_model,
            c_out=num_targets
        ),
        
        loss=LossConfig(
            type=ComponentType.MSE
        )
    )
    
    return config


def create_bayesian_enhanced_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create structured configuration for Bayesian enhanced autoformer"""
    
    # Start with enhanced config
    config = create_enhanced_config(num_targets, num_covariates, **kwargs)
    
    # Modify for Bayesian
    config.sampling.type = ComponentType.BAYESIAN
    config.sampling.n_samples = kwargs.get('n_samples', 50)
    config.loss.type = ComponentType.BAYESIAN_MSE
    
    # Enable Bayesian layers
    config.bayesian.enabled = True
    config.bayesian.layers_to_convert = kwargs.get('bayesian_layers', ['projection'])
    
    return config


def create_quantile_bayesian_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create structured configuration for quantile Bayesian autoformer"""
    
    quantile_levels = kwargs.get('quantile_levels', [0.1, 0.5, 0.9])
    
    # Start with Bayesian config
    config = create_bayesian_enhanced_config(num_targets, num_covariates, **kwargs)
    
    # Modify for quantile
    config.quantile_levels = quantile_levels
    config.c_out = num_targets * len(quantile_levels)
    config.sampling.quantile_levels = quantile_levels
    config.output_head.type = ComponentType.QUANTILE
    config.output_head.num_quantiles = len(quantile_levels)
    config.output_head.c_out = num_targets  # Base targets, not multiplied
    config.loss.type = ComponentType.BAYESIAN_QUANTILE
    config.loss.quantiles = quantile_levels
    
    return config


def create_hierarchical_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create structured configuration for hierarchical autoformer"""
    
    n_levels = kwargs.get('n_levels', 3)
    
    # Start with enhanced config
    config = create_enhanced_config(num_targets, num_covariates, **kwargs)
    
    # Modify for hierarchical
    config.attention.type = ComponentType.CROSS_RESOLUTION
    config.attention.n_levels = n_levels
    config.decomposition.type = ComponentType.WAVELET_DECOMP
    config.decomposition.levels = n_levels
    config.encoder.type = ComponentType.HIERARCHICAL_ENCODER
    config.encoder.n_levels = n_levels
    # Keep decoder enhanced unless a hierarchical decoder is later introduced
    config.decoder.type = ComponentType.ENHANCED_DECODER  # maintain consistency with encoder variant naming
    
    return config


# ---------------------------------------------------------------------------
# Additional configuration factory helpers (extending coverage)
# ---------------------------------------------------------------------------

def create_quantile_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create a deterministic quantile forecasting configuration.

    This variant produces quantile outputs without Bayesian sampling. It is a
    lighter alternative to `create_quantile_bayesian_config` when probabilistic
    uncertainty via MC/Bayesian sampling is not required.
    """
    quantile_levels = kwargs.get('quantile_levels', [0.1, 0.5, 0.9])
    base = create_enhanced_config(num_targets, num_covariates, **kwargs)
    base.quantile_levels = quantile_levels
    # Adjust output dimensionalities: internal c_out holds total channels
    base.c_out = num_targets * len(quantile_levels)
    base.output_head.type = ComponentType.QUANTILE
    base.output_head.num_quantiles = len(quantile_levels)
    base.output_head.c_out = num_targets  # base targets
    base.sampling.type = ComponentType.DETERMINISTIC
    base.sampling.quantile_levels = quantile_levels
    base.loss.type = ComponentType.QUANTILE_LOSS
    base.loss.quantiles = quantile_levels
    return base


def create_adaptive_mixture_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create configuration using adaptive mixture sampling.

    Adaptive mixture combines deterministic backbone representations with a
    learned sampling/mixing mechanism (e.g., gating across sample paths).
    """
    n_samples = kwargs.get('n_samples', 32)
    base = create_enhanced_config(num_targets, num_covariates, **kwargs)
    base.sampling.type = ComponentType.ADAPTIVE_MIXTURE
    base.sampling.n_samples = n_samples
    # Keep standard head & MSE unless caller overrides
    return base


def create_advanced_wavelet_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create configuration emphasizing advanced wavelet decomposition.

    Uses ADVANCED_WAVELET decomposition and hierarchical encoder to exploit
    multi-scale temporal structure.
    """
    levels = kwargs.get('levels', 3)
    base = create_enhanced_config(num_targets, num_covariates, **kwargs)
    base.decomposition.type = ComponentType.ADVANCED_WAVELET
    base.decomposition.levels = levels
    base.encoder.type = ComponentType.HIERARCHICAL_ENCODER
    base.encoder.n_levels = levels
    # Retain enhanced decoder unless a hierarchical decoder is later added
    return base


def create_temporal_conv_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create configuration utilizing temporal convolution encoder blocks."""
    base = create_enhanced_config(num_targets, num_covariates, **kwargs)
    base.encoder.type = ComponentType.TEMPORAL_CONV_ENCODER
    # Attention may be less central; still keep adaptive autocorrelation unless overridden
    return base


def create_meta_learning_adapter_config(num_targets: int, num_covariates: int, **kwargs) -> ModularAutoformerConfig:
    """Create configuration enabling meta-learning adapter within encoder.

    This variant sets the encoder type to META_LEARNING_ADAPTER allowing rapid
    adaptation across related time series tasks.
    """
    base = create_enhanced_config(num_targets, num_covariates, **kwargs)
    base.encoder.type = ComponentType.META_LEARNING_ADAPTER
    return base
