"""
Structured Configuration Schemas for Modular Autoformer

This module implements the GCLI recommendations for replacing the flat Namespace
configuration with structured, validated configuration objects using Pydantic.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum


class ComponentType(str, Enum):
    """Enumeration of all available component types"""
    # Attention components
    MULTI_HEAD = "multi_head"
    AUTOCORRELATION = "autocorrelation"
    ADAPTIVE_AUTOCORRELATION = "adaptive_autocorrelation_layer"
    
    # Phase 2: Fourier Attention Components
    FOURIER_ATTENTION = "fourier_attention"
    FOURIER_BLOCK = "fourier_block"
    FOURIER_CROSS_ATTENTION = "fourier_cross_attention"
    
    # Phase 2: Wavelet Attention Components
    WAVELET_ATTENTION = "wavelet_attention"
    WAVELET_DECOMPOSITION = "wavelet_decomposition"
    ADAPTIVE_WAVELET_ATTENTION = "adaptive_wavelet_attention"
    MULTI_SCALE_WAVELET_ATTENTION = "multi_scale_wavelet_attention"
    
    # Phase 2: Enhanced AutoCorrelation Components
    ENHANCED_AUTOCORRELATION = "enhanced_autocorrelation"
    NEW_ADAPTIVE_AUTOCORRELATION_LAYER = "new_adaptive_autocorrelation_layer"
    HIERARCHICAL_AUTOCORRELATION = "hierarchical_autocorrelation"
    
    # Phase 2: Bayesian Attention Components
    BAYESIAN_ATTENTION = "bayesian_attention"
    BAYESIAN_MULTI_HEAD_ATTENTION = "bayesian_multi_head_attention"
    VARIATIONAL_ATTENTION = "variational_attention"
    BAYESIAN_CROSS_ATTENTION = "bayesian_cross_attention"
    
    # Phase 2: Adaptive Components
    META_LEARNING_ADAPTER_ATTN = "meta_learning_adapter"
    ADAPTIVE_MIXTURE_ATTN = "adaptive_mixture"
    
    # Phase 2: Temporal Convolution Attention Components
    CAUSAL_CONVOLUTION = "causal_convolution"
    TEMPORAL_CONV_NET = "temporal_conv_net"
    CONVOLUTIONAL_ATTENTION = "convolutional_attention"
    
    # Other attention types
    SPARSE = "sparse"
    LOG_SPARSE = "log_sparse"
    PROB_SPARSE = "prob_sparse"
    CROSS_RESOLUTION = "cross_resolution_attention"
    
    # Decomposition components
    MOVING_AVG = "moving_avg"
    LEARNABLE_DECOMP = "learnable_decomp"
    WAVELET_DECOMP = "wavelet_hierarchical_decomp"
    ADVANCED_WAVELET = "advanced_wavelet_decomp"
    
    # Encoder/Decoder components
    STANDARD_ENCODER = "standard_encoder"
    STANDARD_DECODER = "standard_decoder"
    ENHANCED_ENCODER = "enhanced_encoder"
    ENHANCED_DECODER = "enhanced_decoder"
    HIERARCHICAL_ENCODER = "hierarchical_encoder"
    HIERARCHICAL_DECODER = "hierarchical_decoder"
    TEMPORAL_CONV_ENCODER = "temporal_conv_encoder"
    META_LEARNING_ADAPTER = "meta_learning_adapter"
    
    # Sampling components
    DETERMINISTIC = "deterministic"
    BAYESIAN = "bayesian"
    MONTE_CARLO = "monte_carlo"
    ADAPTIVE_MIXTURE = "adaptive_mixture"
    
    # Output head components
    STANDARD_HEAD = "standard"
    QUANTILE = "quantile"
    BAYESIAN_HEAD = "bayesian_head"
    
    # Loss components
    MSE = "mse"
    MAE = "mae"
    QUANTILE_LOSS = "quantile"
    BAYESIAN_MSE = "bayesian"
    BAYESIAN_QUANTILE = "bayesian_quantile"
    ADAPTIVE_AUTOFORMER_LOSS = "adaptive_autoformer_loss"
    
    # Advanced metric losses
    MAPE_LOSS = "mape_loss"
    SMAPE_LOSS = "smape_loss"
    MASE_LOSS = "mase_loss"
    PS_LOSS = "ps_loss"
    FOCAL_LOSS = "focal_loss"
    
    # Advanced adaptive losses
    FREQUENCY_AWARE_LOSS = "frequency_aware_loss"
    MULTI_QUANTILE_LOSS = "multi_quantile_loss"
    UNCERTAINTY_CALIBRATION_LOSS = "uncertainty_calibration_loss"
    ADAPTIVE_LOSS_WEIGHTING = "adaptive_loss_weighting"
    
    # Backbone components (for future HF integration)
    CHRONOS = "chronos"
    CHRONOS_X = "chronos_x"


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
    config.encoder.type = ComponentType.HIERARCHICAL
    config.encoder.n_levels = n_levels
    config.decoder.type = ComponentType.ENHANCED  # Keep decoder standard
    
    return config
