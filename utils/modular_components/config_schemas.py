"""
Configuration Schemas for Modular Components

These dataclasses define the configuration structure for all modular components,
providing type safety and validation for component parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
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
class BackboneConfig(ComponentConfig):
    """Configuration for backbone models"""
    backbone_type: str = "chronos"  # chronos, t5, minimal, custom
    model_name: str = "amazon/chronos-t5-tiny"
    pretrained: bool = True
    freeze_backbone: bool = False
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    
    # Fallback configuration
    fallback_models: List[str] = field(default_factory=lambda: [
        "google/flan-t5-small",
        "minimal"
    ])
    
    # Model-specific parameters
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    vocab_size: int = 1000


@dataclass
class EmbeddingConfig(ComponentConfig):
    """Configuration for embedding components"""
    embedding_type: str = "temporal"  # temporal, positional, hybrid, covariate
    
    # Input/output dimensions
    input_dim: int = 1
    output_dim: int = 256  # Should match d_model
    
    # Temporal embedding parameters
    embed_method: str = "timeF"  # timeF, fixed, learnable
    freq: str = "h"  # frequency for time features
    
    # Positional embedding parameters
    pos_encoding_type: str = "sinusoidal"  # sinusoidal, learnable, relative
    max_seq_len: int = 5000
    
    # Covariate handling
    covariate_strategy: str = "add"  # add, concat, gated, attention
    covariate_dim: Optional[int] = None
    use_time_features: bool = True
    use_learned_position: bool = False


@dataclass
class AttentionConfig(ComponentConfig):
    """Configuration for attention mechanisms"""
    attention_type: str = "self_attention"  # self_attention, autocorrelation, fourier, hybrid
    
    # Standard attention parameters
    num_heads: int = 8
    head_dim: Optional[int] = None  # If None, d_model // num_heads
    attention_dropout: float = 0.1
    
    # Autocorrelation parameters
    factor: int = 1
    adaptive_k: bool = True
    multi_scale: bool = True
    scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # Fourier attention parameters
    use_fft: bool = True
    frequency_domain: bool = True
    
    # Output parameters
    output_attention: bool = False
    mask_flag: bool = True


@dataclass
class ProcessorConfig(ComponentConfig):
    """Configuration for processing strategies"""
    processor_type: str = "seq2seq"  # seq2seq, encoder_only, hierarchical, autoregressive
    
    # Sequence processing parameters
    seq_len: int = 96
    pred_len: int = 24
    label_len: int = 48
    
    # Hierarchical processing parameters
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cross_scale_attention: bool = True
    
    # Processing strategy parameters
    use_decoder: bool = True
    decoder_strategy: str = "teacher_forcing"  # teacher_forcing, autoregressive
    pooling_method: str = "adaptive"  # adaptive, average, max, attention


@dataclass
class FeedForwardConfig(ComponentConfig):
    """Configuration for feed-forward networks"""
    ffn_type: str = "standard"  # standard, mixture_experts, adaptive, gated
    
    # Standard FFN parameters
    d_ff: int = 1024
    activation: str = "relu"  # relu, gelu, swish, mish
    use_bias: bool = True
    
    # Mixture of Experts parameters
    num_experts: int = 4
    expert_dropout: float = 0.1
    gate_type: str = "top_k"  # top_k, softmax, learned
    top_k: int = 2
    
    # Adaptive FFN parameters
    adaptive_method: str = "linear"  # linear, attention, meta
    adaptation_dim: int = 64


@dataclass
class LossConfig(ComponentConfig):
    """Configuration for loss functions"""
    loss_type: str = "mse"  # mse, mae, mape, smape, quantile, bayesian, multi_task
    
    # Basic loss parameters
    reduction: str = "mean"  # mean, sum, none
    
    # Quantile loss parameters
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    quantile_weights: Optional[List[float]] = None
    
    # Bayesian loss parameters
    kl_weight: float = 1e-5
    uncertainty_weight: float = 0.1
    
    # Multi-task loss parameters
    task_weights: Dict[str, float] = field(default_factory=dict)
    adaptive_weights: bool = False
    
    # Advanced loss parameters
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0


@dataclass
class OutputConfig(ComponentConfig):
    """Configuration for output heads"""
    output_type: str = "regression"  # regression, uncertainty, hierarchical, multi_task
    
    # Output dimensions
    output_dim: int = 1
    
    # Uncertainty quantification
    uncertainty_method: str = "mc_dropout"  # mc_dropout, bayesian, ensemble
    num_samples: int = 10
    
    # Multi-task outputs
    task_outputs: Dict[str, int] = field(default_factory=dict)
    
    # Hierarchical outputs
    scale_outputs: Dict[str, int] = field(default_factory=dict)
    
    # Output processing
    activation: Optional[str] = None  # None, relu, sigmoid, tanh
    use_batch_norm: bool = False
    use_layer_norm: bool = True


@dataclass
class ModularModelConfig:
    """Complete configuration for modular time series models"""
    
    # Model identification
    model_name: str = "ModularHFAutoformer"
    model_version: str = "1.0"
    
    # Global parameters
    seq_len: int = 96
    pred_len: int = 24
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    d_model: int = 256
    
    # Component configurations
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    feedforward: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    
    # Validation and compatibility
    validate_config: bool = True
    strict_compatibility: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and synchronization"""
        # Synchronize d_model across all components
        components = [self.backbone, self.embedding, self.attention, 
                     self.processor, self.feedforward, self.loss, self.output]
        
        for component in components:
            if hasattr(component, 'd_model'):
                component.d_model = self.d_model
        
        # Set input/output dimensions
        self.embedding.input_dim = self.enc_in
        self.embedding.output_dim = self.d_model
        self.output.output_dim = self.c_out
        
        # Set sequence lengths
        self.processor.seq_len = self.seq_len
        self.processor.pred_len = self.pred_len
        
        if self.validate_config:
            self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration consistency"""
        # Check dimension compatibility
        assert self.embedding.output_dim == self.d_model, \
            f"Embedding output dim {self.embedding.output_dim} != d_model {self.d_model}"
        
        # Check sequence length consistency
        assert self.processor.seq_len == self.seq_len, \
            f"Processor seq_len {self.processor.seq_len} != global seq_len {self.seq_len}"
        
        # Check attention head compatibility
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
