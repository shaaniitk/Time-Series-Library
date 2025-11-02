# models/celestial_modules/config.py

import logging
from dataclasses import dataclass, field, MISSING
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

@dataclass
class CelestialPGATConfig:
    """Configuration for the Celestial Enhanced PGAT Model."""
    # Core parameters
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    enc_in: int = 118
    dec_in: int = 4
    c_out: int = 4
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 3
    d_layers: int = 2
    dropout: float = 0.1
    embed: str = 'timeF'
    freq: str = 'h'

    # Celestial system parameters
    use_celestial_graph: bool = True
    celestial_fusion_layers: int = 3
    num_celestial_bodies: int = 13

    # Petri Net Architecture
    use_petri_net_combiner: bool = True
    num_message_passing_steps: int = 2
    edge_feature_dim: int = 6
    use_temporal_attention: bool = True
    use_spatial_attention: bool = True
    bypass_spatiotemporal_with_petri: bool = True

    # Enhanced Features
    use_mixture_decoder: bool = False
    use_stochastic_learner: bool = False
    use_hierarchical_mapping: bool = False
    use_gated_graph_combiner: bool = False
    use_hierarchical_mapper: bool = False  # Alias for consistency
    use_efficient_covariate_interaction: bool = False

    # Adaptive TopK Pooling
    enable_adaptive_topk: bool = False
    adaptive_topk_ratio: float = 0.5
    adaptive_topk_temperature: float = 1.0
    adaptive_topk_k: Optional[int] = None

    # Stochastic Control
    use_stochastic_control: bool = False
    stochastic_temperature_start: float = 1.0
    stochastic_temperature_end: float = 0.1
    stochastic_decay_steps: int = 1000
    stochastic_noise_std: float = 1.0
    stochastic_use_external_step: bool = False

    # MDN Decoder
    enable_mdn_decoder: bool = False
    mdn_components: int = 5
    mdn_sigma_min: float = 1e-3
    mdn_use_softplus: bool = True

    # Target Autocorrelation
    use_target_autocorrelation: bool = True
    target_autocorr_layers: int = 2

    # Calendar Effects
    use_calendar_effects: bool = True
    calendar_embedding_dim: int = 128  # d_model // 4

    # Celestial-to-Target Attention
    use_celestial_target_attention: bool = True
    celestial_target_use_gated_fusion: bool = True
    celestial_target_diagnostics: bool = True
    use_c2t_edge_bias: bool = False
    c2t_edge_bias_weight: float = 0.2
    c2t_aux_rel_loss_weight: float = 0.0

    # Wave aggregation
    aggregate_waves_to_celestial: bool = True
    num_input_waves: int = 118
    target_wave_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # Dynamic Spatiotemporal Encoder
    use_dynamic_spatiotemporal_encoder: bool = True

    # Logging and Diagnostics
    verbose_logging: bool = False
    enable_memory_debug: bool = False
    enable_memory_diagnostics: bool = False
    collect_diagnostics: bool = True
    enable_fusion_diagnostics: bool = True
    fusion_diag_batches: int = 10

    # Covariate Interaction
    enable_target_covariate_attention: bool = False

    # Sequential Mixture Decoder
    use_sequential_mixture_decoder: bool = False

    # Multi-Scale Context Fusion
    use_multi_scale_context: bool = True
    context_fusion_mode: str = 'multi_scale'  # 'simple', 'gated', 'attention', 'multi_scale'
    short_term_kernel_size: int = 5
    medium_term_kernel_size: int = 15
    long_term_kernel_size: int = 0  # 0 means global average
    context_fusion_dropout: float = 0.1
    enable_context_diagnostics: bool = False

    # Internal / Derived
    celestial_dim: int = 32
    celestial_feature_dim: int = 416 # 13 * 32
    num_graph_nodes: int = 13
    expected_embedding_input_dim: int = 118

    @classmethod
    def from_original_configs(cls, configs):
        """Creates a structured config from the original attribute-based config object."""
        kwargs = {}
        for f in cls.__dataclass_fields__.values():
            value = getattr(configs, f.name, f.default)
            # Handle default_factory for lists
            if hasattr(f, 'default_factory') and f.default_factory is not MISSING:
                if value == f.default:
                    value = f.default_factory()
            kwargs[f.name] = value
        return cls(**kwargs)

    def __post_init__(self):
        """Post-initialization checks and derivations."""
        # Validate and adjust d_model to be compatible with n_heads
        if self.d_model % self.n_heads != 0:
            original_d_model = self.d_model
            self.d_model = ((self.d_model // self.n_heads) + 1) * self.n_heads
            logger.warning(
                "d_model=%s adjusted to %s for attention head compatibility (n_heads=%s)",
                original_d_model, self.d_model, self.n_heads
            )

        # Calculate calendar embedding dimension if not explicitly set
        self.calendar_embedding_dim = self.d_model // 4

        # FIX ISSUE #1: Honor celestial_dim from config when safe
        base_celestial_dim = 32
        config_celestial_dim = self.celestial_dim  # User-specified value from dataclass default or config
        
        if config_celestial_dim != 32:  # User explicitly set a non-default value
            # Validate config value is compatible with n_heads
            if config_celestial_dim % self.n_heads == 0:
                self.celestial_dim = config_celestial_dim
                logger.info(
                    "Using celestial_dim=%s from config (compatible with n_heads=%s)",
                    config_celestial_dim, self.n_heads
                )
            else:
                # Round up to nearest head-compatible value
                self.celestial_dim = ((config_celestial_dim + self.n_heads - 1) // self.n_heads) * self.n_heads
                logger.warning(
                    "celestial_dim=%s adjusted to %s for n_heads=%s compatibility",
                    config_celestial_dim, self.celestial_dim, self.n_heads
                )
        else:
            # No explicit config - compute minimal compatible dimension
            self.celestial_dim = ((base_celestial_dim + self.n_heads - 1) // self.n_heads) * self.n_heads
        
        self.celestial_feature_dim = self.num_celestial_bodies * self.celestial_dim
        
        # Validate d_model consistency
        if self.use_celestial_graph and self.aggregate_waves_to_celestial:
            if self.celestial_feature_dim != self.d_model:
                logger.warning(
                    "celestial_feature_dim=%s does not match d_model=%s; this may cause dimension issues",
                    self.celestial_feature_dim, self.d_model
                )

        # Set the number of graph nodes based on aggregation
        if self.use_celestial_graph and self.aggregate_waves_to_celestial:
            self.num_graph_nodes = self.num_celestial_bodies
        else:
            self.num_graph_nodes = self.enc_in

        # Set adaptive_topk_k
        if self.enable_adaptive_topk:
            k = max(1, int(round(self.adaptive_topk_ratio * self.num_graph_nodes)))
            self.adaptive_topk_k = min(k, self.num_graph_nodes)