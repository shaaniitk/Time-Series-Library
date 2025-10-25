"""
Celestial Enhanced PGAT - Revolutionary Astrological AI for Financial Markets

This model combines the Enhanced SOTA PGAT with celestial body graph nodes,
creating the world's first astronomically-informed time series forecasting model.
"""

import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.modular.graph.celestial_body_nodes import CelestialBodyNodes, CelestialBody
from layers.modular.graph.celestial_graph_combiner import CelestialGraphCombiner
from layers.modular.graph.celestial_petri_net_combiner import CelestialPetriNetCombiner
from layers.modular.decoder.mdn_decoder import MDNDecoder
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder
from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureDensityDecoder
from layers.modular.embedding.hierarchical_mapper import HierarchicalTemporalSpatialMapper
from layers.modular.encoder.spatiotemporal_encoding import (
    JointSpatioTemporalEncoding,
    DynamicJointSpatioTemporalEncoding,
)
from layers.modular.graph.gated_graph_combiner import GatedGraphCombiner
from layers.Embed import DataEmbedding  # type: ignore[assignment]
from utils.celestial_wave_aggregator import CelestialWaveAggregator, CelestialDataProcessor
from layers.modular.aggregation.phase_aware_celestial_processor import PhaseAwareCelestialProcessor
from layers.modular.decoder.target_autocorrelation_module import (
    TargetAutocorrelationModule, 
    DualStreamDecoder
)
from layers.modular.embedding.calendar_aware_embedding import (
    EnhancedTemporalEmbedding,
    CalendarEffectsEncoder
)


class Model(nn.Module):
    """
    Celestial Enhanced PGAT - Astrological AI for Financial Time Series
    
    Revolutionary model that represents market influences as celestial bodies
    with learned relationships based on astrological aspects and market dynamics.
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.verbose_logging = bool(getattr(configs, "verbose_logging", False))
        # Unify memory diagnostic flags
        self.enable_memory_debug = bool(getattr(configs, "enable_memory_debug", False))
        self.enable_memory_diagnostics = bool(getattr(configs, "enable_memory_diagnostics", False))
        
        # Core parameters
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len  
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 3)
        self.d_layers = getattr(configs, 'd_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # Celestial system parameters
        self.use_celestial_graph = getattr(configs, 'use_celestial_graph', True)
        self.celestial_fusion_layers = getattr(configs, 'celestial_fusion_layers', 3)
        self.num_celestial_bodies = len(CelestialBody)  # 13 celestial bodies
        
        # 🚀 NEW: Petri Net Architecture Configuration
        self.use_petri_net_combiner = getattr(configs, 'use_petri_net_combiner', True)
        self.num_message_passing_steps = getattr(configs, 'num_message_passing_steps', 2)
        self.edge_feature_dim = getattr(configs, 'edge_feature_dim', 6)  # theta_diff, phi_diff, etc.
        self.use_temporal_attention = getattr(configs, 'use_temporal_attention', True)
        self.use_spatial_attention = getattr(configs, 'use_spatial_attention', True)
        # Option to bypass legacy spatiotemporal encoder when Petri + edge-conditioned attention is active
        self.bypass_spatiotemporal_with_petri = bool(getattr(configs, 'bypass_spatiotemporal_with_petri', True))
        
        # Enhanced features - FIXED: Defaults match production config
        self.use_mixture_decoder = getattr(configs, 'use_mixture_decoder', False)  # Production default: False
        self.use_stochastic_learner = getattr(configs, 'use_stochastic_learner', False)  # Production default: False
        self.use_hierarchical_mapping = getattr(configs, 'use_hierarchical_mapping', False)  # Production default: False
        self.collect_diagnostics = bool(getattr(configs, 'collect_diagnostics', True))
        self.use_efficient_covariate_interaction = getattr(configs, 'use_efficient_covariate_interaction', False)
        
        # 🎲 NEW: MDN Decoder for probabilistic forecasting (Phase 1)
        self.enable_mdn_decoder = bool(getattr(configs, 'enable_mdn_decoder', False))
        self.mdn_components = int(getattr(configs, 'mdn_components', 5))
        self.mdn_sigma_min = float(getattr(configs, 'mdn_sigma_min', 1e-3))
        self.mdn_use_softplus = bool(getattr(configs, 'mdn_use_softplus', True))
        
        # 🎯 NEW: Target autocorrelation modeling
        self.use_target_autocorrelation = getattr(configs, 'use_target_autocorrelation', True)
        self.target_autocorr_layers = getattr(configs, 'target_autocorr_layers', 2)
        
        # 📅 NEW: Calendar effects modeling
        self.use_calendar_effects = getattr(configs, 'use_calendar_effects', True)
        self.calendar_embedding_dim = getattr(configs, 'calendar_embedding_dim', self.d_model // 4)
        
        # --- FIXED: Dimension Consistency Management ---
        # Determine celestial feature dimension
        self.celestial_feature_dim = self.num_celestial_bodies * 32  # 13 × 32 = 416
        
        # Validate d_model configuration
        if self.use_celestial_graph and getattr(configs, 'aggregate_waves_to_celestial', True):
            # For celestial mode, d_model should match or be compatible with celestial features
            if self.d_model != self.celestial_feature_dim:
                if self.d_model < self.celestial_feature_dim:
                    self.logger.warning(
                        "d_model=%s is smaller than celestial_feature_dim=%s. This will cause information loss!",
                        self.d_model, self.celestial_feature_dim
                    )
                    # Keep user's d_model but warn about information loss
                else:
                    # d_model is larger, which is fine - we can project up
                    self.logger.info(
                        "d_model=%s is larger than celestial_feature_dim=%s. Will project celestial features up.",
                        self.d_model, self.celestial_feature_dim
                    )
            
            # Ensure d_model is divisible by n_heads for attention
            if self.d_model % self.n_heads != 0:
                original_d_model = self.d_model
                self.d_model = ((self.d_model // self.n_heads) + 1) * self.n_heads
                self.logger.warning(
                    "d_model=%s adjusted to %s for attention head compatibility (n_heads=%s)",
                    original_d_model, self.d_model, self.n_heads
                )
        else:
            # For non-celestial mode, ensure d_model is compatible with n_heads
            if self.d_model % self.n_heads != 0:
                original_d_model = self.d_model
                self.d_model = ((self.d_model // self.n_heads) + 1) * self.n_heads
                self.logger.warning(
                    "d_model=%s adjusted to %s for attention head compatibility (n_heads=%s)",
                    original_d_model, self.d_model, self.n_heads
                )
        # --- End of dimension management ---
        
        # Wave aggregation settings
        self.aggregate_waves_to_celestial = getattr(configs, 'aggregate_waves_to_celestial', True)
        self.num_input_waves = getattr(configs, 'num_input_waves', 118)
        self.target_wave_indices = getattr(configs, 'target_wave_indices', [0, 1, 2, 3])

        # Initialize celestial fusion components; will be replaced when celestial graph is active
        self.celestial_fusion_attention: Optional[nn.MultiheadAttention] = None
        self.celestial_fusion_gate: Optional[nn.Sequential] = None
        
        # Enhanced Phase-Aware Wave Aggregation System
        if self.aggregate_waves_to_celestial:
            # Use the new phase-aware processor that understands the actual feature structure
            self.phase_aware_processor = PhaseAwareCelestialProcessor(
                num_input_waves=self.num_input_waves,
                celestial_dim=32,  # Rich 32D representation per celestial body
                waves_per_body=9,  # Average waves per celestial body
                num_heads=self.n_heads
            )
            
            # Keep the old processor for target extraction
            self.wave_aggregator = CelestialWaveAggregator(
                num_input_waves=self.num_input_waves,
                num_celestial_bodies=self.num_celestial_bodies
            )
            self.data_processor = CelestialDataProcessor(
                self.wave_aggregator,
                target_indices=self.target_wave_indices
            )
            
            # New projection layer to fix information bottleneck
            self.rich_feature_to_celestial = nn.Linear(self.num_celestial_bodies * 32, self.num_celestial_bodies)
        
        self._log_configuration_summary()
        
        # Input embeddings - FIXED: Correct dimension flow
        if self.aggregate_waves_to_celestial:
            # Phase-aware processor outputs 13 celestial bodies × 32D = 416D
            celestial_output_dim = self.celestial_feature_dim  # 416D
            
            # CRITICAL FIX: Project celestial features to d_model before embedding
            self.celestial_projection = nn.Sequential(
                nn.Linear(celestial_output_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
            
            if self.d_model < celestial_output_dim:
                self.logger.warning(
                    "Celestial projection compresses: %sD → %sD (%.1fx compression)",
                    celestial_output_dim, self.d_model, celestial_output_dim / self.d_model
                )
            elif self.d_model > celestial_output_dim:
                self.logger.info(
                    "Celestial projection expands: %sD → %sD (%.1fx expansion)",
                    celestial_output_dim, self.d_model, self.d_model / celestial_output_dim
                )
            else:
                self.logger.info(
                    "Celestial projection preserves: %sD → %sD (1.0x)",
                    celestial_output_dim, self.d_model
                )
                
            # Embedding expects d_model dimensions (after projection)
            embedding_input_dim = self.d_model
        else:
            # Non-celestial mode: use original features
            embedding_input_dim = self.enc_in
            self.celestial_projection = None
            
        self.enc_embedding = DataEmbedding(
            embedding_input_dim, self.d_model, configs.embed, configs.freq, self.dropout
        )
        
        # Store the expected embedding input dimension for validation
        self.expected_embedding_input_dim = embedding_input_dim
        self.dec_embedding = DataEmbedding(
            self.dec_in, self.d_model, configs.embed, configs.freq, self.dropout
        )

        # Dedicated memory logger dumps to file (configured by training script)
        import logging as _logging  # local import to avoid polluting module namespace
        self.memory_logger = _logging.getLogger("scripts.train.train_celestial_production.memory")
        
        # Celestial Body Graph System
        if self.use_celestial_graph:
            self.celestial_nodes = CelestialBodyNodes(
                d_model=self.d_model,
                num_aspects=5
            )
            
            # 🚀 PETRI NET COMBINER: Preserves edge features as vectors (NO compression!)
            if self.use_petri_net_combiner:
                self.logger.info(
                    "🚀 Initializing Petri Net Combiner with message passing (edge_feature_dim=%d, num_steps=%d)",
                    self.edge_feature_dim, self.num_message_passing_steps
                )
                self.celestial_combiner = CelestialPetriNetCombiner(
                    num_nodes=self.num_celestial_bodies,
                    d_model=self.d_model,
                    edge_feature_dim=self.edge_feature_dim,  # 6D edge features preserved!
                    num_message_passing_steps=self.num_message_passing_steps,
                    num_attention_heads=self.n_heads,
                    dropout=self.dropout,
                    use_temporal_attention=self.use_temporal_attention,
                    use_spatial_attention=self.use_spatial_attention,
                    use_gradient_checkpointing=True,  # Memory optimization
                    enable_memory_debug=bool(getattr(configs, 'petri_memory_debug', False)),
                    memory_debug_prefix="PETRI"
                )
                self.logger.info("✅ Petri Net Combiner initialized - memory efficient with ZERO information loss!")
            else:
                # OLD COMBINER: Uses fusion layers
                self.logger.warning(
                    "⚠️ Using legacy CelestialGraphCombiner (fusion_layers=%d)", self.celestial_fusion_layers
                )
                self.celestial_combiner = CelestialGraphCombiner(
                    num_nodes=self.num_celestial_bodies,
                    d_model=self.d_model,
                    num_attention_heads=self.n_heads,
                    fusion_layers=self.celestial_fusion_layers,
                    dropout=self.dropout,
                    use_gradient_checkpointing=True  # Enable gradient checkpointing for memory efficiency
                )

            fusion_dim_cfg = getattr(configs, 'celestial_fusion_dim', min(self.d_model, 64))
            fusion_dim = max(self.n_heads, fusion_dim_cfg)
            if fusion_dim % self.n_heads != 0:
                fusion_dim = ((fusion_dim // self.n_heads) + 1) * self.n_heads
            self.celestial_fusion_dim = fusion_dim
            self.celestial_query_projection = nn.Linear(self.d_model, self.celestial_fusion_dim)
            self.celestial_key_projection = nn.Linear(self.d_model, self.celestial_fusion_dim)
            self.celestial_value_projection = nn.Linear(self.d_model, self.celestial_fusion_dim)
            self.celestial_output_projection = nn.Linear(self.celestial_fusion_dim, self.d_model)
            self.celestial_fusion_attention = nn.MultiheadAttention(
                embed_dim=self.celestial_fusion_dim,
                num_heads=self.n_heads,
                dropout=self.dropout,
                batch_first=True
            )

            # FIXED: Standardized celestial graph processing
            # Always use celestial bodies as graph nodes for consistency
            if self.aggregate_waves_to_celestial:
                # After aggregation: d_model → 13 celestial nodes
                self.feature_to_celestial = nn.Linear(self.d_model, self.num_celestial_bodies)
                self.celestial_to_feature = nn.Linear(self.num_celestial_bodies, self.d_model)
            else:
                # Before aggregation: d_model → 13 celestial nodes (learned mapping)
                self.feature_to_celestial = nn.Linear(self.d_model, self.num_celestial_bodies)
                self.celestial_to_feature = nn.Linear(self.num_celestial_bodies, self.d_model)
            self.celestial_fusion_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
                nn.Sigmoid()
            )
        
        # Hierarchical Mapping - adjust for celestial aggregation
        if self.use_hierarchical_mapping:
            if self.use_celestial_graph and self.aggregate_waves_to_celestial:
                # Use celestial body dimensions
                num_nodes_for_mapping = self.num_celestial_bodies
            else:
                # Use original input dimensions
                num_nodes_for_mapping = self.enc_in
                
            self.hierarchical_mapper = HierarchicalTemporalSpatialMapper(
                d_model=self.d_model,
                num_nodes=num_nodes_for_mapping,
                n_heads=self.n_heads,
                num_attention_layers=2
            )
            self.hierarchical_projection = nn.Linear(num_nodes_for_mapping * self.d_model, self.d_model)
        
        # Spatiotemporal Encoding - adjust for celestial aggregation
        if self.use_celestial_graph and self.aggregate_waves_to_celestial:
            # Use celestial body dimensions
            num_nodes_for_encoding = self.num_celestial_bodies
        else:
            # Use original input dimensions
            num_nodes_for_encoding = self.enc_in
            
        self.num_graph_nodes = num_nodes_for_encoding

        if self.use_efficient_covariate_interaction:
            if self.d_model % self.num_graph_nodes != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by num_graph_nodes ({self.num_graph_nodes}) for efficient processing.")
            node_dim = self.d_model // self.num_graph_nodes

            # Processes the covariate graph with a shared weight matrix (graph convolution)
            self.covariate_interaction_layer = nn.Linear(node_dim, node_dim)
            
            # Fuses each target's features with the aggregated summary from the covariate graph.
            self.target_fusion_layer = nn.Sequential(
                nn.Linear(node_dim * 2, node_dim), # Combines target node features and covariate context
                nn.GELU(),
                nn.Linear(node_dim, node_dim)
            )
        
        self.use_dynamic_spatiotemporal_encoder = getattr(
            configs,
            'use_dynamic_spatiotemporal_encoder',
            True
        )
            
        self.spatiotemporal_encoder = JointSpatioTemporalEncoding(
            d_model=self.d_model,
            seq_len=self.seq_len,
            num_nodes=num_nodes_for_encoding,
            num_heads=self.n_heads,
            dropout=self.dropout
        )

        self.use_dynamic_spatiotemporal_encoder = getattr(
            configs,
            'use_dynamic_spatiotemporal_encoder',
            True
        )
        self.dynamic_spatiotemporal_encoder: Optional[DynamicJointSpatioTemporalEncoding]
        if self.use_dynamic_spatiotemporal_encoder:
            self.dynamic_spatiotemporal_encoder = DynamicJointSpatioTemporalEncoding(
                d_model=self.d_model,
                seq_len=self.seq_len,
                num_nodes=num_nodes_for_encoding,
                num_heads=self.n_heads,
                dropout=self.dropout
            )
        else:
            self.dynamic_spatiotemporal_encoder = None
        
        # Traditional graph learning (for comparison/fallback) - adjust for celestial aggregation
        if self.use_celestial_graph and self.aggregate_waves_to_celestial:
            # Use celestial body dimensions for adjacency matrix
            adj_output_dim = self.num_celestial_bodies * self.num_celestial_bodies
        else:
            # Use original input dimensions
            adj_output_dim = self.enc_in * self.enc_in
            
        self.traditional_graph_learner = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, adj_output_dim),
            nn.Tanh()
        )
        
        # Stochastic Graph Learner - adjust for celestial aggregation
        if self.use_stochastic_learner:
            if self.use_celestial_graph and self.aggregate_waves_to_celestial:
                # Use celestial body dimensions for adjacency matrix
                adj_output_dim = self.num_celestial_bodies * self.num_celestial_bodies
            else:
                # Use original input dimensions
                adj_output_dim = self.enc_in * self.enc_in
                
            self.stochastic_mean = nn.Linear(self.d_model, adj_output_dim)
            self.stochastic_logvar = nn.Linear(self.d_model, adj_output_dim)
        
        # Graph attention layers - use edge-conditioned version for Petri net
        if self.use_petri_net_combiner:
            # 🚀 EDGE-CONDITIONED ATTENTION: Uses rich edge features directly!
            from layers.modular.graph.adjacency_aware_attention import EdgeConditionedGraphAttention
            self.graph_attention_layers = nn.ModuleList([
                EdgeConditionedGraphAttention(
                    d_model=self.d_model,
                    d_ff=self.d_model,
                    n_heads=self.n_heads,
                    edge_feature_dim=self.edge_feature_dim,  # 6D rich features!
                    dropout=self.dropout
                ) for _ in range(self.e_layers)
            ])
            self.logger.info(
                "🚀 Using EdgeConditionedGraphAttention - ZERO information loss from edge features!"
            )
        else:
            # OLD: Adjacency-aware attention with scalar adjacency
            from layers.modular.graph.adjacency_aware_attention import AdjacencyAwareGraphAttention
            self.graph_attention_layers = nn.ModuleList([
                AdjacencyAwareGraphAttention(
                    d_model=self.d_model, 
                    d_ff=self.d_model, 
                    n_heads=self.n_heads, 
                    dropout=self.dropout,
                    use_adjacency_mask=True
                ) for _ in range(self.e_layers)
            ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                self.d_model, self.n_heads, self.dropout
            ) for _ in range(self.d_layers)
        ])
        
        # Output projection
        if self.use_mixture_decoder:
            # Use sequential mixture decoder to preserve temporal structure
            self.mixture_decoder = SequentialMixtureDensityDecoder(
                d_model=self.d_model,
                pred_len=self.pred_len,
                num_components=3,
                num_targets=self.c_out,
                num_decoder_layers=2,
                num_heads=self.n_heads,
                dropout=self.dropout
            )
        
        # Always have a fallback projection layer
        self.projection = nn.Linear(self.d_model, self.c_out)
        
        # � MDN Decoder for Probabilistic Forecasting (Phase 1)
        self.mdn_decoder: Optional[MDNDecoder] = None
        if self.enable_mdn_decoder:
            self.mdn_decoder = MDNDecoder(
                d_input=self.d_model,
                n_targets=self.c_out,
                n_components=self.mdn_components,
                sigma_min=self.mdn_sigma_min,
                use_softplus=self.mdn_use_softplus,
            )
            self.logger.info(
                "🎲 MDN Decoder enabled | components=%d sigma_min=%.1e use_softplus=%s",
                self.mdn_components,
                self.mdn_sigma_min,
                self.mdn_use_softplus,
            )
        
        # �🎯 Target Autocorrelation Module
        if self.use_target_autocorrelation:
            self.dual_stream_decoder = DualStreamDecoder(
                d_model=self.d_model,
                num_targets=self.c_out,
                num_heads=self.n_heads,
                dropout=self.dropout
            )
            self.logger.info(
                "🎯 Target Autocorrelation Module enabled with %d layers",
                self.target_autocorr_layers
            )
        else:
            self.dual_stream_decoder = None
        
        # 📅 Calendar Effects Encoder
        if self.use_calendar_effects:
            self.calendar_effects_encoder = CalendarEffectsEncoder(self.calendar_embedding_dim)
            self.calendar_fusion = nn.Sequential(
                nn.Linear(self.d_model + self.calendar_embedding_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
            self.logger.info(
                "📅 Calendar Effects Module enabled with %dD embeddings",
                self.calendar_embedding_dim
            )
        else:
            self.calendar_effects_encoder = None
            self.calendar_fusion = None
        
        # Market context encoder
        self.market_context_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Context-aware adjacency fusion weights (learned convex weights for 3 adj sources)
        self.adj_weight_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 3)
        )
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _log_info(self, message: str, *args: Any) -> None:
        """Log informational messages when verbose logging is enabled."""
        if self.verbose_logging:
            self.logger.info(message, *args)

    def _log_debug(self, message: str, *args: Any) -> None:
        """Log debug diagnostics when diagnostics are requested."""
        if self.collect_diagnostics or self.verbose_logging:
            self.logger.debug(message, *args)

    def _log_configuration_summary(self) -> None:
        """Emit a concise configuration summary when verbose logging is enabled."""
        if not self.verbose_logging:
            return
        self.logger.info(
            (
                "Initializing Celestial Enhanced PGAT | seq_len=%s pred_len=%s d_model=%s "
                "celestial_bodies=%s wave_aggregation=%s mixture_decoder=%s stochastic_learner=%s "
                "hierarchical_mapping=%s"
            ),
            self.seq_len,
            self.pred_len,
            self.d_model,
            self.num_celestial_bodies,
            self.aggregate_waves_to_celestial,
            self.use_mixture_decoder,
            self.use_stochastic_learner,
            self.use_hierarchical_mapping,
        )
        if self.aggregate_waves_to_celestial:
            self.logger.info(
                "Phase-aware aggregation configured | input_waves=%s target_wave_indices=%s",
                self.num_input_waves,
                self.target_wave_indices,
            )

    def _debug_memory(self, stage: str) -> None:
        """Emit lightweight memory diagnostics when requested."""
        if not (self.enable_memory_debug or self.collect_diagnostics or self.verbose_logging):
            return
        try:
            import psutil  # type: ignore[import-not-found]
            process = psutil.Process(os.getpid())
            cpu_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.debug("MODEL_DEBUG [%s]: CPU=%0.1fMB", stage, cpu_mb)
        except Exception:  # pragma: no cover - best effort diagnostics
            self.logger.debug("MODEL_DEBUG [%s]: memory diagnostic unavailable", stage)

    @staticmethod
    def _move_to_cpu(value: Any) -> Any:
        """Recursively detach tensors and move them to CPU for lightweight diagnostics."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: Model._move_to_cpu(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [Model._move_to_cpu(item) for item in value]
            return tuple(converted) if isinstance(value, tuple) else converted
        return value
    
    def _print_memory_debug(self, stage, extra_info=""):
        """Emit memory diagnostics to the dedicated memory logger (file), not stdout."""
        if not (self.enable_memory_debug or getattr(self, "enable_memory_diagnostics", False)):
            return
        import torch
        import gc
        import os
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                self.memory_logger.debug(
                    "MEMORY [%s] %s | GPU_Allocated=%.2fGB GPU_Reserved=%.2fGB GPU_MaxAllocated=%.2fGB",
                    stage,
                    extra_info,
                    allocated,
                    reserved,
                    max_allocated,
                )
            else:
                import psutil  # type: ignore[import-not-found]
                process = psutil.Process(os.getpid())
                cpu_gb = process.memory_info().rss / 1024**3
                self.memory_logger.debug(
                    "MEMORY [%s] %s | CPU=%.2fGB",
                    stage,
                    extra_info,
                    cpu_gb,
                )
        except Exception:
            # Best-effort; avoid raising from diagnostics
            self.memory_logger.debug("MEMORY [%s] %s | unavailable", stage, extra_info)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass of Celestial Enhanced PGAT
        
        Args:
            x_enc: [batch_size, seq_len, enc_in] Encoder input (114 waves or 13 celestial bodies)
            x_mark_enc: [batch_size, seq_len, mark_dim] Encoder time features
            x_dec: [batch_size, label_len + pred_len, dec_in] Decoder input
            x_mark_dec: [batch_size, label_len + pred_len, mark_dim] Decoder time features
            mask: Optional attention mask
            
        Returns:
            Predictions and metadata
        """
        self._print_memory_debug("FORWARD_START", f"Input shapes: x_enc={x_enc.shape}, x_dec={x_dec.shape}")
        self._debug_memory("FORWARD_START")
        batch_size, seq_len, enc_in = x_enc.shape
        self._log_debug("MODEL_DEBUG: Input shapes - x_enc=%s x_dec=%s", x_enc.shape, x_dec.shape)
        
        # 0. Enhanced Phase-Aware Wave Processing (if enabled)
        self._debug_memory("PHASE_PROCESSING_START")
        wave_metadata: Dict[str, Any] = {}
        phase_based_adj: Optional[torch.Tensor] = None
        target_shape = (batch_size, seq_len, len(self.target_wave_indices))
        if self.aggregate_waves_to_celestial and enc_in == self.num_input_waves:
            # Use phase-aware processor for rich celestial representations
            self._log_debug("MODEL_DEBUG: Starting phase-aware processing...")
            celestial_features, adjacency_matrix, phase_metadata = self.phase_aware_processor(x_enc)
            self._debug_memory("PHASE_PROCESSING_COMPLETE")
            # celestial_features: [batch, seq_len, 13 * 32] Rich multi-dimensional representations
            # adjacency_matrix: [batch, 13, 13] Phase-difference based edges
            # FIXED: Expand phase-based adjacency to 4D for consistency
            phase_based_adj = adjacency_matrix.unsqueeze(1).expand(-1, seq_len, -1, -1)
            self._log_debug(
                "MODEL_DEBUG: Phase processing complete - celestial_features=%s",
                celestial_features.shape,
            )

            # FIXED: Skip old processor to avoid dimension mismatch
            # The old data_processor has incorrect wave mappings that cause index errors
            target_metadata: Dict[str, Any] = {}
            target_waves = None  # Disable old processor diagnostics
            
            # TODO: Update old data_processor to use correct mappings if diagnostics needed
            if self.collect_diagnostics:
                self.logger.debug("Diagnostics disabled due to dimension mismatch in old processor")
                target_metadata['diagnostics_disabled'] = True

            # FIXED: Apply celestial projection to match d_model
            x_enc_processed = self.celestial_projection(celestial_features)  # [batch, seq_len, d_model]
            
            # Store comprehensive metadata
            if self.collect_diagnostics:
                wave_metadata = self._move_to_cpu({
                    'original_targets': target_waves,  # [batch, seq_len, 4]
                    'phase_metadata': phase_metadata,
                    'target_metadata': target_metadata,
                    'adjacency_matrix_stats': {
                        'mean': adjacency_matrix.mean().item(),
                        'std': adjacency_matrix.std().item(),
                    },
                    'celestial_features_shape': celestial_features.shape,
                })
            
            self._log_debug(
                "Phase-aware processing: %s → celestial %s, targets %s",
                x_enc.shape,
                celestial_features.shape,
                target_shape,
            )
            self._log_debug("Phase-based adjacency matrix: %s", adjacency_matrix.shape)
        else:
            x_enc_processed = x_enc
            wave_metadata['original_targets'] = None
        
        # 1. Input embeddings - FIXED: Ensure correct dimensions
        self._debug_memory("EMBEDDING_START")
        try:
            self._log_debug("MODEL_DEBUG: Starting encoder embedding...")
            self._log_debug("MODEL_DEBUG: x_enc_processed shape=%s, expected_dim=%s", 
                          x_enc_processed.shape, self.expected_embedding_input_dim)
            
            # Validate input dimensions
            if x_enc_processed.shape[-1] != self.expected_embedding_input_dim:
                self.logger.error(
                    "Dimension mismatch: x_enc_processed has %d features, but embedding expects %d",
                    x_enc_processed.shape[-1], self.expected_embedding_input_dim
                )
                raise ValueError(
                    f"Input dimension mismatch: got {x_enc_processed.shape[-1]}, "
                    f"expected {self.expected_embedding_input_dim}"
                )
            
            self._print_memory_debug("BEFORE_ENC_EMBEDDING")
            enc_out = self.enc_embedding(x_enc_processed, x_mark_enc)  # [batch, seq_len, d_model]
            self._print_memory_debug("AFTER_ENC_EMBEDDING", f"enc_out shape: {enc_out.shape}")
            
            # 📅 Apply Calendar Effects Enhancement
            if self.use_calendar_effects and self.calendar_effects_encoder is not None:
                self._log_debug("MODEL_DEBUG: Applying calendar effects...")
                # Extract date information from time marks (assuming first column is date-related)
                date_info = x_mark_enc[:, :, 0]  # [batch, seq_len] - first time feature as date proxy
                calendar_embeddings = self.calendar_effects_encoder(date_info)  # [batch, seq_len, calendar_dim]
                
                # Fuse calendar effects with encoder output
                import torch as torch_module  # Explicit import to avoid scoping issues
                combined_features = torch_module.cat([enc_out, calendar_embeddings], dim=-1)
                enc_out = self.calendar_fusion(combined_features)  # [batch, seq_len, d_model]
                self._log_debug("MODEL_DEBUG: Calendar effects applied - enc_out=%s", enc_out.shape)
            
            self._debug_memory("ENC_EMBEDDING_COMPLETE")
            self._log_debug("MODEL_DEBUG: Encoder embedding complete - enc_out=%s", enc_out.shape)
        except Exception as exc:
            self.logger.exception(
                "Encoder embedding failed | input_shape=%s expected_dim=%s embedding_c_in=%s",
                x_enc_processed.shape,
                self.expected_embedding_input_dim,
                self.enc_embedding.value_embedding.tokenConv.in_channels,
            )
            self._debug_memory("ENC_EMBEDDING_ERROR")
            raise exc

        self._log_debug("MODEL_DEBUG: Starting decoder embedding...")
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [batch, label_len+pred_len, d_model]
        
        # 📅 Apply Calendar Effects to Decoder
        if self.use_calendar_effects and self.calendar_effects_encoder is not None:
            self._log_debug("MODEL_DEBUG: Applying calendar effects to decoder...")
            # Extract date information from decoder time marks
            dec_date_info = x_mark_dec[:, :, 0]  # [batch, label_len+pred_len]
            dec_calendar_embeddings = self.calendar_effects_encoder(dec_date_info)
            
            # Fuse calendar effects with decoder output
            import torch as torch_module  # Explicit import to avoid scoping issues
            dec_combined_features = torch_module.cat([dec_out, dec_calendar_embeddings], dim=-1)
            dec_out = self.calendar_fusion(dec_combined_features)  # [batch, label_len+pred_len, d_model]
            self._log_debug("MODEL_DEBUG: Decoder calendar effects applied - dec_out=%s", dec_out.shape)
        
        self._debug_memory("DEC_EMBEDDING_COMPLETE")
        self._log_debug("MODEL_DEBUG: Decoder embedding complete - dec_out=%s", dec_out.shape)
        
        # 2. Generate market context from encoder output (fix information bottleneck)
        # FIXED: Use full temporal sequence instead of just last timestep for dynamic adjacency fusion
        self._print_memory_debug("BEFORE_MARKET_CONTEXT")
        market_context = self.market_context_encoder(enc_out)  # [batch, seq_len, d_model] - DYNAMIC!
        self._print_memory_debug("AFTER_MARKET_CONTEXT", f"market_context shape: {market_context.shape}")
        
        # 3. Enhanced Celestial Body Graph Processing
        if self.use_celestial_graph:
            if self.aggregate_waves_to_celestial and phase_based_adj is not None:
                # Use phase-based adjacency matrix from phase-aware processor
                
                # Still process celestial graph for additional features
                celestial_results = self._process_celestial_graph(x_enc_processed, enc_out)
                astronomical_adj = celestial_results['astronomical_adj']
                dynamic_adj = celestial_results['dynamic_adj'] 
                celestial_features = celestial_results['celestial_features']
                celestial_metadata = celestial_results['metadata']
                enc_out = celestial_results['enhanced_enc_out'] # Use enhanced output
                
                # Learned context-aware fusion of phase, astronomical, and dynamic adjacencies
                # FIXED: Dynamic fusion weights that adapt per timestep
                self._print_memory_debug("BEFORE_ADJ_WEIGHT_MLP", f"market_context: {market_context.shape}")
                weights = F.softmax(self.adj_weight_mlp(market_context), dim=-1)  # [batch, seq_len, 3] - DYNAMIC!
                self._print_memory_debug("AFTER_ADJ_WEIGHT_MLP", f"weights: {weights.shape}")
                w_phase, w_astro, w_dyn = weights[..., 0], weights[..., 1], weights[..., 2]  # [batch, seq_len] each
                
                # Normalize and combine
                self._print_memory_debug("BEFORE_ADJ_NORMALIZE")
                phase_norm = self._normalize_adj(phase_based_adj)
                astro_norm = self._normalize_adj(astronomical_adj) # Now dynamic
                dyn_norm = self._normalize_adj(dynamic_adj)       # Now dynamic
                self._print_memory_debug("AFTER_ADJ_NORMALIZE", f"phase_norm: {phase_norm.shape}, astro_norm: {astro_norm.shape}")
                
                # Unsqueeze weights for broadcasting with dynamic adjs [batch, seq_len, nodes, nodes]
                w_phase = w_phase.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, 1, 1]
                w_astro = w_astro.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, 1, 1]
                w_dyn = w_dyn.unsqueeze(-1).unsqueeze(-1)      # [batch, seq_len, 1, 1]
                
                # FIXED: phase_norm is already 4D, no need to expand
                self._print_memory_debug("BEFORE_ADJ_FUSION", f"Broadcasting {w_phase.shape} with {phase_norm.shape}")
                combined_phase_adj = (
                    w_phase * phase_norm +
                    w_astro * astro_norm +
                    w_dyn * dyn_norm
                )
                self._print_memory_debug("AFTER_ADJ_FUSION", f"combined_phase_adj: {combined_phase_adj.shape}")
                astronomical_adj = combined_phase_adj  # Use learned-fusion matrix
                self._log_debug("Using phase-based adjacency matrix with learned fusion weights")
            else:
                # Fallback to traditional celestial graph processing
                celestial_results = self._process_celestial_graph(x_enc_processed, enc_out)
                astronomical_adj = celestial_results['astronomical_adj']
                dynamic_adj = celestial_results['dynamic_adj'] 
                celestial_features = celestial_results['celestial_features']
                celestial_metadata = celestial_results['metadata']
                enc_out = celestial_results['enhanced_enc_out'] # Use enhanced output
        else:
            # Fallback to traditional graph learning
            traditional_adj = self._learn_traditional_graph(enc_out)
            astronomical_adj = traditional_adj
            dynamic_adj = self._learn_simple_dynamic_graph(enc_out)
            celestial_features = None
            celestial_metadata = {}
        
        # 4. Learn data-driven graph
        learned_adj = self._learn_data_driven_graph(enc_out)
        
        # 5. Combine all graph types with hierarchical fusion
        if self.use_celestial_graph:
            # FIXED: All adjacency matrices are now consistently 4D [batch, seq_len, nodes, nodes]
            # CelestialBodyNodes already returns 4D tensors - no manual broadcasting needed
            
            # Validate dimensions for safety and debugging
            self._validate_adjacency_dimensions(astronomical_adj, learned_adj, dynamic_adj, enc_out)

            # DEBUG: Memory check before celestial_combiner
            import torch
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
                self.memory_logger.debug(
                    "MEMORY BEFORE celestial_combiner | Allocated=%.2fGB Reserved=%.2fGB",
                    allocated_before,
                    reserved_before,
                )
            else:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                cpu_before = process.memory_info().rss / 1024**3
                self.memory_logger.debug("MEMORY BEFORE celestial_combiner | CPU=%.2fGB", cpu_before)
            
            # Debug input shapes
            self._log_debug(
                "INPUT to celestial_combiner: astronomical_adj=%s learned_adj=%s dynamic_adj=%s enc_out=%s",
                astronomical_adj.shape,
                learned_adj.shape,
                dynamic_adj.shape,
                enc_out.shape,
            )
            
            # 🚀 PETRI NET COMBINER: Preserves rich edge features!
            if self.use_petri_net_combiner:
                # Call with return_rich_features=True to get full edge vectors
                combined_adj, rich_edge_features, fusion_metadata = self.celestial_combiner(
                    astronomical_adj, learned_adj, dynamic_adj, enc_out,
                    return_rich_features=True  # Get [batch, seq, 13, 13, 6] edge features!
                )
                
                # Log edge feature preservation
                self._log_debug(
                    "PETRI NET: Rich edge features preserved! combined_adj=%s rich_edge_features=%s",
                    combined_adj.shape,
                    (rich_edge_features.shape if rich_edge_features is not None else None),
                )
                
                # Store rich features in metadata for analysis
                fusion_metadata['rich_edge_features_shape'] = rich_edge_features.shape
                fusion_metadata['edge_features_preserved'] = True
                fusion_metadata['no_compression'] = True
            else:
                # OLD COMBINER: Returns only scalar adjacency
                combined_adj, fusion_metadata = self.celestial_combiner(
                    astronomical_adj, learned_adj, dynamic_adj, enc_out
                )
                rich_edge_features = None  # No rich features in old combiner
                fusion_metadata['edge_features_preserved'] = False
            
            # DEBUG: Memory check after celestial_combiner
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                self.memory_logger.debug(
                    "MEMORY AFTER celestial_combiner | Allocated=%.2fGB Reserved=%.2fGB Delta_Allocated=%+.2fGB Delta_Reserved=%+.2fGB",
                    allocated_after,
                    reserved_after,
                    allocated_after - allocated_before,
                    reserved_after - reserved_before,
                )
            else:
                cpu_after = process.memory_info().rss / 1024**3
                self.memory_logger.debug(
                    "MEMORY AFTER celestial_combiner | CPU=%.2fGB Delta_CPU=%+.2fGB",
                    cpu_after,
                    cpu_after - cpu_before,
                )
            
            # Debug output shapes
            self._log_debug(
                "OUTPUT from celestial_combiner: combined_adj=%s fusion_metadata_keys=%s",
                combined_adj.shape,
                list(fusion_metadata.keys()) if fusion_metadata else None,
            )
            
            # DEBUG: Force garbage collection and check memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                post_gc_allocated = torch.cuda.memory_allocated() / 1024**3
                self.memory_logger.debug("MEMORY AFTER GC | Allocated=%.2fGB", post_gc_allocated)
            else:
                import psutil
                process = psutil.Process()
                post_gc_cpu = process.memory_info().rss / 1024**3
                self.memory_logger.debug("MEMORY AFTER GC | CPU=%.2fGB", post_gc_cpu)
        else:
            # Simple combination for fallback with dynamic weighting
            # Generate dynamic weights for each time step from enc_out
            weights = F.softmax(self.adj_weight_mlp(enc_out), dim=-1)  # [batch, seq_len, 3]
            w_astro, w_learned, w_dyn = weights[..., 0], weights[..., 1], weights[..., 2]

            # Normalize adjacency matrices (now all dynamic)
            astro_norm = self._normalize_adj(astronomical_adj)
            learned_norm = self._normalize_adj(learned_adj)
            dyn_norm = self._normalize_adj(dynamic_adj)
            
            # Unsqueeze weights to be broadcastable with [batch, seq_len, nodes, nodes]
            w_astro = w_astro.unsqueeze(-1).unsqueeze(-1)
            w_learned = w_learned.unsqueeze(-1).unsqueeze(-1)
            w_dyn = w_dyn.unsqueeze(-1).unsqueeze(-1)

            combined_adj = (
                w_astro * astro_norm +
                w_learned * learned_norm +
                w_dyn * dyn_norm
            )
            fusion_metadata = {
                'weights': weights.detach().cpu()
            }
        
        # 6. Apply hierarchical mapping if enabled
        self._log_debug("STEP 6: Hierarchical mapping check (enabled=%s)", self.use_hierarchical_mapping)
        if self.use_hierarchical_mapping:
            self._log_debug("Starting hierarchical mapper...")
            try:
                hierarchical_features = self.hierarchical_mapper(enc_out)  # [batch, num_nodes, d_model]
                self._log_debug("Hierarchical mapper completed: %s", hierarchical_features.shape)
                
                # Reshape and project to preserve spatial information
                batch_size, seq_len, _ = enc_out.shape
                reshaped_features = hierarchical_features.view(batch_size, -1)
                projected_features = self.hierarchical_projection(reshaped_features).unsqueeze(1)
                
                # Add to encoder output, broadcasting across sequence length
                enc_out = enc_out + projected_features
                
            except Exception as exc:
                self.logger.warning("Hierarchical mapping failed: %s", exc)
                self._log_debug("Continuing without hierarchical features")
        
        # 7. Spatiotemporal encoding with graph attention
        self._log_debug("STEP 7: Spatiotemporal encoding | enc_out=%s combined_adj=%s", enc_out.shape, combined_adj.shape)
        encoded_features: torch.Tensor

        # Fast path: when Petri net combiner is active, it already applies temporal and spatial attention
        # and Step 8 uses EdgeConditionedGraphAttention. The legacy spatiotemporal encoder may be redundant.
        if self.use_petri_net_combiner and self.bypass_spatiotemporal_with_petri:
            self._log_debug("Petri bypass enabled — using encoder output directly for graph processing")
            encoded_features = enc_out  # [batch, seq_len, d_model]
        else:
            use_dynamic_encoder = (
                self.use_dynamic_spatiotemporal_encoder
                and self.dynamic_spatiotemporal_encoder is not None
            )
            self._log_debug("use_dynamic_encoder=%s", use_dynamic_encoder)

            if use_dynamic_encoder:
                self._log_debug("Using DYNAMIC spatiotemporal encoder...")
                dynamic_encoder = self.dynamic_spatiotemporal_encoder
                if dynamic_encoder is None:
                    encoded_features = self._apply_static_spatiotemporal_encoding(enc_out, combined_adj)
                else:
                    try:
                        self._log_debug("Calling dynamic_encoder with enc_out=%s adj=%s", enc_out.shape, combined_adj.shape)
                        encoded_features = dynamic_encoder(enc_out, combined_adj)
                        self._log_debug("Dynamic encoder completed: %s", encoded_features.shape)
                    except ValueError as exc:
                        self.logger.warning("Dynamic spatiotemporal encoder fallback: %s", exc)
                        encoded_features = self._apply_static_spatiotemporal_encoding(enc_out, combined_adj)
            else:
                self._log_debug("Using STATIC spatiotemporal encoding...")
                encoded_features = self._apply_static_spatiotemporal_encoding(enc_out, combined_adj)
                self._log_debug("Static encoding completed: %s", encoded_features.shape)
        
        # 8. Graph attention processing
        self._log_debug(
            "STEP 8: Graph attention processing | encoded_features=%s use_efficient_covariate_interaction=%s",
            encoded_features.shape,
            self.use_efficient_covariate_interaction,
        )
        if self.use_efficient_covariate_interaction:
            # Efficient, partitioned graph processing that respects covariate independence
            graph_features = self._efficient_graph_processing(encoded_features, combined_adj)
            self._log_debug("Using efficient partitioned graph processing.")
        else:
            # Original, iterative graph attention processing
            graph_features = encoded_features
            
            # 🚀 ZERO-INFORMATION-LOSS: Use rich edge features if available!
            if self.use_petri_net_combiner and rich_edge_features is not None:
                self._log_debug(
                    "Using rich edge features in graph attention | rich_edge_features=%s",
                    rich_edge_features.shape,
                )
                
                # Process with edge-conditioned attention (no time loop needed - handles full sequence)
                graph_features = encoded_features
                for i, layer in enumerate(self.graph_attention_layers):
                    try:
                        # EdgeConditionedGraphAttention accepts full sequence with rich edge features
                        graph_features = layer(graph_features, edge_features=rich_edge_features)
                        self._log_debug("Layer %s: Used 6D edge features directly in attention!", i + 1)
                    except Exception as exc:
                        self.logger.warning(
                            "Edge-conditioned graph attention layer %s failed: %s",
                            i + 1,
                            exc,
                        )
                        # Fallback to adjacency-only
                        graph_features = layer(graph_features, adj_matrix=combined_adj)
                
                self._log_debug("Applied edge-conditioned graph attention with ZERO information loss!")
            else:
                # OLD PATH: Process each timestep with scalar adjacency
                self._log_debug("Using scalar adjacency (old method)")
                processed_features_over_time = []
                for t in range(self.seq_len):
                    time_step_features = graph_features[:, t:t+1, :]
                    adj_for_step = combined_adj[:, t, :, :]
                    
                    processed_step = time_step_features
                    for i, layer in enumerate(self.graph_attention_layers):
                        try:
                            processed_step = layer(processed_step, adj_for_step)
                        except Exception as exc:
                            self.logger.warning(
                                "Graph attention layer %s at step %s failed: %s",
                                i + 1,
                                t,
                                exc,
                            )
                            processed_step = layer(processed_step, None)
                    
                    processed_features_over_time.append(processed_step)
                
                graph_features = torch.cat(processed_features_over_time, dim=1)
                self._log_debug("Applied dynamic graph attention across %s time steps.", self.seq_len)
        
        # 9. Enhanced Decoder processing with Target Autocorrelation
        decoder_features = dec_out
        for layer in self.decoder_layers:
            decoder_features = layer(decoder_features, graph_features)
        
        # 🎯 Apply Target Autocorrelation Module if enabled
        if self.use_target_autocorrelation and self.dual_stream_decoder is not None:
            self._log_debug("Applying target autocorrelation processing...")
            decoder_features = self.dual_stream_decoder(decoder_features, graph_features)
            self._log_debug("Target autocorrelation processing complete - shape=%s", decoder_features.shape)
        
        # 10. Final prediction with MDN or fallback decoders
        output: torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        predictions: Optional[torch.Tensor] = None
        mdn_components: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        aux_loss: float = 0.0

        # Priority: MDN decoder (Phase 1) > Sequential mixture > Simple projection
        if self.enable_mdn_decoder and self.mdn_decoder is not None:
            # 🎲 MDN Decoder: probabilistic forecasting with Gaussian mixtures
            prediction_features = decoder_features[:, -self.pred_len:, :]  # [batch, pred_len, d_model]
            pi, mu, sigma = self.mdn_decoder(prediction_features)  # [B, pred_len, c_out, K]
            
            # Compute point prediction as mixture mean for logging/metrics
            predictions = self.mdn_decoder.mean_prediction(pi, mu)  # [batch, pred_len, c_out]
            
            # Store mixture components for loss computation
            mdn_components = (pi, mu, sigma)
            
            self._log_debug(
                "MDN decoder output | pi=%s mu=%s sigma=%s predictions=%s",
                pi.shape, mu.shape, sigma.shape, predictions.shape,
            )
        elif self.use_mixture_decoder:
            # Sequential mixture density decoder preserves temporal structure
            try:
                # Extract the prediction portion of decoder features
                prediction_features = decoder_features[:, -self.pred_len:, :]  # [batch, pred_len, d_model]
                
                # Use sequential mixture decoder with cross-attention to encoder features
                means, log_stds, log_weights = self.mixture_decoder(
                    encoder_output=graph_features,  # Full encoder output for cross-attention
                    decoder_input=prediction_features  # Only prediction portion
                )
                
                # Get point prediction using the decoder's method
                predictions = self.mixture_decoder.get_point_prediction((means, log_stds, log_weights))
                
                # Ensure output has correct shape [batch, pred_len, c_out]
                if predictions.size(-1) != self.c_out:
                    self.logger.warning(
                        "Sequential mixture decoder output shape mismatch | got=%s expected_c_out=%s",
                        predictions.shape,
                        self.c_out,
                    )
                    predictions = self.projection(prediction_features)
                
                mdn_components = (means, log_stds, log_weights)
                if predictions is not None:
                    self._log_debug(
                        "Sequential mixture decoder output | means=%s predictions=%s",
                        means.shape,
                        predictions.shape,
                    )
                
            except Exception as exc:
                self.logger.exception("Sequential mixture decoder failed: %s", exc)
                # Fallback to simple projection
                predictions = self.projection(decoder_features[:, -self.pred_len:, :])
        else:
            # Ensure consistent slicing for deterministic case
            predictions = self.projection(decoder_features[:, -self.pred_len:, :])
        
        # Prepare metadata for return
        final_metadata: Optional[Dict[str, Any]]
        if self.collect_diagnostics:
            final_metadata = {
                **wave_metadata,
                'celestial_metadata': self._move_to_cpu(celestial_metadata) if self.use_celestial_graph else {},
                'fusion_metadata': self._move_to_cpu(fusion_metadata) if self.use_celestial_graph else {},
            }
        else:
            final_metadata = None
        
        # Return format aligns with _normalize_model_output in training script:
        # When MDN enabled: (point_pred, aux_loss, (pi, mu, sigma), metadata)
        # Otherwise: (predictions, metadata) or just predictions
        if mdn_components is not None:
            # Return tuple compatible with _normalize_model_output
            if predictions is not None:
                # MDN case: (point_pred, aux_loss, mdn_tuple, metadata)
                return (predictions, aux_loss, mdn_components, final_metadata)
            else:
                # Fallback if point prediction failed
                return (mdn_components, final_metadata)
        elif predictions is not None:
            return (predictions, final_metadata)
        else:
            output = self.projection(decoder_features[:, -self.pred_len:, :])
            return (output, final_metadata)
    
    def get_regularization_loss(self):
        """Get the regularization loss from the stochastic graph learner."""
        loss = 0.0
        if self.use_stochastic_learner and hasattr(self, 'latest_stochastic_loss'):
            loss += self.latest_stochastic_loss
        return loss

    def get_point_prediction(self, forward_output):
        """Extracts a single point prediction from the model's output, handling the probabilistic case."""
        if self.use_mixture_decoder and isinstance(forward_output, tuple):
            means, _, log_weights = forward_output
            with torch.no_grad():
                if means.dim() == 4:  # [batch, pred_len, num_targets, num_components]
                    # Calculate weighted average of component means
                    weights = torch.softmax(log_weights, dim=-1).unsqueeze(2).expand_as(means)
                    predictions = (means * weights).sum(dim=-1)
                else:  # Univariate case
                    weights = torch.softmax(log_weights, dim=-1)
                    predictions = (means * weights).sum(dim=-1, keepdim=True)
                
                # Fallback to ensure correct output shape if something goes wrong
                if predictions.size(-1) != self.c_out:
                    return means[..., 0] if means.dim() == 4 else means[..., 0].unsqueeze(-1)
                return predictions
        return forward_output  # Output is already a point prediction
    
    def _process_celestial_graph(self, x_enc: torch.Tensor, enc_out: torch.Tensor) -> Dict[str, Any]:
        """Process encoder features through the dynamic celestial body graph system."""
        astronomical_adj, dynamic_adj, celestial_features, metadata = self.celestial_nodes(enc_out)
        
        # The celestial features are dynamic: [batch, seq_len, num_bodies, d_model]
        batch_size, seq_len, num_bodies, hidden_dim = celestial_features.shape
        if self.celestial_fusion_attention is None or self.celestial_fusion_gate is None:
            raise RuntimeError("Celestial fusion modules must be initialized before processing the graph.")

        projected_query = self.celestial_query_projection(enc_out)
        projected_keys = self.celestial_key_projection(celestial_features)
        projected_values = self.celestial_value_projection(celestial_features)

        query = projected_query.reshape(batch_size * seq_len, 1, self.celestial_fusion_dim)
        key = projected_keys.reshape(batch_size * seq_len, num_bodies, self.celestial_fusion_dim)
        value = projected_values.reshape(batch_size * seq_len, num_bodies, self.celestial_fusion_dim)

        fused_output, attention_weights = self.celestial_fusion_attention(query, key, value)
        fused_output = fused_output.reshape(batch_size, seq_len, self.celestial_fusion_dim)
        fused_output = self.celestial_output_projection(fused_output)
        gate_input = torch.cat([enc_out, fused_output], dim=-1)
        fusion_gate = self.celestial_fusion_gate(gate_input)
        celestial_influence = fusion_gate * fused_output
        enhanced_enc_out = enc_out + celestial_influence
        if self.collect_diagnostics:
            metadata['celestial_attention_weights'] = (
                attention_weights.reshape(batch_size, seq_len, num_bodies).detach().cpu()
            )
        else:
            metadata = {}
        
        return {
            'astronomical_adj': astronomical_adj,  # [batch, seq_len, num_nodes, num_nodes]
            'dynamic_adj': dynamic_adj,            # [batch, seq_len, num_nodes, num_nodes]
            'celestial_features': celestial_features,  # [batch, seq_len, num_nodes, d_model]
            'enhanced_enc_out': enhanced_enc_out,  # [batch, seq_len, d_model]
            'metadata': metadata,
        }

    def _apply_static_spatiotemporal_encoding(
        self,
        enc_out: torch.Tensor,
        combined_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback path that uses the legacy joint encoder with best effort adjacencies."""
        try:
            # FIXED: combined_adj is always 4D, use last timestep for static encoder
            adj_for_encoder = combined_adj[:, -1, :, :]

            if self.use_celestial_graph and self.aggregate_waves_to_celestial:
                num_nodes = self.num_celestial_bodies
                if adj_for_encoder.size(0) != num_nodes:
                    adj_for_encoder = torch.eye(num_nodes, device=enc_out.device, dtype=torch.float32)
                return self.spatiotemporal_encoder(enc_out, adj_for_encoder)

            batch_size, seq_len, d_model = enc_out.shape
            num_nodes = self.enc_in
            if adj_for_encoder.size(0) != num_nodes:
                adj_for_encoder = torch.eye(num_nodes, device=enc_out.device, dtype=torch.float32)

            if d_model % num_nodes == 0 and num_nodes > 0:
                node_dim = d_model // num_nodes
                enc_out_4d = enc_out.view(batch_size, seq_len, num_nodes, node_dim)
                encoded_features_4d = self.spatiotemporal_encoder(enc_out_4d, adj_for_encoder)
                return encoded_features_4d.view(batch_size, seq_len, d_model)

            return self.spatiotemporal_encoder(enc_out, adj_for_encoder)

        except Exception as error:  # pylint: disable=broad-except
            self.logger.warning("Spatiotemporal encoding failed: %s", error)
            return enc_out
    
    def _learn_traditional_graph(self, enc_out):
        """Learn a dynamic traditional graph adjacency for each time step."""
        batch_size, seq_len, d_model = enc_out.shape
        
        # Reshape to apply the learner to each time step independently
        enc_out_flat = enc_out.reshape(batch_size * seq_len, d_model)
        adj_flat = self.traditional_graph_learner(enc_out_flat)
        
        # Determine the number of nodes
        num_nodes = self.num_celestial_bodies if (self.use_celestial_graph and self.aggregate_waves_to_celestial) else self.enc_in
        
        # Reshape back to a time-varying adjacency matrix
        adj_matrix = adj_flat.view(batch_size, seq_len, num_nodes, num_nodes)
        return adj_matrix
    
    def _learn_simple_dynamic_graph(self, enc_out):
        """Compute a simple distinct dynamic identity adjacency for each time step."""
        batch_size, seq_len, _ = enc_out.shape
        device = enc_out.device
        num_nodes = self.num_celestial_bodies if (self.use_celestial_graph and self.aggregate_waves_to_celestial) else self.enc_in
        identity = torch.eye(num_nodes, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return identity.expand(batch_size, seq_len, num_nodes, num_nodes)
    
    def _learn_data_driven_graph(self, enc_out):
        """Learn a dynamic data-driven graph from encoded features for each time step."""
        batch_size, seq_len, d_model = enc_out.shape
        
        # Reshape to apply the learner to each time step
        enc_out_flat = enc_out.reshape(batch_size * seq_len, d_model)
        
        if self.use_stochastic_learner:
            # Stochastic graph learning for each time step
            mean = self.stochastic_mean(enc_out_flat)
            logvar = self.stochastic_logvar(enc_out_flat)
            
            if self.training:
                # Sample during training
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                adj_flat = mean + eps * std
                
                # Calculate KL divergence for each element in the batch*seq_len dimension
                kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
                
                # Reshape to [batch, seq_len] to correctly average over the sequence
                kl_div_per_step = kl_div.view(batch_size, seq_len)
                
                # Average across sequence length and then batch to get a single scalar loss
                self.latest_stochastic_loss = kl_div_per_step.mean()
            else:
                # Use mean during inference
                adj_flat = mean
                self.latest_stochastic_loss = 0.0
        else:
            # Deterministic graph learning
            adj_flat = self.traditional_graph_learner(enc_out_flat)
            self.latest_stochastic_loss = 0.0
        
        # Determine the number of nodes
        num_nodes = self.num_celestial_bodies if self.use_celestial_graph else self.enc_in
        
        # Reshape to a time-varying adjacency matrix
        adj_matrix = adj_flat.view(batch_size, seq_len, num_nodes, num_nodes)
        
        return adj_matrix
    
    def _validate_adjacency_dimensions(self, astronomical_adj: torch.Tensor, 
                                     learned_adj: torch.Tensor, 
                                     dynamic_adj: torch.Tensor, 
                                     enc_out: torch.Tensor) -> None:
        """Validate that all adjacency matrices have consistent 4D dimensions."""
        batch_size, seq_len, d_model = enc_out.shape
        expected_shape = (batch_size, seq_len, self.num_celestial_bodies, self.num_celestial_bodies)
        
        matrices = {
            'astronomical_adj': astronomical_adj,
            'learned_adj': learned_adj, 
            'dynamic_adj': dynamic_adj
        }
        
        for name, matrix in matrices.items():
            if matrix.shape != expected_shape:
                self.logger.error(
                    "Adjacency matrix dimension mismatch: %s has shape %s, expected %s",
                    name, matrix.shape, expected_shape
                )
                raise ValueError(
                    f"Adjacency matrix {name} has incorrect dimensions: "
                    f"got {matrix.shape}, expected {expected_shape}"
                )
        
        self._log_debug("✅ All adjacency matrices validated: %s", expected_shape)
    
    def _normalize_adj(self, adj_matrix):
        """Normalize adjacency matrix for stable fusion. Handles 4D dynamic matrices only."""
        # FIXED: All adjacency matrices are now 4D [batch, seq_len, nodes, nodes]
        
        # Add self-loops for stability
        identity = torch.eye(adj_matrix.size(-1), device=adj_matrix.device, dtype=torch.float32)
        identity = identity.unsqueeze(0).unsqueeze(0)  # Expand to [1, 1, nodes, nodes]
        
        adj_with_self_loops = adj_matrix + identity.expand_as(adj_matrix)
        
        # Row normalization (convert to stochastic matrix)
        row_sums = adj_with_self_loops.sum(dim=-1, keepdim=True)
        normalized = adj_with_self_loops / (row_sums + 1e-8)
        
        return normalized


    def _efficient_graph_processing(self, encoded_features, combined_adj):
        """
        Performs partitioned graph processing.
        1. Computes a context vector from the independent covariate graph.
        2. Updates target node features based on their interactions and influence from the covariate context.
        This is much more memory-efficient than processing the full graph iteratively.
        """
        batch_size, seq_len, _ = encoded_features.shape
        num_nodes = self.num_graph_nodes
        
        # This architecture assumes d_model is divisible by the number of nodes.
        if self.d_model % num_nodes != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_graph_nodes ({num_nodes}) for efficient processing.")
        node_dim = self.d_model // num_nodes
        
        features_per_node = encoded_features.view(batch_size, seq_len, num_nodes, node_dim)
        
        # Partition features and adjacency matrix
        num_covariates = self.num_graph_nodes - self.c_out
        covariate_features = features_per_node[:, :, :num_covariates, :]
        target_features = features_per_node[:, :, num_covariates:, :]
        
        adj_cov_cov = combined_adj[:, :, :num_covariates, :num_covariates]
        adj_tar_cov = combined_adj[:, :, num_covariates:, :num_covariates]
        adj_tar_tar = combined_adj[:, :, num_covariates:, num_covariates:]
        
        # 1. Process Covariates to get a context summary
        # Perform graph convolution: A * X
        cov_interaction = torch.einsum('bsnm,bsmd->bsnd', adj_cov_cov, covariate_features)
        # Apply shared weight matrix: (A * X) * W
        processed_covariates = self.covariate_interaction_layer(cov_interaction)
        # Aggregate across covariate nodes to get a single context vector per timestep
        covariate_context = processed_covariates.mean(dim=2)  # Shape: [batch, seq_len, node_dim]
        
        # 2. Update Target Features
        # Influence from covariates on targets: A_tc * C
        influence_from_cov = torch.einsum('bsnm,bsmd->bsnd', adj_tar_cov, covariate_features)
        # Self-influence of targets: A_tt * T
        influence_from_self = torch.einsum('bsnm,bsmd->bsnd', adj_tar_tar, target_features)
        
        # The new features for targets are a combination of their old state and influences
        updated_target_features = target_features + influence_from_cov + influence_from_self
        
        # 3. Fuse with global covariate context
        # Expand context to be concatenated with each target node's features
        expanded_context = covariate_context.unsqueeze(2).expand(-1, -1, self.c_out, -1)
        
        # Reshape for fusion layer
        fusion_input = torch.cat([updated_target_features, expanded_context], dim=-1)
        
        fused_target_features = self.target_fusion_layer(fusion_input)
        
        # 4. Reconstruct the full feature tensor
        # We use the original, unchanged covariate features and the updated target features
        final_features_per_node = torch.cat([covariate_features, fused_target_features], dim=2)
        
        # Reshape back to the expected [batch, seq_len, d_model]
        return final_features_per_node.view(batch_size, seq_len, self.d_model)


class DecoderLayer(nn.Module):
    """Decoder layer with cross-attention to encoder features"""
    
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, dec_input, enc_output):
        """
        Args:
            dec_input: [batch, dec_len, d_model] Decoder input
            enc_output: [batch, enc_len, d_model] Encoder output
        """
        # Self-attention
        self_attn_out, _ = self.self_attention(dec_input, dec_input, dec_input)
        dec_input = self.norm1(dec_input + self.dropout(self_attn_out))
        
        # Cross-attention
        cross_attn_out, _ = self.cross_attention(dec_input, enc_output, enc_output)
        dec_input = self.norm2(dec_input + self.dropout(cross_attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(dec_input)
        dec_input = self.norm3(dec_input + self.dropout(ff_out))
        
        return dec_input


class DataEmbedding(nn.Module):
    """Data embedding with positional and temporal encodings"""
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        # Enforce fixed input dimension for stability
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.c_in:
            raise ValueError(
                f"TokenEmbedding input_dim ({input_dim}) does not match expected c_in ({self.c_in})."
                f" Ensure encoder/embedding configuration is consistent."
            )
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe.requires_grad = False
        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super().__init__()
        
        minute_size = 4
        hour_size = 25
        weekday_size = 8
        day_size = 32
        month_size = 13
        
        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        # Handle variable number of time features gracefully
        num_features = x.size(-1)
        
        # Initialize embeddings to zero
        minute_x = 0.
        hour_x = 0.
        weekday_x = 0.
        day_x = 0.
        month_x = 0.
        
        # Apply embeddings based on available features
        # Convert continuous temporal features to discrete indices
        if num_features > 0:
            month_indices = torch.clamp((x[:, :, 0] * 12).long(), 0, 11)
            month_x = self.month_embed(month_indices)
        if num_features > 1:
            day_indices = torch.clamp((x[:, :, 1] * 31).long(), 0, 30)
            day_x = self.day_embed(day_indices)
        if num_features > 2:
            weekday_indices = torch.clamp((x[:, :, 2] * 7).long(), 0, 6)
            weekday_x = self.weekday_embed(weekday_indices)
        if num_features > 3:
            hour_indices = torch.clamp((x[:, :, 3] * 24).long(), 0, 23)
            hour_x = self.hour_embed(hour_indices)
        if num_features > 4 and hasattr(self, 'minute_embed'):
            minute_indices = torch.clamp((x[:, :, 4] * 60).long(), 0, 59)
            minute_x = self.minute_embed(minute_indices)
        
        return hour_x + weekday_x + day_x + month_x + minute_x