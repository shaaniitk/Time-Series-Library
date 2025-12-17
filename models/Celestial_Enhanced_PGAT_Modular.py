# models/Celestial_Enhanced_PGAT_Modular.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

# Import the new modular components
from .celestial_modules.config import CelestialPGATConfig
from .celestial_modules.embedding import EmbeddingModule, EmbeddingError, DimensionMismatchError, CelestialProcessingError
from .celestial_modules.graph import GraphModule
from .celestial_modules.encoder import EncoderModule
from .celestial_modules.postprocessing import PostProcessingModule
from .celestial_modules.decoder import DecoderModule

# Note: These imports are optional and may not exist yet
try:
    from .celestial_modules.context_fusion import ContextFusionFactory, MultiScaleContextFusion
except ImportError:
    ContextFusionFactory = None
    MultiScaleContextFusion = None

try:
    from .celestial_modules.utils import ModelUtils
except ImportError:
    ModelUtils = None

try:
    from .celestial_modules.diagnostics import ModelDiagnostics
except ImportError:
    ModelDiagnostics = None

class ModularModelError(Exception):
    """Base exception for modular model errors"""
    pass

class ConfigurationError(ModularModelError):
    """Raised when model configuration is invalid"""
    pass

class ProcessingError(ModularModelError):
    """Raised when a processing stage fails"""
    pass

class Model(nn.Module):
    """
    Celestial Enhanced PGAT - Modular Version
    
    This model orchestrates the data flow through a series of specialized, modular components,
    improving readability and maintainability without changing core functionality.
    
    Key Enhancement: Implements a Parallel Context Stream for long-term awareness without
    sacrificing the temporal precision required for celestial graph dynamics.
    """
    def __init__(self, configs):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 1. Centralized Configuration
        self.model_config = CelestialPGATConfig.from_original_configs(configs)
        
        # Validate configuration early
        self.validate_configuration(self.model_config)
        
        # 2. Initialize utility and diagnostics systems (if available)
        if ModelUtils is not None:
            self.utils = ModelUtils(self.model_config, self.logger)
            self.utils.log_configuration_summary()
        else:
            self.utils = None
            self.logger.info("ModelUtils not available - using basic functionality")
            
        if ModelDiagnostics is not None:
            self.diagnostics = ModelDiagnostics(self.model_config, self.logger)
        else:
            self.diagnostics = None
            self.logger.info("ModelDiagnostics not available - diagnostics disabled")

        # 3. Instantiate all modules
        self.logger.info("Initializing EmbeddingModule...")
        self.embedding_module = EmbeddingModule(self.model_config)
        self.logger.info("EmbeddingModule initialized.")
        
        # Multi-Scale Context Fusion Module (if available)
        if ContextFusionFactory is not None:
            try:
                ContextFusionFactory.validate_config(self.model_config)
                self.context_fusion = ContextFusionFactory.create_context_fusion(self.model_config)
            except Exception as e:
                self.logger.warning(f"Context fusion initialization failed: {e}")
                self.context_fusion = None
        else:
            self.context_fusion = None
        
        if self.model_config.use_celestial_graph:
            self.logger.info("Initializing GraphModule...")
            self.graph_module = GraphModule(self.model_config)
            self.logger.info("GraphModule initialized.")
        else:
            self.graph_module = None
            self.logger.info("GraphModule disabled.")
        
        self.logger.info("Initializing EncoderModule...")
        self.encoder_module = EncoderModule(self.model_config)
        self.logger.info("EncoderModule initialized.")
        
        self.logger.info("Initializing PostProcessingModule...")
        self.postprocessing_module = PostProcessingModule(self.model_config)
        self.logger.info("PostProcessingModule initialized.")
        
        self.logger.info("Initializing DecoderModule...")
        self.decoder_module = DecoderModule(self.model_config)
        self.logger.info("DecoderModule initialized.")

        # Enhanced components from original model
        self.logger.info("Initializing market_context_encoder...")
        self.market_context_encoder = nn.Sequential(
            nn.Linear(self.model_config.d_model, self.model_config.d_model),
            nn.GELU(),
            nn.Linear(self.model_config.d_model, self.model_config.d_model),
            nn.LayerNorm(self.model_config.d_model)
        )
        self.logger.info("market_context_encoder initialized.")
        
        # FIX ISSUE #10: Stochastic warmup for MDN stability
        # Disable stochastic noise for first N epochs to let MDN calibrate
        self.stochastic_warmup_epochs = int(getattr(configs, 'stochastic_warmup_epochs', 3))
        self.current_epoch = 0  # Updated externally by training script
        self.logger.info(
            "🎲 Stochastic warmup: %d epochs (noise disabled during warmup)",
            self.stochastic_warmup_epochs
        )
        
        # Efficient covariate interaction (if enabled)
        if self.model_config.use_efficient_covariate_interaction:
            self.logger.info("Setting up efficient covariate interaction...")
            self._setup_efficient_covariate_interaction()
            self.logger.info("Efficient covariate interaction setup.")
        
        self._external_global_step = None
        
        # Initialize parameters
        self.logger.info("Initializing parameters...")
        self._initialize_parameters()
        self.logger.info("Parameters initialized. __init__ complete.")
    
    def _initialize_parameters(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def set_current_epoch(self, epoch: int) -> None:
        """Update current epoch for scheduling (called by training script).
        
        FIX ISSUE #10: Enables stochastic warmup - stochastic noise is disabled
        for the first stochastic_warmup_epochs to allow MDN to stabilize.
        
        Args:
            epoch: Current training epoch (0-indexed)
        """
        self.current_epoch = epoch
        
        # Log warmup status transitions
        if epoch == self.stochastic_warmup_epochs:
            self.logger.info(
                "✅ Stochastic warmup complete (epoch %d) - enabling stochastic noise",
                epoch
            )
        elif epoch < self.stochastic_warmup_epochs:
            self.logger.debug(
                "⏳ Stochastic warmup: epoch %d/%d (noise disabled)",
                epoch, self.stochastic_warmup_epochs
            )
        
        # Propagate stochastic mode to sub-modules
        stochastic_enabled = (epoch >= self.stochastic_warmup_epochs)
        
        if self.graph_module is not None and hasattr(self.graph_module, 'set_stochastic_mode'):
            self.graph_module.set_stochastic_mode(stochastic_enabled)
            
        if self.decoder_module is not None and hasattr(self.decoder_module, 'set_stochastic_mode'):
            self.decoder_module.set_stochastic_mode(stochastic_enabled)
    
    def _setup_efficient_covariate_interaction(self):
        """Setup efficient covariate interaction components."""
        if self.model_config.d_model % self.model_config.num_graph_nodes != 0:
            raise ValueError(
                f"d_model ({self.model_config.d_model}) must be divisible by "
                f"num_graph_nodes ({self.model_config.num_graph_nodes}) for efficient processing."
            )
        
        node_dim = self.model_config.d_model // self.model_config.num_graph_nodes
        
        # FEATURE 1: Learned projections for semantic validity
        # Map global d_model embedding to node-specific features
        self.projection_to_nodes = nn.Linear(self.model_config.d_model, self.model_config.num_graph_nodes * node_dim)
        # Map node-specific features back to global d_model
        self.projection_from_nodes = nn.Linear(self.model_config.num_graph_nodes * node_dim, self.model_config.d_model)
        
        self.covariate_interaction_layer = nn.Linear(node_dim, node_dim)
        self.target_fusion_layer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim)
        )
        
        if self.model_config.enable_target_covariate_attention:
            attn_heads = self.model_config.n_heads if (node_dim % self.model_config.n_heads) == 0 else 1
            self.covariate_attention = nn.MultiheadAttention(
                embed_dim=node_dim,
                num_heads=attn_heads,
                dropout=self.model_config.dropout,
                batch_first=True,
            )
        else:
            self.covariate_attention = None

    @staticmethod
    def validate_configuration(config):
        """
        Validate configuration for dimension compatibility issues BEFORE model initialization.
        Call this early in training scripts to catch configuration errors with helpful messages.
        
        Args:
            config: Configuration object with model parameters
            
        Raises:
            ValueError: With detailed suggestions if configuration is invalid
        """
        d_model = getattr(config, 'd_model', 512)
        n_heads = getattr(config, 'n_heads', 8)
        use_efficient_covariate = getattr(config, 'use_efficient_covariate_interaction', False)
        use_celestial_graph = getattr(config, 'use_celestial_graph', True)
        aggregate_waves = getattr(config, 'aggregate_waves_to_celestial', True)
        num_celestial_bodies = getattr(config, 'num_celestial_bodies', 13)
        
        errors = []
        suggestions = []
        
        # Check d_model and n_heads compatibility
        if d_model % n_heads != 0:
            closest_d_model = ((d_model // n_heads) + 1) * n_heads
            errors.append(f"d_model ({d_model}) not divisible by n_heads ({n_heads})")
            suggestions.append(f"d_model: {closest_d_model}  # Auto-adjusted for attention compatibility")
        
        # Check efficient covariate interaction requirements
        if use_efficient_covariate and use_celestial_graph and aggregate_waves:
            num_graph_nodes = num_celestial_bodies
            if d_model % num_graph_nodes != 0:
                # Find LCM for perfect compatibility
                import math
                lcm = (num_graph_nodes * n_heads) // math.gcd(num_graph_nodes, n_heads)
                perfect_d_model = ((d_model // lcm) + 1) * lcm
                simple_d_model = ((d_model // num_graph_nodes) + 1) * num_graph_nodes
                
                errors.append(f"d_model ({d_model}) not divisible by num_graph_nodes ({num_graph_nodes}) for efficient covariate interaction")
                suggestions.extend([
                    f"d_model: {perfect_d_model}  # Perfect compatibility (graph nodes + attention heads)",
                    f"d_model: {simple_d_model}  # Simple compatibility (graph nodes only)",
                    "# OR disable: use_efficient_covariate_interaction: false"
                ])
        
        # Check celestial_dim compatibility if specified
        celestial_dim = getattr(config, 'celestial_dim', None)
        if celestial_dim and celestial_dim % n_heads != 0:
            closest_celestial_dim = ((celestial_dim // n_heads) + 1) * n_heads
            errors.append(f"celestial_dim ({celestial_dim}) not divisible by n_heads ({n_heads})")
            suggestions.append(f"celestial_dim: {closest_celestial_dim}  # Attention head compatibility")
        
        if errors:
            error_msg = f"""
🚫 CONFIGURATION VALIDATION FAILED

❌ Issues found:
{chr(10).join(f"   • {error}" for error in errors)}

✅ RECOMMENDED FIXES:
{chr(10).join(f"   {suggestion}" for suggestion in suggestions)}

💡 Add these to your YAML config file to resolve all dimension compatibility issues.

🔧 QUICK COPY-PASTE:
# Dimension fixes
{chr(10).join(suggestion for suggestion in suggestions if not suggestion.startswith('#'))}
"""
            raise ValueError(error_msg)
        
        return True  # All validations passed

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, 
                future_celestial_x=None, future_celestial_mark=None):
        """
        Forward pass with modular architecture and comprehensive functionality.
        
        Args:
            x_enc: [batch_size, seq_len, enc_in] Encoder input
            x_mark_enc: [batch_size, seq_len, mark_dim] Encoder time features
            x_dec: [batch_size, label_len + pred_len, dec_in] Decoder input
            x_mark_dec: [batch_size, label_len + pred_len, mark_dim] Decoder time features
            mask: Optional attention mask
            future_celestial_x: [batch_size, pred_len, enc_in] Future celestial states (deterministic)
            future_celestial_mark: [batch_size, pred_len, mark_dim] Future time features
            
        Returns:
            Predictions and metadata
        """
        batch_size, seq_len, enc_in = x_enc.shape
        
        # --- Stage 1: Embedding & Phase-Aware Processing ---
        enc_out, dec_out, past_celestial_features, phase_based_adj, x_enc_processed, wave_metadata = self.embedding_module(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        
        # --- Stage 2: Enhanced Context Fusion ---
        context_diagnostics = {}
        
        if self.context_fusion is not None:
            try:
                enc_out_with_context, context_diagnostics = self.context_fusion(enc_out)
            except Exception as e:
                raise RuntimeError(
                    f"Context fusion failed: {e}. "
                    f"Check context fusion configuration and input dimensions. "
                    f"Input shape: {enc_out.shape}"
                ) from e
        else:
            # When context fusion is disabled, use simple parallel context stream
            context_vector = torch.mean(enc_out, dim=1, keepdim=True)
            enc_out_with_context = enc_out + context_vector
            context_diagnostics = {
                'mode': 'simple_parallel_context',
                'context_norm': torch.norm(context_vector).item()
            }
        
        # Generate market context from the enhanced encoder output
        market_context = self.market_context_encoder(enc_out_with_context)
        
        # --- Stage 3: Graph Generation & Fusion ---
        enhanced_enc_out = enc_out_with_context
        combined_adj, rich_edge_features = None, None
        fusion_metadata = {}
        celestial_features = None
        
        if self.graph_module is not None:
            try:
                enhanced_enc_out, combined_adj, rich_edge_features, fusion_metadata, celestial_features = self.graph_module(
                    enc_out_with_context, market_context, phase_based_adj
                )
                
                # Store celestial features for downstream processing
                if celestial_features is not None:
                    wave_metadata['celestial_features'] = celestial_features
                    
            except Exception as e:
                raise RuntimeError(
                    f"Graph module processing failed: {e}. "
                    f"Check celestial graph configuration and input dimensions. "
                    f"Input shape: {enc_out_with_context.shape}, "
                    f"Market context shape: {market_context.shape}"
                ) from e
        else:
            # When celestial graph is disabled, we must have alternative graph processing
            if not self.model_config.use_celestial_graph:
                # Use identity adjacency for non-graph processing
                num_nodes = self.model_config.num_graph_nodes
                device = enc_out.device
                identity = torch.eye(num_nodes, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                combined_adj = identity.expand(batch_size, seq_len, num_nodes, num_nodes)
                rich_edge_features = None
                fusion_metadata = {'mode': 'identity_adjacency'}
                celestial_features = None
            else:
                raise RuntimeError(
                    "Celestial graph is enabled in configuration but graph_module is None. "
                    "This indicates an initialization error. Check model configuration."
                )

        # --- Stage 4: Core Encoding ---
        try:
            if self.model_config.use_efficient_covariate_interaction:
                graph_features = self._efficient_graph_processing(enhanced_enc_out, combined_adj)
            else:
                graph_features = self.encoder_module(enhanced_enc_out, combined_adj, rich_edge_features)
        except Exception as e:
            raise RuntimeError(
                f"Encoder module processing failed: {e}. "
                f"Check encoder configuration and input dimensions. "
                f"Enhanced encoder output shape: {enhanced_enc_out.shape}, "
                f"Combined adjacency shape: {combined_adj.shape if combined_adj is not None else None}"
            ) from e

        # --- Stage 5: Optional Post-Processing ---
        try:
            graph_features = self.postprocessing_module(
                graph_features, self._external_global_step
            )
        except Exception as e:
            raise RuntimeError(
                f"Post-processing module failed: {e}. "
                f"Check post-processing configuration. "
                f"Graph features shape: {graph_features.shape}"
            ) from e

        # --- Stage 6: Decoding & Prediction ---
        future_celestial_features = None
        if future_celestial_x is not None:
            if not self.embedding_module.phase_aware_processor:
                raise RuntimeError(
                    "Future celestial data provided but phase_aware_processor is not available. "
                    "Either enable celestial processing or remove future celestial data from input."
                )
            
            try:
                # Process future deterministic covariates
                future_cel_feats, future_adj, future_phase_meta = self.embedding_module.phase_aware_processor(future_celestial_x)
                
                # Reshape for attention
                future_celestial_features = future_cel_feats.reshape(
                    batch_size, self.model_config.pred_len, self.model_config.num_celestial_bodies, -1
                )
            except Exception as e:
                raise RuntimeError(
                    f"Future celestial processing failed: {e}. "
                    f"Check future celestial data dimensions and processor configuration. "
                    f"Future celestial input shape: {future_celestial_x.shape}"
                ) from e

        try:
            predictions, aux_loss, mdn_components = self.decoder_module(
                dec_out, graph_features, past_celestial_features, future_celestial_features
            )
        except Exception as e:
            raise RuntimeError(
                f"Decoder module processing failed: {e}. "
                f"Check decoder configuration and input dimensions. "
                f"Decoder input shape: {dec_out.shape}, "
                f"Graph features shape: {graph_features.shape}"
            ) from e
        
        # Prepare comprehensive metadata for diagnostics
        if self.diagnostics is not None:
            try:
                final_metadata = self.diagnostics.prepare_final_metadata(
                    wave_metadata=wave_metadata,
                    celestial_metadata=self.diagnostics.collect_celestial_metadata({'metadata': {}}),
                    fusion_metadata=self.diagnostics.collect_fusion_metadata(fusion_metadata),
                    context_fusion_diagnostics=context_diagnostics,
                    enhanced_enc_out_norm=torch.norm(enhanced_enc_out).item(),
                    graph_features_norm=torch.norm(graph_features).item(),
                )
            except Exception as e:
                self.logger.warning(f"Diagnostics metadata preparation failed: {e}")
                final_metadata = {
                    'wave_metadata': wave_metadata,
                    'fusion_metadata': fusion_metadata,
                    'context_diagnostics': context_diagnostics,
                    'enhanced_enc_out_norm': torch.norm(enhanced_enc_out).item(),
                    'graph_features_norm': torch.norm(graph_features).item(),
                }
        else:
            # Basic metadata when diagnostics not available
            final_metadata = {
                'wave_metadata': wave_metadata,
                'fusion_metadata': fusion_metadata,
                'context_diagnostics': context_diagnostics,
                'enhanced_enc_out_norm': torch.norm(enhanced_enc_out).item(),
                'graph_features_norm': torch.norm(graph_features).item(),
            }
        
        # Return format aligns with original model - STRICT validation
        if predictions is None:
            raise RuntimeError(
                "Decoder module returned None predictions. "
                "This indicates a critical failure in the decoder processing. "
                "Check decoder configuration and ensure all required components are properly initialized."
            )
        
        # Validate prediction dimensions
        expected_pred_shape = (batch_size, self.model_config.pred_len, self.model_config.c_out)
        if predictions.shape != expected_pred_shape:
            raise RuntimeError(
                f"Prediction shape mismatch: expected {expected_pred_shape}, got {predictions.shape}. "
                f"Check decoder output projection and configuration."
            )
        
        # CRITICAL FIX: Return only what Exp_Long_Term_Forecast expects (pred, aux, mdn)
        # Verify if final_metadata breaks the training loop.
        # Returning tuple of length 3 or 2 depending on MDN.
        
        if mdn_components is not None:
            return (predictions, aux_loss, mdn_components)
        else:
            return (predictions, aux_loss)

    def _efficient_graph_processing(self, encoded_features, combined_adj):
        """
        Performs partitioned graph processing.
        1. Projects global embedding to node-specific features (Learned Projection).
        2. Computes a context vector from the independent covariate graph.
        3. Updates target node features based on their interactions and influence from the covariate context.
        4. Projects back to global embedding.
        """
        batch_size, seq_len, _ = encoded_features.shape
        num_nodes = self.model_config.num_graph_nodes
        
        if self.model_config.d_model % num_nodes != 0:
            raise ValueError(f"d_model ({self.model_config.d_model}) must be divisible by num_graph_nodes ({num_nodes}) for efficient processing.")
        node_dim = self.model_config.d_model // num_nodes
        
        # Learned projection instead of view
        features_per_node = self.projection_to_nodes(encoded_features).view(batch_size, seq_len, num_nodes, node_dim)
        
        # Partition features and adjacency matrix
        num_covariates = num_nodes - self.model_config.c_out
        covariate_features = features_per_node[:, :, :num_covariates, :]
        target_features = features_per_node[:, :, num_covariates:, :]
        
        adj_cov_cov = combined_adj[:, :, :num_covariates, :num_covariates]
        adj_tar_cov = combined_adj[:, :, num_covariates:, :num_covariates]
        adj_tar_tar = combined_adj[:, :, num_covariates:, num_covariates:]
        
        # 1. Process Covariates
        cov_interaction = torch.einsum('bsnm,bsmd->bsnd', adj_cov_cov, covariate_features)
        processed_covariates = self.covariate_interaction_layer(cov_interaction)
        
        if not self.model_config.enable_target_covariate_attention or self.covariate_attention is None:
            covariate_context = processed_covariates.mean(dim=2)
        
        # 2. Update Target Features
        influence_from_cov = torch.einsum('bsnm,bsmd->bsnd', adj_tar_cov, covariate_features)
        influence_from_self = torch.einsum('bsnm,bsmd->bsnd', adj_tar_tar, target_features)
        updated_target_features = target_features + influence_from_cov + influence_from_self
        
        # 3. Fuse targets with covariate context
        if self.model_config.enable_target_covariate_attention and self.covariate_attention is not None:
            query = updated_target_features.reshape(batch_size * seq_len, self.model_config.c_out, node_dim)
            key = processed_covariates.reshape(batch_size * seq_len, num_covariates, node_dim)
            value = key
            attn_output, _ = self.covariate_attention(query, key, value)
            attention_context = attn_output.reshape(batch_size, seq_len, self.model_config.c_out, node_dim)
            fusion_input = torch.cat([updated_target_features, attention_context], dim=-1)
        else:
            expanded_context = covariate_context.unsqueeze(2).expand(-1, -1, self.model_config.c_out, -1)
            fusion_input = torch.cat([updated_target_features, expanded_context], dim=-1)
        
        fused_target_features = self.target_fusion_layer(fusion_input)
        
        # 4. Reconstruct the full feature tensor
        final_features_per_node = torch.cat([covariate_features, fused_target_features], dim=2)
        
        # Project back
        return self.projection_from_nodes(final_features_per_node.view(batch_size, seq_len, -1))

    def _learn_traditional_graph(self, enc_out):
        """Learn a traditional graph adjacency matrix - NOT IMPLEMENTED."""
        raise NotImplementedError(
            "Traditional graph learning is not implemented in the modular architecture. "
            "Either enable celestial graph processing (use_celestial_graph=True) "
            "or implement a proper graph learning module."
        )

    def set_global_step(self, step: int):
        """Set global step for stochastic control in post-processing."""
        self._external_global_step = step
        
    def get_model_config(self) -> CelestialPGATConfig:
        """Get the centralized model configuration."""
        return self.model_config
        
    def get_regularization_loss(self) -> torch.Tensor:
        """Get regularization loss from stochastic components if enabled."""
        loss = 0.0
        if (self.graph_module is not None and 
            hasattr(self.graph_module, 'latest_stochastic_loss')):
            loss += self.graph_module.latest_stochastic_loss
        return torch.tensor(loss, device=next(self.parameters()).device)
    
    def get_point_prediction(self, forward_output):
        """Extracts a single point prediction from the model's output, handling the probabilistic case."""
        if self.utils is not None:
            return self.utils.get_point_prediction(forward_output)
        else:
            # Basic implementation when utils not available
            if isinstance(forward_output, tuple):
                return forward_output[0]  # Return predictions
            return forward_output
    
    def print_fusion_diagnostics_summary(self):
        """Print summary of fusion diagnostics."""
        if self.diagnostics is not None:
            self.diagnostics.print_fusion_diagnostics_summary()
        else:
            self.logger.info("Diagnostics not available")
    
    def increment_fusion_diagnostics_batch(self):
        """Increment the batch counter for fusion diagnostics."""
        if self.diagnostics is not None:
            self.diagnostics.increment_fusion_diagnostics_batch()
    
    def print_celestial_target_diagnostics(self):
        """Print celestial-to-target attention diagnostics."""
        if (self.model_config.use_celestial_target_attention and 
            hasattr(self.decoder_module, 'celestial_to_target_attention') and
            self.decoder_module.celestial_to_target_attention is not None):
            if hasattr(self.decoder_module.celestial_to_target_attention, 'print_diagnostics_summary'):
                self.decoder_module.celestial_to_target_attention.print_diagnostics_summary()
            else:
                self.logger.info("Celestial-to-target attention diagnostics not available")
        else:
            self.logger.info("Celestial-to-target attention not enabled or not initialized")
    
    def print_context_fusion_diagnostics(self):
        """Print multi-scale context fusion diagnostics."""
        if self.context_fusion is not None and hasattr(self.context_fusion, 'get_diagnostics_summary'):
            summary = self.context_fusion.get_diagnostics_summary()
            self.logger.info(summary)
        else:
            self.logger.info("Multi-scale context fusion not enabled or diagnostics not available")
    
    def get_context_fusion_mode(self) -> str:
        """Get the current context fusion mode."""
        if self.context_fusion is not None and hasattr(self.model_config, 'context_fusion_mode'):
            return self.model_config.context_fusion_mode
        return "disabled"
    
    def set_context_fusion_diagnostics(self, enabled: bool):
        """Enable or disable context fusion diagnostics."""
        if self.context_fusion is not None and hasattr(self.context_fusion, 'enable_diagnostics'):
            self.context_fusion.enable_diagnostics = enabled
            if hasattr(self.model_config, 'enable_context_diagnostics'):
                self.model_config.enable_context_diagnostics = enabled

    def compute_loss(self, outputs, targets, criterion, curriculum_mask=None, logger=None):
        """
        Unpack model outputs and calculate the total loss.
        Handles:
        - Output unpacking (Tensor vs Tuple)
        - MDN vs Standard Loss (using checks on criterion or output structure)
        - Auxiliary Loss (from unpacking)
        - Curriculum Masking (slicing inputs/targets logic)
        """
        predictions = outputs
        aux_loss = 0.0
        mdn_components = None

        # 1. Unpack outputs generically
        if isinstance(outputs, (tuple, list)):
            if len(outputs) >= 3:
                # Assuming (pred, aux, mdn) signature based on recent implementation
                # But be careful about strictness.
                predictions = outputs[0]
                # Try to extract aux
                if isinstance(outputs[1], (int, float, torch.Tensor)):
                     # Basic heuristic: 2nd element is aux loss
                     if isinstance(outputs[1], torch.Tensor) and outputs[1].numel() == 1:
                         aux_loss = outputs[1].item()
                     elif isinstance(outputs[1], (int, float)):
                         aux_loss = outputs[1]
                
                # Try to extract MDN
                if isinstance(outputs[2], tuple):
                    mdn_components = outputs[2]
            
            elif len(outputs) == 2:
                # (pred, aux) or (pred, mdn)?
                predictions = outputs[0]
                if isinstance(outputs[1], tuple):
                    mdn_components = outputs[1] # MDN tuple
                elif isinstance(outputs[1], (int, float)) or (isinstance(outputs[1], torch.Tensor) and outputs[1].numel() == 1):
                    aux_loss = outputs[1] if isinstance(outputs[1], (int, float)) else outputs[1].item()

        # 2. Check for MDN Loss requirement
        # If mdn_components is present OR criterion is MixtureNLLLoss, we use MDN path
        from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss
        use_mdn_loss = isinstance(criterion, MixtureNLLLoss) or (mdn_components is not None)

        loss_val = 0.0
        
        # 3. Apply Curriculum Masking (Slicing) logic
        # Slice predictions/targets/components to effective length
        eff_p = predictions
        eff_t = targets
        eff_mdn = mdn_components
        
        if curriculum_mask is not None:
            eff_len = int(curriculum_mask.sum().item())
            # Ensure at least 1 step
            eff_len = max(1, eff_len) 
            
            # Slice Predictions: [Batch, Len, Dim]
            eff_p = predictions[:, :eff_len, :]
            
            # Slice Targets (handle dimension mismatch if any)
            # targets usually [Batch, Len, Dim] or [Batch, Len, 1]
            eff_t = targets[:, :eff_len, ...]

            # Slice MDN components if present
            if mdn_components is not None:
                means, log_stds, log_weights = mdn_components
                # Usually [Batch, Len, Guassians, Dim] or similar. Time is dim 1.
                # Check dim 1 size
                if means.size(1) >= eff_len:
                    means = means[:, :eff_len, ...]
                    log_stds = log_stds[:, :eff_len, ...]
                    log_weights = log_weights[:, :eff_len, ...]
                    eff_mdn = (means, log_stds, log_weights)
        else:
            # If no curriculum, still might need to crop MDN to pred_len if output is longer?
            # Or assume they match. Standardize on matching.
            if use_mdn_loss and mdn_components is not None:
                means, _, _ = mdn_components
                pred_len = predictions.size(1)
                if means.size(1) > pred_len:
                    # Crop MDN extra steps if any
                     means, log_stds, log_weights = mdn_components
                     eff_mdn = (means[:, :pred_len, ...], log_stds[:, :pred_len, ...], log_weights[:, :pred_len, ...])

        # 4. Compute Main Loss
        if use_mdn_loss:
            if eff_mdn is None:
                # Fallback to MSE if MDN expected but components missing (e.g. failure case)
                if logger: logger.warning("MDN components missing for MixtureNLLLoss. Falling back to simple criterion on point predictions.")
                loss_val = F.mse_loss(eff_p, eff_t) # Basic fallback
            else:
                # Reshape targets for MDN if needed (squeeze last dim if 1)
                if eff_t.dim() == 3 and eff_t.size(-1) == 1:
                    target_for_mdn = eff_t.squeeze(-1)
                else:
                    target_for_mdn = eff_t
                
                loss_val = criterion(eff_mdn, target_for_mdn)
        else:
            # Deterministic Loss
            # ensure targets and preds match shapes?
            # Typically criterion handles it, but let's be safe
             loss_val = criterion(eff_p, eff_t)

        return loss_val + aux_loss