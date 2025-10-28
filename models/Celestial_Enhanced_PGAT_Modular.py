# models/Celestial_Enhanced_PGAT_Modular.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

# Import the new modular components
from .celestial_modules.config import CelestialPGATConfig
from .celestial_modules.embedding import EmbeddingModule
from .celestial_modules.graph import GraphModule
from .celestial_modules.encoder import EncoderModule
from .celestial_modules.postprocessing import PostProcessingModule
from .celestial_modules.decoder import DecoderModule
from .celestial_modules.context_fusion import ContextFusionFactory, MultiScaleContextFusion
from .celestial_modules.utils import ModelUtils
from .celestial_modules.diagnostics import ModelDiagnostics

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
        
        # Validate context fusion configuration
        ContextFusionFactory.validate_config(self.model_config)
        
        # 2. Initialize utility and diagnostics systems
        self.utils = ModelUtils(self.model_config, self.logger)
        self.diagnostics = ModelDiagnostics(self.model_config, self.logger)
        
        # Log configuration summary
        self.utils.log_configuration_summary()

        # 3. Instantiate all modules
        self.embedding_module = EmbeddingModule(self.model_config)
        
        # Multi-Scale Context Fusion Module
        self.context_fusion = ContextFusionFactory.create_context_fusion(self.model_config)
        
        if self.model_config.use_celestial_graph:
            self.graph_module = GraphModule(self.model_config)
        else:
            self.graph_module = None
        
        self.encoder_module = EncoderModule(self.model_config)
        self.postprocessing_module = PostProcessingModule(self.model_config)
        self.decoder_module = DecoderModule(self.model_config)

        # Enhanced components from original model
        self.market_context_encoder = nn.Sequential(
            nn.Linear(self.model_config.d_model, self.model_config.d_model),
            nn.GELU(),
            nn.Linear(self.model_config.d_model, self.model_config.d_model),
            nn.LayerNorm(self.model_config.d_model)
        )
        
        # Efficient covariate interaction (if enabled)
        if self.model_config.use_efficient_covariate_interaction:
            self._setup_efficient_covariate_interaction()
        
        self._external_global_step = None
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _setup_efficient_covariate_interaction(self):
        """Setup efficient covariate interaction components."""
        if self.model_config.d_model % self.model_config.num_graph_nodes != 0:
            raise ValueError(
                f"d_model ({self.model_config.d_model}) must be divisible by "
                f"num_graph_nodes ({self.model_config.num_graph_nodes}) for efficient processing."
            )
        
        node_dim = self.model_config.d_model // self.model_config.num_graph_nodes
        
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
            enc_out_with_context, context_diagnostics = self.context_fusion(enc_out)
        else:
            # Fallback to simple context fusion
            context_vector = torch.mean(enc_out, dim=1, keepdim=True)
            enc_out_with_context = enc_out + context_vector
        
        # Generate market context from the enhanced encoder output
        market_context = self.market_context_encoder(enc_out_with_context)
        
        # --- Stage 3: Graph Generation & Fusion ---
        enhanced_enc_out = enc_out_with_context
        combined_adj, rich_edge_features = None, None
        fusion_metadata = {}
        celestial_features = None
        
        if self.graph_module is not None:
            enhanced_enc_out, combined_adj, rich_edge_features, fusion_metadata, celestial_features = self.graph_module(
                enc_out_with_context, market_context, phase_based_adj
            )
            
            # Store celestial features for downstream processing
            if celestial_features is not None:
                wave_metadata['celestial_features'] = celestial_features
        else:
            # Fallback to traditional graph learning
            if hasattr(self, 'traditional_graph_learner'):
                combined_adj = self._learn_traditional_graph(enc_out)
            else:
                # Simple identity adjacency
                num_nodes = self.model_config.num_graph_nodes
                device = enc_out.device
                identity = torch.eye(num_nodes, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                combined_adj = identity.expand(batch_size, seq_len, num_nodes, num_nodes)

        # --- Stage 4: Core Encoding ---
        if self.model_config.use_efficient_covariate_interaction:
            graph_features = self._efficient_graph_processing(enhanced_enc_out, combined_adj)
        else:
            graph_features = self.encoder_module(enhanced_enc_out, combined_adj, rich_edge_features)

        # --- Stage 5: Optional Post-Processing ---
        graph_features = self.postprocessing_module(
            graph_features, self._external_global_step
        )

        # --- Stage 6: Decoding & Prediction ---
        future_celestial_features = None
        if future_celestial_x is not None and self.embedding_module.phase_aware_processor:
            # Process future deterministic covariates
            future_cel_feats, future_adj, future_phase_meta = self.embedding_module.phase_aware_processor(future_celestial_x)
            
            # Reshape for attention
            future_celestial_features = future_cel_feats.reshape(
                batch_size, self.model_config.pred_len, self.model_config.num_celestial_bodies, -1
            )

        predictions, aux_loss, mdn_components = self.decoder_module(
            dec_out, graph_features, past_celestial_features, future_celestial_features
        )
        
        # Prepare comprehensive metadata for diagnostics
        final_metadata = self.diagnostics.prepare_final_metadata(
            wave_metadata=wave_metadata,
            celestial_metadata=self.diagnostics.collect_celestial_metadata({'metadata': {}}),
            fusion_metadata=self.diagnostics.collect_fusion_metadata(fusion_metadata),
            context_fusion_diagnostics=context_diagnostics,
            enhanced_enc_out_norm=torch.norm(enhanced_enc_out).item(),
            graph_features_norm=torch.norm(graph_features).item(),
        )
        
        # Return format aligns with original model
        if mdn_components is not None:
            return (predictions, aux_loss, mdn_components, final_metadata)
        elif predictions is not None:
            return (predictions, final_metadata)
        else:
            # Fallback projection
            output = nn.Linear(self.model_config.d_model, self.model_config.c_out).to(graph_features.device)(
                graph_features[:, -self.model_config.pred_len:, :]
            )
            return (output, final_metadata)

    def _efficient_graph_processing(self, encoded_features, combined_adj):
        """
        Performs partitioned graph processing.
        1. Computes a context vector from the independent covariate graph.
        2. Updates target node features based on their interactions and influence from the covariate context.
        """
        batch_size, seq_len, _ = encoded_features.shape
        num_nodes = self.model_config.num_graph_nodes
        
        if self.model_config.d_model % num_nodes != 0:
            raise ValueError(f"d_model ({self.model_config.d_model}) must be divisible by num_graph_nodes ({num_nodes}) for efficient processing.")
        node_dim = self.model_config.d_model // num_nodes
        
        features_per_node = encoded_features.view(batch_size, seq_len, num_nodes, node_dim)
        
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
        return final_features_per_node.view(batch_size, seq_len, self.model_config.d_model)

    def _learn_traditional_graph(self, enc_out):
        """Learn a traditional graph adjacency matrix."""
        batch_size, seq_len, d_model = enc_out.shape
        num_nodes = self.model_config.num_graph_nodes
        
        # Simple identity matrix as fallback
        device = enc_out.device
        identity = torch.eye(num_nodes, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return identity.expand(batch_size, seq_len, num_nodes, num_nodes)

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
        return self.utils.get_point_prediction(forward_output)
    
    def print_fusion_diagnostics_summary(self):
        """Print summary of fusion diagnostics."""
        self.diagnostics.print_fusion_diagnostics_summary()
    
    def increment_fusion_diagnostics_batch(self):
        """Increment the batch counter for fusion diagnostics."""
        self.diagnostics.increment_fusion_diagnostics_batch()
    
    def print_celestial_target_diagnostics(self):
        """Print celestial-to-target attention diagnostics."""
        if (self.model_config.use_celestial_target_attention and 
            self.decoder_module.celestial_to_target_attention is not None):
            self.decoder_module.celestial_to_target_attention.print_diagnostics_summary()
        else:
            self.logger.info("Celestial-to-target attention not enabled or not initialized")
    
    def print_context_fusion_diagnostics(self):
        """Print multi-scale context fusion diagnostics."""
        if self.context_fusion is not None:
            summary = self.context_fusion.get_diagnostics_summary()
            self.logger.info(summary)
        else:
            self.logger.info("Multi-scale context fusion not enabled")
    
    def get_context_fusion_mode(self) -> str:
        """Get the current context fusion mode."""
        return self.model_config.context_fusion_mode if self.context_fusion is not None else "disabled"
    
    def set_context_fusion_diagnostics(self, enabled: bool):
        """Enable or disable context fusion diagnostics."""
        if self.context_fusion is not None:
            self.context_fusion.enable_diagnostics = enabled
            self.model_config.enable_context_diagnostics = enabled