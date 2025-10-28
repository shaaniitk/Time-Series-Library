# models/Celestial_Enhanced_PGAT_Modular.py

import logging
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

# Import the new modular components
from .celestial_modules.config import CelestialPGATConfig
from .celestial_modules.embedding import EmbeddingModule
from .celestial_modules.graph import GraphModule
from .celestial_modules.encoder import EncoderModule
from .celestial_modules.postprocessing import PostProcessingModule
from .celestial_modules.decoder import DecoderModule

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

        # 2. Instantiate all modules
        self.embedding_module = EmbeddingModule(self.model_config)
        
        if self.model_config.use_celestial_graph:
            self.graph_module = GraphModule(self.model_config)
        else:
            self.graph_module = None
        
        self.encoder_module = EncoderModule(self.model_config)
        self.postprocessing_module = PostProcessingModule(self.model_config)
        self.decoder_module = DecoderModule(self.model_config)

        # Other components from original model
        self.market_context_encoder = nn.Sequential(
            nn.Linear(self.model_config.d_model, self.model_config.d_model), nn.GELU(),
            nn.Linear(self.model_config.d_model, self.model_config.d_model), nn.LayerNorm(self.model_config.d_model)
        )
        self._external_global_step = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, 
                future_celestial_x=None, future_celestial_mark=None):
        """
        Forward pass with modular architecture and parallel context stream.
        
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
        
        # --- Stage 1: Embedding & Phase-Aware Processing ---
        enc_out, dec_out, past_celestial_features, phase_based_adj, x_enc_processed = self.embedding_module(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        
        # --- Stage 2: NEW Parallel Context Stream ---
        # Create a low-resolution summary of the entire sequence
        context_vector = torch.mean(enc_out, dim=1, keepdim=True)  # Shape: [B, 1, D]
        # Fuse the context into the high-resolution stream by adding it to each time step
        enc_out_with_context = enc_out + context_vector
        
        self.logger.debug(
            "Parallel Context Stream applied: context_vector=%s, enhanced_enc_out=%s",
            context_vector.shape, enc_out_with_context.shape
        )
        
        # --- Stage 3: Graph Generation & Fusion ---
        market_context = self.market_context_encoder(enc_out_with_context)
        enhanced_enc_out = enc_out_with_context
        combined_adj, rich_edge_features = None, None
        
        if self.graph_module is not None:
            enhanced_enc_out, combined_adj, rich_edge_features = self.graph_module(
                enc_out_with_context, market_context, phase_based_adj
            )

        # --- Stage 4: Core Encoding ---
        graph_features = self.encoder_module(
            enhanced_enc_out, combined_adj, rich_edge_features
        )

        # --- Stage 5: Optional Post-Processing ---
        graph_features = self.postprocessing_module(
            graph_features, self._external_global_step
        )

        # --- Stage 6: Decoding & Prediction ---
        future_celestial_features = None
        if future_celestial_x is not None and self.embedding_module.phase_aware_processor:
            # Process future deterministic covariates
            future_celestial_features, _, _ = self.embedding_module.phase_aware_processor(future_celestial_x)

        predictions, aux_loss, mdn_components = self.decoder_module(
            dec_out, graph_features, past_celestial_features, future_celestial_features
        )
        
        # The metadata dictionary can be reconstructed here if needed for diagnostics
        final_metadata = {
            'context_vector_norm': torch.norm(context_vector).item(),
            'enhanced_enc_out_norm': torch.norm(enhanced_enc_out).item(),
            'graph_features_norm': torch.norm(graph_features).item(),
        }
        
        if mdn_components is not None:
            return (predictions, aux_loss, mdn_components, final_metadata)
        else:
            return (predictions, final_metadata)

    def set_global_step(self, step: int):
        """Set global step for stochastic control in post-processing."""
        self._external_global_step = step
        
    def get_model_config(self) -> CelestialPGATConfig:
        """Get the centralized model configuration."""
        return self.model_config
        
    def get_regularization_loss(self) -> torch.Tensor:
        """Get regularization loss from stochastic components if enabled."""
        if hasattr(self.graph_module, 'config') and self.graph_module.config.use_stochastic_learner:
            # Implement KL divergence regularization for stochastic graph learner
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.tensor(0.0, device=next(self.parameters()).device)