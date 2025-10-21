"""
Memory-Optimized Celestial Enhanced PGAT

This version addresses the critical >64GB memory issues by:
1. Using static adjacency matrices instead of dynamic time-varying ones
2. Implementing gradient checkpointing for memory-intensive operations
3. Reducing tensor dimensions and simplifying complex operations
4. Adding explicit memory management and cleanup
5. Using the memory-optimized celestial processor
"""

import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers.modular.graph.celestial_body_nodes import CelestialBodyNodes, CelestialBody
from layers.modular.graph.celestial_graph_combiner import CelestialGraphCombiner
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder
from layers.modular.embedding.hierarchical_mapper import HierarchicalTemporalSpatialMapper
from layers.modular.encoder.spatiotemporal_encoding import JointSpatioTemporalEncoding
from layers.modular.graph.gated_graph_combiner import GatedGraphCombiner
from layers.Embed import DataEmbedding
from utils.celestial_wave_aggregator import CelestialWaveAggregator, CelestialDataProcessor
from layers.modular.aggregation.memory_optimized_celestial_processor import MemoryOptimizedCelestialProcessor


class Model(nn.Module):
    """
    Memory-Optimized Celestial Enhanced PGAT
    
    Solves the >64GB memory issue while maintaining astrological AI capabilities.
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.verbose_logging = bool(getattr(configs, "verbose_logging", False))
        
        # Core parameters
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len  
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = getattr(configs, 'd_model', 104)  # Reduced from 130
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 3)  # Reduced from 4
        self.d_layers = getattr(configs, 'd_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # Memory optimization settings
        self.use_static_adjacency = getattr(configs, 'use_static_adjacency', True)
        self.use_gradient_checkpointing = getattr(configs, 'use_gradient_checkpointing', True)
        self.celestial_dim = getattr(configs, 'celestial_dim', 16)  # Reduced from 32
        
        # Celestial system parameters
        self.use_celestial_graph = getattr(configs, 'use_celestial_graph', True)
        self.celestial_fusion_layers = getattr(configs, 'celestial_fusion_layers', 2)  # Reduced
        self.num_celestial_bodies = len(CelestialBody)
        
        # Simplified features for memory optimization
        self.use_mixture_decoder = False  # Disabled for memory
        self.use_stochastic_learner = False  # Disabled for memory
        self.use_hierarchical_mapping = False  # Disabled for memory
        self.collect_diagnostics = False  # Disabled for memory
        
        # Wave aggregation settings
        self.aggregate_waves_to_celestial = getattr(configs, 'aggregate_waves_to_celestial', True)
        self.num_input_waves = getattr(configs, 'num_input_waves', 118)
        self.target_wave_indices = getattr(configs, 'target_wave_indices', [0, 1, 2, 3])
        
        # Auto-adjust d_model for compatibility
        if self.use_celestial_graph and self.aggregate_waves_to_celestial:
            num_nodes_for_adjustment = self.num_celestial_bodies
        else:
            num_nodes_for_adjustment = self.enc_in
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return abs(a * b) // gcd(a, b) if a != 0 and b != 0 else 0

        required_multiple = lcm(num_nodes_for_adjustment, self.n_heads)
        
        if self.d_model % required_multiple != 0:
            original_d_model = self.d_model
            self.d_model = (original_d_model + required_multiple - 1) // required_multiple * required_multiple
            self.logger.warning(
                "d_model adjusted from %s to %s for compatibility (nodes=%s heads=%s)",
                original_d_model, self.d_model, num_nodes_for_adjustment, self.n_heads
            )
        
        self._log_configuration_summary()
        
        # Memory-optimized wave aggregation
        if self.aggregate_waves_to_celestial:
            self.memory_optimized_processor = MemoryOptimizedCelestialProcessor(
                num_input_waves=self.num_input_waves,
                celestial_dim=self.celestial_dim,
                waves_per_body=9,
                num_heads=self.n_heads,
                use_gradient_checkpointing=self.use_gradient_checkpointing
            )
            
            # Simplified projection
            self.celestial_to_model = nn.Linear(
                self.num_celestial_bodies * self.celestial_dim, 
                self.d_model
            )
            
            enc_embedding_dim = self.d_model  # Direct to d_model
        else:
            enc_embedding_dim = self.enc_in
        
        # Input embeddings
        self.enc_embedding = DataEmbedding(
            enc_embedding_dim, self.d_model, configs.embed, configs.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in, self.d_model, configs.embed, configs.freq, self.dropout
        )
        
        # Simplified celestial system for memory optimization
        if self.use_celestial_graph:
            # Use static adjacency matrices
            self.static_celestial_adj = nn.Parameter(
                torch.randn(self.num_celestial_bodies, self.num_celestial_bodies) * 0.1
            )
            
            # Simplified fusion
            self.celestial_fusion = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
                nn.Sigmoid()
            )
        
        # Simplified spatiotemporal encoding
        self.spatiotemporal_encoder = JointSpatioTemporalEncoding(
            d_model=self.d_model,
            seq_len=self.seq_len,
            num_nodes=self.num_celestial_bodies if self.aggregate_waves_to_celestial else self.enc_in,
            num_heads=self.n_heads,
            dropout=self.dropout
        )
        
        # Memory-optimized graph attention layers
        from layers.modular.graph.adjacency_aware_attention import AdjacencyAwareGraphAttention
        self.graph_attention_layers = nn.ModuleList([
            AdjacencyAwareGraphAttention(
                d_model=self.d_model, 
                d_ff=self.d_model * 2,  # Reduced from 4x to 2x
                n_heads=self.n_heads, 
                dropout=self.dropout,
                use_adjacency_mask=True
            ) for _ in range(self.e_layers)
        ])
        
        # Simplified decoder layers
        self.decoder_layers = nn.ModuleList([
            SimpleDecoderLayer(self.d_model, self.n_heads, self.dropout)
            for _ in range(self.d_layers)
        ])
        
        # Simple output projection (no mixture decoder)
        self.projection = nn.Linear(self.d_model, self.c_out)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _log_configuration_summary(self) -> None:
        """Log configuration summary."""
        if self.verbose_logging:
            self.logger.info(
                "Memory-Optimized Celestial Enhanced PGAT | seq_len=%s pred_len=%s d_model=%s "
                "celestial_dim=%s static_adjacency=%s gradient_checkpointing=%s",
                self.seq_len, self.pred_len, self.d_model, self.celestial_dim,
                self.use_static_adjacency, self.use_gradient_checkpointing
            )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Memory-optimized forward pass.
        
        Args:
            x_enc: [batch_size, seq_len, enc_in] Encoder input
            x_mark_enc: [batch_size, seq_len, mark_dim] Encoder time features
            x_dec: [batch_size, label_len + pred_len, dec_in] Decoder input
            x_mark_dec: [batch_size, label_len + pred_len, mark_dim] Decoder time features
            mask: Optional attention mask
            
        Returns:
            Predictions tensor
        """
        batch_size, seq_len, enc_in = x_enc.shape
        
        # Memory-optimized celestial processing
        if self.aggregate_waves_to_celestial and enc_in == self.num_input_waves:
            if self.use_gradient_checkpointing and self.training:
                celestial_features, static_adjacency, _ = checkpoint(
                    self.memory_optimized_processor, x_enc
                )
            else:
                celestial_features, static_adjacency, _ = self.memory_optimized_processor(x_enc)
            
            # Project to d_model dimension
            x_enc_processed = self.celestial_to_model(celestial_features)
        else:
            x_enc_processed = x_enc
            static_adjacency = torch.eye(
                self.enc_in, device=x_enc.device
            ).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Input embeddings
        enc_out = self.enc_embedding(x_enc_processed, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Celestial graph processing with static adjacency
        if self.use_celestial_graph:
            # Use static adjacency matrix
            celestial_adj = torch.sigmoid(self.static_celestial_adj)
            celestial_adj = celestial_adj.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Simple celestial fusion
            celestial_context = torch.mean(enc_out, dim=1, keepdim=True)  # [batch, 1, d_model]
            celestial_context = celestial_context.expand(-1, seq_len, -1)  # [batch, seq_len, d_model]
            
            fusion_input = torch.cat([enc_out, celestial_context], dim=-1)
            fusion_gate = self.celestial_fusion(fusion_input)
            enc_out = enc_out + fusion_gate * celestial_context
            
            combined_adj = celestial_adj
        else:
            combined_adj = static_adjacency
        
        # Spatiotemporal encoding
        if self.use_gradient_checkpointing and self.training:
            encoded_features = checkpoint(
                self._apply_spatiotemporal_encoding, enc_out, combined_adj
            )
        else:
            encoded_features = self._apply_spatiotemporal_encoding(enc_out, combined_adj)
        
        # Graph attention processing with gradient checkpointing
        graph_features = encoded_features
        for i, layer in enumerate(self.graph_attention_layers):
            if self.use_gradient_checkpointing and self.training:
                graph_features = checkpoint(layer, graph_features, combined_adj)
            else:
                graph_features = layer(graph_features, combined_adj)
        
        # Decoder processing
        decoder_features = dec_out
        for layer in self.decoder_layers:
            if self.use_gradient_checkpointing and self.training:
                decoder_features = checkpoint(layer, decoder_features, graph_features)
            else:
                decoder_features = layer(decoder_features, graph_features)
        
        # Final prediction
        predictions = self.projection(decoder_features[:, -self.pred_len:, :])
        
        # Explicit memory cleanup
        if self.training:
            self._cleanup_intermediate_tensors()
        
        return predictions
    
    def _apply_spatiotemporal_encoding(self, enc_out: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Apply spatiotemporal encoding with static adjacency."""
        try:
            if adj_matrix.dim() == 3:
                adj_for_encoder = adj_matrix[0]  # Use first batch's adjacency
            else:
                adj_for_encoder = adj_matrix
            
            return self.spatiotemporal_encoder(enc_out, adj_for_encoder)
        except Exception as error:
            self.logger.warning("Spatiotemporal encoding failed: %s", error)
            return enc_out
    
    def _cleanup_intermediate_tensors(self):
        """Explicit cleanup of intermediate tensors to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SimpleDecoderLayer(nn.Module):
    """Simplified decoder layer for memory optimization."""
    
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # Reduced from 4x to 2x
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, dec_input, enc_output):
        """Simplified forward pass."""
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


# Import necessary embedding classes
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
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.c_in:
            raise ValueError(f"TokenEmbedding input_dim ({input_dim}) != c_in ({self.c_in})")
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
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
        num_features = x.size(-1)
        
        minute_x = 0.
        hour_x = 0.
        weekday_x = 0.
        day_x = 0.
        month_x = 0.
        
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