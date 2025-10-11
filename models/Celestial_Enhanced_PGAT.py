"""
Celestial Enhanced PGAT - Revolutionary Astrological AI for Financial Markets

This model combines the Enhanced SOTA PGAT with celestial body graph nodes,
creating the world's first astronomically-informed time series forecasting model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from layers.modular.graph.celestial_body_nodes import CelestialBodyNodes, CelestialBody
from layers.modular.graph.celestial_graph_combiner import CelestialGraphCombiner
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder
from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureDensityDecoder
from layers.modular.embedding.hierarchical_mapper import HierarchicalTemporalSpatialMapper
from layers.modular.encoder.spatiotemporal_encoding import JointSpatioTemporalEncoding
from layers.modular.graph.gated_graph_combiner import GatedGraphCombiner
from layers.Embed import DataEmbedding
from utils.celestial_wave_aggregator import CelestialWaveAggregator, CelestialDataProcessor
from layers.modular.aggregation.phase_aware_celestial_processor import PhaseAwareCelestialProcessor


class Model(nn.Module):
    """
    Celestial Enhanced PGAT - Astrological AI for Financial Time Series
    
    Revolutionary model that represents market influences as celestial bodies
    with learned relationships based on astrological aspects and market dynamics.
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
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
        
        # Enhanced features
        self.use_mixture_decoder = getattr(configs, 'use_mixture_decoder', True)
        self.use_stochastic_learner = getattr(configs, 'use_stochastic_learner', True)
        self.use_hierarchical_mapping = getattr(configs, 'use_hierarchical_mapping', True)
        
        # Wave aggregation settings
        self.aggregate_waves_to_celestial = getattr(configs, 'aggregate_waves_to_celestial', True)
        self.num_input_waves = getattr(configs, 'num_input_waves', 118)
        self.target_wave_indices = getattr(configs, 'target_wave_indices', [0, 1, 2, 3])
        
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
        
        print(f"🌌 Initializing Celestial Enhanced PGAT:")
        print(f"   - Sequence length: {self.seq_len}")
        print(f"   - Prediction length: {self.pred_len}")
        print(f"   - Model dimension: {self.d_model}")
        print(f"   - Celestial bodies: {self.num_celestial_bodies}")
        print(f"   - Wave aggregation: {self.aggregate_waves_to_celestial}")
        if self.aggregate_waves_to_celestial:
            print(f"   - Input waves: {self.num_input_waves} → {self.num_celestial_bodies} celestial bodies")
            print(f"   - Target waves: {len(self.target_wave_indices)} (OHLC)")
        print(f"   - Enhanced features: MixtureDec={self.use_mixture_decoder}, "
              f"Stochastic={self.use_stochastic_learner}, Hierarchical={self.use_hierarchical_mapping}")
        
        # Input embeddings - adjust for phase-aware celestial aggregation
        if self.aggregate_waves_to_celestial:
            # Phase-aware processor outputs 13 celestial bodies × 32D = 416D
            enc_embedding_dim = self.num_celestial_bodies * 32
        else:
            # Embedding expects original wave count
            enc_embedding_dim = self.enc_in
            
        self.enc_embedding = DataEmbedding(
            enc_embedding_dim, self.d_model, configs.embed, configs.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in, self.d_model, configs.embed, configs.freq, self.dropout
        )
        
        # Celestial Body Graph System
        if self.use_celestial_graph:
            self.celestial_nodes = CelestialBodyNodes(
                d_model=self.d_model,
                num_aspects=5
            )
            
            self.celestial_combiner = CelestialGraphCombiner(
                num_nodes=self.num_celestial_bodies,
                d_model=self.d_model,
                num_attention_heads=self.n_heads,
                fusion_layers=self.celestial_fusion_layers,
                dropout=self.dropout
            )
            
            # Map input features to celestial space - adjust for aggregation
            if self.aggregate_waves_to_celestial:
                # After aggregation: 13 celestial → 13 celestial (identity or refinement)
                self.feature_to_celestial = nn.Linear(self.num_celestial_bodies, self.num_celestial_bodies)
                self.celestial_to_feature = nn.Linear(self.num_celestial_bodies, self.num_celestial_bodies)
            else:
                # Before aggregation: 118 waves → 13 celestial
                self.feature_to_celestial = nn.Linear(self.enc_in, self.num_celestial_bodies)
                self.celestial_to_feature = nn.Linear(self.num_celestial_bodies, self.enc_in)
        
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
            
        self.spatiotemporal_encoder = JointSpatioTemporalEncoding(
            d_model=self.d_model,
            seq_len=self.seq_len,
            num_nodes=num_nodes_for_encoding,
            num_heads=self.n_heads,
            dropout=self.dropout
        )
        
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
        
        # Graph attention layers - use the fixed adjacency-aware version
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
        
        # Market context encoder
        self.market_context_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
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
        batch_size, seq_len, enc_in = x_enc.shape
        
        # 0. Enhanced Phase-Aware Wave Processing (if enabled)
        wave_metadata = {}
        if self.aggregate_waves_to_celestial and enc_in == self.num_input_waves:
            # Use phase-aware processor for rich celestial representations
            celestial_features, adjacency_matrix, phase_metadata = self.phase_aware_processor(x_enc)
            # celestial_features: [batch, seq_len, 13 * 32] Rich multi-dimensional representations
            # adjacency_matrix: [batch, 13, 13] Phase-difference based edges
            
            # Extract targets using the old processor (for compatibility)
            _, target_waves, target_metadata = self.data_processor.process_batch(x_enc)
            
            # Use rich celestial features as encoder input
            x_enc_processed = celestial_features  # [batch, seq_len, 13 * 32]
            
            # Store comprehensive metadata
            wave_metadata = {
                'original_targets': target_waves,  # [batch, seq_len, 4]
                'phase_metadata': phase_metadata,
                'target_metadata': target_metadata,
                'adjacency_matrix': adjacency_matrix,
                'celestial_features_shape': celestial_features.shape
            }
            
            print(f"🌌 Phase-aware processing: {x_enc.shape} → celestial {celestial_features.shape}, targets {target_waves.shape}")
            print(f"🔗 Phase-based adjacency matrix: {adjacency_matrix.shape}")
        else:
            x_enc_processed = x_enc
            wave_metadata['original_targets'] = None
        
        # 1. Input embeddings - ensure correct dimensions
        try:
            enc_out = self.enc_embedding(x_enc_processed, x_mark_enc)  # [batch, seq_len, d_model]
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            print(f"   Input shape: {x_enc_processed.shape}")
            print(f"   Expected embedding input dim: {self.enc_embedding.value_embedding.c_in}")
            raise e
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [batch, label_len+pred_len, d_model]
        
        # 2. Generate market context from encoder output (fix information bottleneck)
        # Use last hidden state instead of mean to preserve temporal information
        market_context = self.market_context_encoder(enc_out[:, -1, :])  # [batch, d_model]
        
        # 3. Enhanced Celestial Body Graph Processing
        if self.use_celestial_graph:
            if self.aggregate_waves_to_celestial and 'adjacency_matrix' in wave_metadata:
                # Use phase-based adjacency matrix from phase-aware processor
                phase_based_adj = wave_metadata['adjacency_matrix']  # [batch, 13, 13]
                
                # Still process celestial graph for additional features
                celestial_results = self._process_celestial_graph(x_enc_processed, market_context)
                astronomical_adj = celestial_results['astronomical_adj']
                dynamic_adj = celestial_results['dynamic_adj'] 
                celestial_features = celestial_results['celestial_features']
                celestial_metadata = celestial_results['metadata']
                
                # Combine phase-based and traditional adjacency matrices
                combined_phase_adj = (phase_based_adj + astronomical_adj + dynamic_adj) / 3
                astronomical_adj = combined_phase_adj  # Use combined matrix
                
                print(f"🌟 Using phase-based adjacency matrix combined with traditional methods")
            else:
                # Fallback to traditional celestial graph processing
                celestial_results = self._process_celestial_graph(x_enc_processed, market_context)
                astronomical_adj = celestial_results['astronomical_adj']
                dynamic_adj = celestial_results['dynamic_adj'] 
                celestial_features = celestial_results['celestial_features']
                celestial_metadata = celestial_results['metadata']
        else:
            # Fallback to traditional graph learning
            traditional_adj = self._learn_traditional_graph(market_context, batch_size)
            astronomical_adj = traditional_adj
            dynamic_adj = traditional_adj
            celestial_features = None
            celestial_metadata = {}
        
        # 4. Learn data-driven graph
        learned_adj = self._learn_data_driven_graph(enc_out, market_context)
        
        # 5. Combine all graph types with hierarchical fusion
        if self.use_celestial_graph:
            combined_adj, fusion_metadata = self.celestial_combiner(
                astronomical_adj, learned_adj, dynamic_adj, market_context
            )
        else:
            # Simple combination for fallback
            combined_adj = (astronomical_adj + learned_adj + dynamic_adj) / 3
            fusion_metadata = {}
        
        # 6. Apply hierarchical mapping if enabled
        if self.use_hierarchical_mapping:
            try:
                hierarchical_features = self.hierarchical_mapper(enc_out)  # [batch, num_nodes, d_model]
                
                # Reshape and project to preserve spatial information
                batch_size, seq_len, _ = enc_out.shape
                reshaped_features = hierarchical_features.view(batch_size, -1)
                projected_features = self.hierarchical_projection(reshaped_features).unsqueeze(1)
                
                # Add to encoder output, broadcasting across sequence length
                enc_out = enc_out + projected_features
                
            except Exception as e:
                print(f"⚠️  Hierarchical mapping failed: {e}")
                print("   Continuing without hierarchical features")
        
        # 7. Spatiotemporal encoding with graph attention
        try:
            # The spatiotemporal encoder now handles both 3D and 4D inputs
            # Prepare adjacency matrix
            if combined_adj.dim() == 3:  # [batch, num_nodes, num_nodes]
                # Use the first batch's adjacency matrix (assuming they're similar)
                adj_for_spatiotemporal = combined_adj[0]  # [num_nodes, num_nodes]
            else:
                adj_for_spatiotemporal = combined_adj
            
            if self.use_celestial_graph and self.aggregate_waves_to_celestial:
                # For celestial aggregation, pass 3D input directly
                # The spatiotemporal encoder will handle it with the aggregated method
                num_nodes = self.num_celestial_bodies
                
                # Ensure adjacency matrix matches celestial bodies
                if adj_for_spatiotemporal.size(0) != num_nodes:
                    adj_for_spatiotemporal = torch.eye(num_nodes, device=enc_out.device)
                
                # Pass 3D input - spatiotemporal encoder will use _forward_aggregated
                encoded_features = self.spatiotemporal_encoder(enc_out, adj_for_spatiotemporal)
            else:
                # For original case, reshape to 4D and use original method
                batch_size, seq_len, d_model = enc_out.shape
                num_nodes = self.enc_in
                
                # Ensure adjacency matrix matches number of nodes
                if adj_for_spatiotemporal.size(0) != num_nodes:
                    adj_for_spatiotemporal = torch.eye(num_nodes, device=enc_out.device)
                
                # Reshape to 4D for original spatiotemporal processing
                node_dim = d_model // num_nodes if d_model % num_nodes == 0 else 1
                if d_model % num_nodes == 0:
                    enc_out_4d = enc_out.view(batch_size, seq_len, num_nodes, node_dim)
                    encoded_features_4d = self.spatiotemporal_encoder(enc_out_4d, adj_for_spatiotemporal)
                    encoded_features = encoded_features_4d.view(batch_size, seq_len, d_model)
                else:
                    # Use 3D aggregated method as fallback
                    encoded_features = self.spatiotemporal_encoder(enc_out, adj_for_spatiotemporal)
            
        except Exception as e:
            print(f"⚠️  Spatiotemporal encoding failed: {e}")
            encoded_features = enc_out
        
        # 8. Graph attention processing with proper adjacency matrix usage
        graph_features = encoded_features
        for i, layer in enumerate(self.graph_attention_layers):
            try:
                # Use the adjacency matrix properly in graph attention
                graph_features = layer(graph_features, combined_adj)
                
                if i == 0:  # Log once to confirm adjacency matrix is being used
                    print(f"🔗 Graph attention layer {i+1} using adjacency matrix: {combined_adj.shape}")
                    
            except Exception as e:
                print(f"⚠️  Graph attention layer {i+1} failed: {e}")
                # Fallback: use without adjacency matrix
                graph_features = layer(graph_features, None)
        
        # 9. Decoder processing
        decoder_features = dec_out
        for layer in self.decoder_layers:
            decoder_features = layer(decoder_features, graph_features)
        
        # 10. Final prediction with Sequential Mixture Decoder
        if self.use_mixture_decoder:
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
                    print(f"⚠️  Sequential mixture decoder output shape mismatch: {predictions.shape} vs expected [..., {self.c_out}]")
                    # Fallback to projection
                    predictions = self.projection(prediction_features)
                
                output = predictions
                mixture_params = {'means': means, 'log_stds': log_stds, 'log_weights': log_weights}
                
                print(f"🎯 Sequential mixture decoder output: means {means.shape}, predictions {predictions.shape}")
                
            except Exception as e:
                print(f"⚠️  Sequential mixture decoder failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simple projection
                output = self.projection(decoder_features[:, -self.pred_len:, :])
                mixture_params = {}
        else:
            output = self.projection(decoder_features)
        
        # Prepare metadata for return
        final_metadata = {
            **wave_metadata,  # Include wave aggregation metadata
            'celestial_metadata': celestial_metadata if self.use_celestial_graph else {},
            'fusion_metadata': fusion_metadata if self.use_celestial_graph else {}
        }
        
        # In case of MDN, the raw tuple/dict is returned for the loss function
        # A point prediction is calculated within the training script for metrics
        if self.use_mixture_decoder and 'mixture_params' in locals():
            output = mixture_params
        else:
            output = predictions if 'predictions' in locals() else output
        
        return output, final_metadata
    
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
        return forward_output # Output is already a point prediction
    
    def _process_celestial_graph(self, x_enc, market_context):
        """Process input through celestial body graph system"""
        batch_size = x_enc.size(0)
        
        # Get celestial body representations
        astronomical_adj, dynamic_adj, celestial_features, metadata = self.celestial_nodes(market_context)
        
        # Map input features to celestial space for enhanced processing
        if hasattr(self, 'phase_aware_processor') and x_enc.size(-1) > 13:
            # Use the new projection layer to correctly map rich features
            celestial_mapped = self.rich_feature_to_celestial(x_enc)
        else:
            # Traditional mapping for 13D input
            celestial_mapped = self.feature_to_celestial(x_enc)  # [batch, seq_len, num_celestial_bodies]
        
        # Enhance celestial features with temporal information (fix oversimplification)
        # Use last time step instead of mean to preserve temporal information
        temporal_celestial = celestial_mapped[:, -1, :]  # [batch, num_celestial_bodies]
        
        # Create a more sophisticated celestial influence using proper dimension handling
        # celestial_features: [batch, num_celestial_bodies, d_model]
        # temporal_celestial: [batch, num_celestial_bodies]
        
        # Project temporal celestial to d_model space
        temporal_celestial_expanded = temporal_celestial.unsqueeze(-1).expand(-1, -1, self.d_model)  # [batch, num_celestial_bodies, d_model]
        
        # Element-wise multiplication and sum to create influence
        celestial_influence_raw = (celestial_features * temporal_celestial_expanded).sum(dim=1)  # [batch, d_model]
        
        # Normalize the influence
        celestial_influence = F.tanh(celestial_influence_raw)  # [batch, d_model]
        
        # Add celestial influence to market context
        enhanced_context = market_context + celestial_influence
        
        return {
            'astronomical_adj': astronomical_adj,
            'dynamic_adj': dynamic_adj,
            'celestial_features': celestial_features,
            'enhanced_context': enhanced_context,
            'metadata': metadata
        }
    
    def _learn_traditional_graph(self, market_context, batch_size):
        """Learn traditional graph adjacency"""
        adj_flat = self.traditional_graph_learner(market_context)  # [batch, enc_in*enc_in]
        adj_matrix = adj_flat.view(batch_size, self.enc_in, self.enc_in)
        return adj_matrix
    
    def _learn_data_driven_graph(self, enc_out, market_context):
        """Learn data-driven graph from encoded features"""
        batch_size = enc_out.size(0)
        
        # Compute feature similarities
        feature_summary = enc_out.mean(dim=1)  # [batch, d_model]
        combined_context = feature_summary + market_context
        
        if self.use_stochastic_learner:
            # Stochastic graph learning
            mean = self.stochastic_mean(combined_context)
            logvar = self.stochastic_logvar(combined_context)
            
            if self.training:
                # Sample during training
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                adj_flat = mean + eps * std
                
                # Store KL divergence as regularization loss (per batch item)
                kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
                self.latest_stochastic_loss = kl_div.mean()
            else:
                # Use mean during inference
                adj_flat = mean
                self.latest_stochastic_loss = 0.0
        else:
            # Deterministic graph learning
            adj_flat = self.traditional_graph_learner(combined_context)
            self.latest_stochastic_loss = 0.0
        
        # Reshape to adjacency matrix
        if self.use_celestial_graph:
            adj_matrix = adj_flat.view(batch_size, self.num_celestial_bodies, self.num_celestial_bodies)
        else:
            adj_matrix = adj_flat.view(batch_size, self.enc_in, self.enc_in)
        
        return adj_matrix


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
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
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