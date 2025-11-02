# models/celestial_modules/encoder.py

import torch
import torch.nn as nn
from .config import CelestialPGATConfig

# Import components directly to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.modular.embedding.hierarchical_mapper import HierarchicalTemporalSpatialMapper
from layers.modular.encoder.spatiotemporal_encoding import JointSpatioTemporalEncoding, DynamicJointSpatioTemporalEncoding
from layers.modular.graph.adjacency_aware_attention import EdgeConditionedGraphAttention, AdjacencyAwareGraphAttention

class EncoderModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config

        # FIX ISSUE #2: Initialize blend gate in __init__ for proper parameter registration
        if config.use_petri_net_combiner and config.bypass_spatiotemporal_with_petri:
            # Blend gate for soft blending between Petri input and spatiotemporal features
            # Initialized to favor Petri input (0.8 → sigmoid ≈ 0.69 weight for Petri)
            self.encoder_blend_gate = nn.Parameter(torch.tensor(0.8))
        else:
            self.encoder_blend_gate = None

        if config.use_hierarchical_mapping:
            self.hierarchical_mapper = HierarchicalTemporalSpatialMapper(
                d_model=config.d_model, num_nodes=config.num_graph_nodes, n_heads=config.n_heads, num_attention_layers=2
            )
            self.hierarchical_projection = nn.Linear(config.num_graph_nodes * config.d_model, config.d_model)

        if config.use_dynamic_spatiotemporal_encoder:
            self.spatiotemporal_encoder = DynamicJointSpatioTemporalEncoding(
                d_model=config.d_model, seq_len=config.seq_len, num_nodes=config.num_graph_nodes,
                num_heads=config.n_heads, dropout=config.dropout
            )
        else:
            self.spatiotemporal_encoder = JointSpatioTemporalEncoding(
                 d_model=config.d_model, seq_len=config.seq_len, num_nodes=config.num_graph_nodes,
                num_heads=config.n_heads, dropout=config.dropout
            )

        if config.use_petri_net_combiner:
            self.graph_attention_layers = nn.ModuleList([
                EdgeConditionedGraphAttention(
                    d_model=config.d_model, d_ff=config.d_model, n_heads=config.n_heads,
                    edge_feature_dim=config.edge_feature_dim, dropout=config.dropout
                ) for _ in range(config.e_layers)
            ])
        else:
            self.graph_attention_layers = nn.ModuleList([
                AdjacencyAwareGraphAttention(
                    d_model=config.d_model, d_ff=config.d_model, n_heads=config.n_heads,
                    dropout=config.dropout, use_adjacency_mask=True
                ) for _ in range(config.e_layers)
            ])

    def _apply_static_spatiotemporal_encoding(self, enc_out: torch.Tensor, combined_adj: torch.Tensor) -> torch.Tensor:
        """Fallback path that uses the legacy joint encoder with best effort adjacencies."""
        try:
            # combined_adj is always 4D, use last timestep for static encoder
            adj_for_encoder = combined_adj[:, -1, :, :]

            if self.config.use_celestial_graph and self.config.aggregate_waves_to_celestial:
                num_nodes = self.config.num_celestial_bodies
                if adj_for_encoder.size(1) != num_nodes:
                    adj_for_encoder = torch.eye(num_nodes, device=enc_out.device, dtype=torch.float32)
                return self.spatiotemporal_encoder(enc_out, adj_for_encoder)

            batch_size, seq_len, d_model = enc_out.shape
            num_nodes = self.config.enc_in
            if adj_for_encoder.size(1) != num_nodes:
                adj_for_encoder = torch.eye(num_nodes, device=enc_out.device, dtype=torch.float32)

            if d_model % num_nodes == 0 and num_nodes > 0:
                node_dim = d_model // num_nodes
                enc_out_4d = enc_out.view(batch_size, seq_len, num_nodes, node_dim)
                encoded_features_4d = self.spatiotemporal_encoder(enc_out_4d, adj_for_encoder)
                return encoded_features_4d.view(batch_size, seq_len, d_model)

            return self.spatiotemporal_encoder(enc_out, adj_for_encoder)

        except Exception as error:
            return enc_out

    def forward(self, enc_out, combined_adj, rich_edge_features):
        # 1. Hierarchical Mapping
        if self.config.use_hierarchical_mapping and hasattr(self, 'hierarchical_mapper'):
            try:
                hierarchical_features = self.hierarchical_mapper(enc_out)
                batch_size, seq_len, _ = enc_out.shape
                
                # Check if hierarchical_features has sequence dimension
                if hierarchical_features.dim() == 3 and hierarchical_features.size(1) == seq_len:
                    # hierarchical_features: [batch, seq_len, num_nodes*d_model] - already temporal
                    reshaped_features = hierarchical_features.view(batch_size, seq_len, -1)
                    projected_features = self.hierarchical_projection(reshaped_features)
                    enc_out = enc_out + projected_features
                else:
                    # Fallback: hierarchical_features is [batch, num_nodes, d_model] - spatial only
                    reshaped_features = hierarchical_features.view(batch_size, -1)
                    projected_features = self.hierarchical_projection(reshaped_features)
                    projected_features = projected_features.unsqueeze(1).expand(-1, seq_len, -1)
                    enc_out = enc_out + projected_features
            except Exception:
                pass  # Continue without hierarchical features

        # 2. Spatiotemporal Encoding
        # FIX ISSUE #2: Always compute spatiotemporal features for soft blending
        # Even when bypass_spatiotemporal_with_petri=True, we need to compute these
        # features to blend them with enc_out and prevent gradient starvation
        
        use_dynamic_encoder = (
            self.config.use_dynamic_spatiotemporal_encoder and 
            hasattr(self, 'spatiotemporal_encoder') and 
            hasattr(self.spatiotemporal_encoder, '__class__') and
            'Dynamic' in self.spatiotemporal_encoder.__class__.__name__
        )
        
        if use_dynamic_encoder:
            try:
                spatiotemporal_features = self.spatiotemporal_encoder(enc_out, combined_adj)
            except ValueError:
                spatiotemporal_features = self._apply_static_spatiotemporal_encoding(enc_out, combined_adj)
        else:
            spatiotemporal_features = self._apply_static_spatiotemporal_encoding(enc_out, combined_adj)
        
        # FIX ISSUE #2 Part 2: Implement soft blending when bypass is enabled
        if self.config.use_petri_net_combiner and self.config.bypass_spatiotemporal_with_petri:
            # Soft blend: learned gate between Petri input (enc_out) and spatiotemporal features
            # This prevents gradient starvation while maintaining Petri's benefits
            if self.encoder_blend_gate is None:
                raise RuntimeError("encoder_blend_gate should have been initialized in __init__")
            
            blend_weight = torch.sigmoid(self.encoder_blend_gate)
            encoded_features = blend_weight * enc_out + (1 - blend_weight) * spatiotemporal_features
        else:
            # No Petri or bypass disabled: use spatiotemporal output directly
            encoded_features = spatiotemporal_features
        
        # 3. Graph Attention Processing
        graph_features = encoded_features
        
        if self.config.use_petri_net_combiner and rich_edge_features is not None:
            # Use rich edge features directly in attention (no time loop needed)
            for i, layer in enumerate(self.graph_attention_layers):
                try:
                    graph_features = layer(graph_features, edge_features=rich_edge_features)
                except Exception:
                    # Fallback to adjacency-only
                    graph_features = layer(graph_features, adj_matrix=combined_adj)
        else:
            # Process each timestep with scalar adjacency
            processed_features_over_time = []
            for t in range(self.config.seq_len):
                time_step_features = graph_features[:, t:t+1, :]
                adj_for_step = combined_adj[:, t, :, :]
                
                processed_step = time_step_features
                for i, layer in enumerate(self.graph_attention_layers):
                    try:
                        processed_step = layer(processed_step, adj_for_step)
                    except Exception:
                        processed_step = layer(processed_step, None)
                
                processed_features_over_time.append(processed_step)
            
            graph_features = torch.cat(processed_features_over_time, dim=1)
        
        return graph_features