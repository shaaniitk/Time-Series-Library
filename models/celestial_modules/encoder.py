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

    def forward(self, enc_out, combined_adj, rich_edge_features):
        # 1. Hierarchical Mapping
        if self.config.use_hierarchical_mapping and hasattr(self, 'hierarchical_mapper'):
            hierarchical_features = self.hierarchical_mapper(enc_out).view(enc_out.size(0), -1)
            projected_features = self.hierarchical_projection(hierarchical_features).unsqueeze(1)
            enc_out = enc_out + projected_features

        # 2. Spatiotemporal Encoding
        if self.config.use_petri_net_combiner and self.config.bypass_spatiotemporal_with_petri:
            encoded_features = enc_out
        else:
            adj_for_encoder = combined_adj if self.config.use_dynamic_spatiotemporal_encoder else combined_adj[:, -1, :, :]
            encoded_features = self.spatiotemporal_encoder(enc_out, adj_for_encoder)
        
        # 3. Graph Attention Processing
        graph_features = encoded_features
        if self.config.use_petri_net_combiner and rich_edge_features is not None:
            for layer in self.graph_attention_layers:
                graph_features = layer(graph_features, edge_features=rich_edge_features)
        else:
            for t in range(self.config.seq_len):
                time_step_features = graph_features[:, t:t+1, :]
                adj_for_step = combined_adj[:, t, :, :]
                for layer in self.graph_attention_layers:
                    time_step_features = layer(time_step_features, adj_for_step)
                graph_features[:, t:t+1, :] = time_step_features
        
        return graph_features