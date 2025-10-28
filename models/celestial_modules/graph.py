# models/celestial_modules/graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CelestialPGATConfig

# Import graph components directly to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.modular.graph.celestial_body_nodes import CelestialBodyNodes
from layers.modular.graph.celestial_petri_net_combiner import CelestialPetriNetCombiner
from layers.modular.graph.celestial_graph_combiner import CelestialGraphCombiner

class GraphModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config

        if not config.use_celestial_graph:
            return

        self.celestial_nodes = CelestialBodyNodes(d_model=config.d_model, num_aspects=5)

        if config.use_petri_net_combiner:
            self.celestial_combiner = CelestialPetriNetCombiner(
                num_nodes=config.num_celestial_bodies,
                d_model=config.d_model,
                edge_feature_dim=config.edge_feature_dim,
                num_message_passing_steps=config.num_message_passing_steps,
                num_attention_heads=config.n_heads,
                dropout=config.dropout,
                use_temporal_attention=config.use_temporal_attention,
                use_spatial_attention=config.use_spatial_attention
            )
        else:
            self.celestial_combiner = CelestialGraphCombiner(
                num_nodes=config.num_celestial_bodies,
                d_model=config.d_model,
                num_attention_heads=config.n_heads,
                fusion_layers=config.celestial_fusion_layers,
                dropout=config.dropout
            )

        # Fusion attention components
        fusion_dim = max(config.n_heads, min(config.d_model, 64))
        if fusion_dim % config.n_heads != 0:
            fusion_dim = ((fusion_dim // config.n_heads) + 1) * config.n_heads
        self.celestial_fusion_dim = fusion_dim
        self.celestial_query_projection = nn.Linear(config.d_model, fusion_dim)
        self.celestial_key_projection = nn.Linear(config.d_model, fusion_dim)
        self.celestial_value_projection = nn.Linear(config.d_model, fusion_dim)
        self.celestial_output_projection = nn.Linear(fusion_dim, config.d_model)
        self.celestial_fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.celestial_fusion_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model), nn.Sigmoid()
        )
        self.celestial_norm = nn.LayerNorm(config.d_model)
        self.encoder_norm = nn.LayerNorm(config.d_model)

        # Graph learners
        adj_output_dim = config.num_graph_nodes * config.num_graph_nodes
        self.traditional_graph_learner = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, adj_output_dim), nn.Tanh()
        )
        if config.use_stochastic_learner:
            self.stochastic_mean = nn.Linear(config.d_model, adj_output_dim)
            self.stochastic_logvar = nn.Linear(config.d_model, adj_output_dim)

        self.adj_weight_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2), nn.ReLU(),
            nn.Linear(config.d_model // 2, 3)
        )
    
    def _normalize_adj(self, adj_matrix):
        identity = torch.eye(adj_matrix.size(-1), device=adj_matrix.device, dtype=torch.float32)
        identity = identity.unsqueeze(0).unsqueeze(0)
        adj_with_self_loops = adj_matrix + identity.expand_as(adj_matrix)
        row_sums = adj_with_self_loops.sum(dim=-1, keepdim=True)
        return adj_with_self_loops / (row_sums + 1e-8)

    def _learn_data_driven_graph(self, enc_out):
        batch_size, seq_len, d_model = enc_out.shape
        enc_out_flat = enc_out.view(batch_size * seq_len, d_model)
        
        if self.config.use_stochastic_learner:
            mean = self.stochastic_mean(enc_out_flat)
            logvar = self.stochastic_logvar(enc_out_flat)
            if self.training:
                std = torch.exp(0.5 * logvar)
                adj_flat = mean + torch.randn_like(std) * std
            else:
                adj_flat = mean
        else:
            adj_flat = self.traditional_graph_learner(enc_out_flat)
            
        return adj_flat.view(batch_size, seq_len, self.config.num_graph_nodes, self.config.num_graph_nodes)

    def forward(self, enc_out, market_context, phase_based_adj):
        # 1. Get Astronomical and Dynamic graphs from Celestial Nodes
        astro_adj, dyn_adj, celestial_graph_feats, _ = self.celestial_nodes(enc_out)
        
        # 2. Fuse celestial graph features back into the main encoder stream
        b, s, n, d = celestial_graph_feats.shape
        query = self.celestial_query_projection(enc_out).view(b * s, 1, -1)
        keys = self.celestial_key_projection(celestial_graph_feats).view(b * s, n, -1)
        values = self.celestial_value_projection(celestial_graph_feats).view(b * s, n, -1)
        
        fused, _ = self.celestial_fusion_attention(query, keys, values)
        fused = self.celestial_output_projection(fused.view(b, s, -1))
        
        enc_out_norm = self.encoder_norm(enc_out)
        fused_norm = self.celestial_norm(fused)
        
        gate_input = torch.cat([enc_out_norm, fused_norm], dim=-1)
        fusion_gate = self.celestial_fusion_gate(gate_input)
        enhanced_enc_out = enc_out_norm + fusion_gate * fused_norm

        # 3. Learn data-driven graph
        learned_adj = self._learn_data_driven_graph(enc_out)

        # 4. Fuse all adjacency matrices
        weights = F.softmax(self.adj_weight_mlp(market_context), dim=-1)
        w_phase, w_astro, w_dyn = weights[..., 0], weights[..., 1], weights[..., 2]
        
        # Combine using weights
        if phase_based_adj is not None:
             # Use the rich phase-based adj if available
            combined_adj = (
                w_phase.unsqueeze(-1).unsqueeze(-1) * self._normalize_adj(phase_based_adj) +
                w_astro.unsqueeze(-1).unsqueeze(-1) * self._normalize_adj(astro_adj) +
                w_dyn.unsqueeze(-1).unsqueeze(-1) * self._normalize_adj(dyn_adj)
            )
        else: # Fallback to learned adj
            combined_adj = (
                w_phase.unsqueeze(-1).unsqueeze(-1) * self._normalize_adj(learned_adj) +
                w_astro.unsqueeze(-1).unsqueeze(-1) * self._normalize_adj(astro_adj) +
                w_dyn.unsqueeze(-1).unsqueeze(-1) * self._normalize_adj(dyn_adj)
            )

        # 5. Use Petri Net Combiner for final graph structure if enabled
        rich_edge_features = None
        if self.config.use_petri_net_combiner:
             combined_adj, rich_edge_features, _ = self.celestial_combiner(
                    astro_adj, learned_adj, dyn_adj, enc_out, return_rich_features=True
                )
        
        return enhanced_enc_out, combined_adj, rich_edge_features