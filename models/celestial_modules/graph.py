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
from layers.modular.graph.gated_graph_combiner import GatedGraphCombiner
from layers.modular.graph.stochastic_learner import StochasticGraphLearner

class GraphModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config
        self.latest_stochastic_loss = 0.0

        if not config.use_celestial_graph:
            return

        self.celestial_nodes = CelestialBodyNodes(d_model=config.d_model, num_aspects=5)

        # Enhanced Combiner Selection
        if config.use_petri_net_combiner:
            self.celestial_combiner = CelestialPetriNetCombiner(
                num_nodes=config.num_celestial_bodies,
                d_model=config.d_model,
                edge_feature_dim=config.edge_feature_dim,
                num_message_passing_steps=config.num_message_passing_steps,
                num_attention_heads=config.n_heads,
                dropout=config.dropout,
                use_temporal_attention=config.use_temporal_attention,
                use_spatial_attention=config.use_spatial_attention,
                use_gradient_checkpointing=True,
                enable_memory_debug=config.enable_memory_debug,
                memory_debug_prefix="PETRI"
            )
        elif config.use_gated_graph_combiner:
            self.celestial_combiner = GatedGraphCombiner(
                num_nodes=config.num_celestial_bodies,
                d_model=config.d_model,
                num_attention_heads=config.n_heads,
                dropout=config.dropout
            )
        else:
            self.celestial_combiner = CelestialGraphCombiner(
                num_nodes=config.num_celestial_bodies,
                d_model=config.d_model,
                num_attention_heads=config.n_heads,
                fusion_layers=config.celestial_fusion_layers,
                dropout=config.dropout,
                use_gradient_checkpointing=True
            )

        # Fusion attention components
        fusion_dim_cfg = getattr(config, 'celestial_fusion_dim', min(config.d_model, 64))
        fusion_dim = max(config.n_heads, fusion_dim_cfg)
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

        # Standardized celestial graph processing
        if config.aggregate_waves_to_celestial:
            # After aggregation: d_model → 13 celestial nodes
            self.feature_to_celestial = nn.Linear(config.d_model, config.num_celestial_bodies)
            self.celestial_to_feature = nn.Linear(config.num_celestial_bodies, config.d_model)
        else:
            # Before aggregation: d_model → 13 celestial nodes (learned mapping)
            self.feature_to_celestial = nn.Linear(config.d_model, config.num_celestial_bodies)
            self.celestial_to_feature = nn.Linear(config.num_celestial_bodies, config.d_model)

        # Graph learners
        adj_output_dim = config.num_graph_nodes * config.num_graph_nodes
        self.traditional_graph_learner = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, adj_output_dim), nn.Tanh()
        )
        
        # Enhanced Stochastic Graph Learner
        if config.use_stochastic_learner:
            num_nodes = config.num_celestial_bodies if (config.use_celestial_graph and config.aggregate_waves_to_celestial) else config.enc_in
            self.stochastic_learner = StochasticGraphLearner(
                d_model=config.d_model,
                num_nodes=num_nodes
            )
        else:
            self.stochastic_learner = None

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

    def _learn_traditional_graph(self, enc_out):
        """Learn a dynamic traditional graph adjacency for each time step."""
        batch_size, seq_len, d_model = enc_out.shape
        
        # Reshape to apply the learner to each time step independently
        enc_out_flat = enc_out.reshape(batch_size * seq_len, d_model)
        adj_flat = self.traditional_graph_learner(enc_out_flat)
        
        # Determine the number of nodes
        num_nodes = self.config.num_celestial_bodies if (self.config.use_celestial_graph and self.config.aggregate_waves_to_celestial) else self.config.enc_in
        
        # Reshape back to a time-varying adjacency matrix
        adj_matrix = adj_flat.view(batch_size, seq_len, num_nodes, num_nodes)
        return adj_matrix
    
    def _learn_simple_dynamic_graph(self, enc_out):
        """Compute a simple distinct dynamic identity adjacency for each time step."""
        batch_size, seq_len, _ = enc_out.shape
        device = enc_out.device
        num_nodes = self.config.num_celestial_bodies if (self.config.use_celestial_graph and self.config.aggregate_waves_to_celestial) else self.config.enc_in
        identity = torch.eye(num_nodes, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return identity.expand(batch_size, seq_len, num_nodes, num_nodes)

    def _learn_data_driven_graph(self, enc_out):
        """Learn a dynamic data-driven graph from encoded features for each time step."""
        batch_size, seq_len, d_model = enc_out.shape
        
        # Reshape to apply the learner to each time step
        enc_out_flat = enc_out.reshape(batch_size * seq_len, d_model)
        
        if self.config.use_stochastic_learner and self.stochastic_learner is not None:
            # Use enhanced stochastic graph learner
            adj_flat, kl_loss = self.stochastic_learner(enc_out_flat)
            self.latest_stochastic_loss = kl_loss
        else:
            # Deterministic graph learning
            adj_flat = self.traditional_graph_learner(enc_out_flat)
            self.latest_stochastic_loss = 0.0
        
        # Determine the number of nodes
        num_nodes = self.config.num_celestial_bodies if self.config.use_celestial_graph else self.config.enc_in
        
        # Reshape to a time-varying adjacency matrix
        adj_matrix = adj_flat.view(batch_size, seq_len, num_nodes, num_nodes)
        
        return adj_matrix

    def _process_celestial_graph(self, x_enc_processed: torch.Tensor, enc_out: torch.Tensor):
        """Process encoder features through the dynamic celestial body graph system."""
        astronomical_adj, dynamic_adj, celestial_features, metadata = self.celestial_nodes(enc_out)
        
        # The celestial features are dynamic: [batch, seq_len, num_bodies, d_model]
        batch_size, seq_len, num_bodies, hidden_dim = celestial_features.shape

        projected_query = self.celestial_query_projection(enc_out)
        projected_keys = self.celestial_key_projection(celestial_features)
        projected_values = self.celestial_value_projection(celestial_features)

        query = projected_query.reshape(batch_size * seq_len, 1, self.celestial_fusion_dim)
        key = projected_keys.reshape(batch_size * seq_len, num_bodies, self.celestial_fusion_dim)
        value = projected_values.reshape(batch_size * seq_len, num_bodies, self.celestial_fusion_dim)

        fused_output, attention_weights = self.celestial_fusion_attention(query, key, value)
        fused_output = fused_output.reshape(batch_size, seq_len, self.celestial_fusion_dim)
        fused_output = self.celestial_output_projection(fused_output)
        
        # Normalize both branches to balance magnitudes before gate fusion
        fused_output = self.celestial_norm(fused_output)
        enc_out_normalized = self.encoder_norm(enc_out)
        
        gate_input = torch.cat([enc_out_normalized, fused_output], dim=-1)
        # Compute pre-sigmoid logits explicitly for diagnostics
        gate_lin1 = self.celestial_fusion_gate[0]
        gate_act = self.celestial_fusion_gate[1]
        gate_lin2 = self.celestial_fusion_gate[2]
        gate_logits = gate_lin2(gate_act(gate_lin1(gate_input)))
        fusion_gate = torch.sigmoid(gate_logits)
        
        # Use normalized enc_out for fusion to maintain balanced gradients
        celestial_influence = fusion_gate * fused_output
        enhanced_enc_out = enc_out_normalized + celestial_influence
        
        if self.config.collect_diagnostics:
            metadata['celestial_attention_weights'] = (
                attention_weights.reshape(batch_size, seq_len, num_bodies).detach().cpu()
            )
        else:
            metadata = {}
        
        return {
            'astronomical_adj': astronomical_adj,
            'dynamic_adj': dynamic_adj,
            'celestial_features': celestial_features,
            'enhanced_enc_out': enhanced_enc_out,
            'metadata': metadata,
        }

    def forward(self, enc_out, market_context, phase_based_adj):
        # 1. Process celestial graph
        celestial_results = self._process_celestial_graph(enc_out, enc_out)
        astronomical_adj = celestial_results['astronomical_adj']
        dynamic_adj = celestial_results['dynamic_adj'] 
        celestial_features = celestial_results['celestial_features']
        celestial_metadata = celestial_results['metadata']
        enhanced_enc_out = celestial_results['enhanced_enc_out']

        # 2. Learn data-driven graph
        learned_adj = self._learn_data_driven_graph(enc_out)

        # 3. Combine all graph types with hierarchical fusion
        if phase_based_adj is not None:
            # Learned context-aware fusion of phase, astronomical, and dynamic adjacencies
            weights = F.softmax(self.adj_weight_mlp(market_context), dim=-1)
            w_phase, w_astro, w_dyn = weights[..., 0], weights[..., 1], weights[..., 2]
            
            # Normalize and combine
            phase_norm = self._normalize_adj(phase_based_adj)
            astro_norm = self._normalize_adj(astronomical_adj)
            dyn_norm = self._normalize_adj(dynamic_adj)
            
            # Unsqueeze weights for broadcasting with dynamic adjs
            w_phase = w_phase.unsqueeze(-1).unsqueeze(-1)
            w_astro = w_astro.unsqueeze(-1).unsqueeze(-1)
            w_dyn = w_dyn.unsqueeze(-1).unsqueeze(-1)
            
            combined_phase_adj = (
                w_phase * phase_norm +
                w_astro * astro_norm +
                w_dyn * dyn_norm
            )
            astronomical_adj = combined_phase_adj
        else:
            # Simple combination for fallback with dynamic weighting
            weights = F.softmax(self.adj_weight_mlp(enc_out), dim=-1)
            w_astro, w_learned, w_dyn = weights[..., 0], weights[..., 1], weights[..., 2]

            # Normalize adjacency matrices
            astro_norm = self._normalize_adj(astronomical_adj)
            learned_norm = self._normalize_adj(learned_adj)
            dyn_norm = self._normalize_adj(dynamic_adj)
            
            # Unsqueeze weights to be broadcastable
            w_astro = w_astro.unsqueeze(-1).unsqueeze(-1)
            w_learned = w_learned.unsqueeze(-1).unsqueeze(-1)
            w_dyn = w_dyn.unsqueeze(-1).unsqueeze(-1)

            combined_adj = (
                w_astro * astro_norm +
                w_learned * learned_norm +
                w_dyn * dyn_norm
            )
            astronomical_adj = combined_adj

        # 4. Use enhanced combiner for final graph structure
        rich_edge_features = None
        fusion_metadata = {}
        
        if self.config.use_petri_net_combiner:
            # Call with return_rich_features=True to get full edge vectors
            combined_adj, rich_edge_features, fusion_metadata = self.celestial_combiner(
                astronomical_adj, learned_adj, dynamic_adj, enc_out,
                return_rich_features=True
            )
            fusion_metadata['rich_edge_features_shape'] = rich_edge_features.shape if rich_edge_features is not None else None
            fusion_metadata['edge_features_preserved'] = True
            fusion_metadata['no_compression'] = True
        else:
            # OLD COMBINER: Returns only scalar adjacency
            combined_adj, fusion_metadata = self.celestial_combiner(
                astronomical_adj, learned_adj, dynamic_adj, enc_out
            )
            fusion_metadata['edge_features_preserved'] = False
        
        return enhanced_enc_out, combined_adj, rich_edge_features, fusion_metadata, celestial_features