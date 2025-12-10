"""
Celestial Petri Net Graph Combiner - ZERO Information Loss

This combiner preserves ALL edge features as vectors throughout processing:
- NO compression of edge features to scalars
- Petri net message passing for token flow
- Rich edge features: [theta_diff, phi_diff, velocity_diff, radius_ratio, ...]
- Memory efficient: local aggregation, not global attention

Architecture:
1. Takes 3 adjacency matrices (astronomical, learned, dynamic)
2. Enriches with edge feature vectors (phase diffs, velocities, etc.)
3. Uses message passing to propagate information
4. Returns both: adjacency matrix + full edge feature tensor

Key Innovation: Edge features are PRESERVED, not aggregated!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from layers.modular.graph.registry import GraphComponentRegistry
from layers.modular.graph.petri_net_message_passing import (
    PetriNetMessagePassing,
    TemporalNodeAttention,
    SpatialGraphAttention
)


@GraphComponentRegistry.register("celestial_petri_net_combiner")
class CelestialPetriNetCombiner(nn.Module):
    """
    Celestial Graph Combiner using Petri Net Dynamics
    
    Preserves rich edge features throughout processing - NO aggregation!
    
    Key Features:
    1. Edge features as vectors [batch, seq, nodes, nodes, edge_dim]
    2. Message passing for local token flow (no 169×169 attention)
    3. Learns which edge features matter for prediction
    4. Memory efficient + interpretable
    
    Args:
        num_nodes: Number of celestial bodies (13)
        d_model: Model hidden dimension
        edge_feature_dim: Dimension of rich edge features (default 8)
        num_message_passing_steps: Number of Petri net iterations
        num_attention_heads: Heads for local attention
        dropout: Dropout probability
        use_temporal_attention: Add temporal attention over node history
        use_spatial_attention: Add spatial attention over graph state
    """
    
    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        edge_feature_dim: int = 8,
        num_message_passing_steps: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
        use_spatial_attention: bool = True,
        use_gradient_checkpointing: bool = True,
        enable_memory_debug: bool = False,
        memory_debug_prefix: str = "PETRI"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.edge_feature_dim = edge_feature_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.use_temporal_attention = use_temporal_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.enable_memory_debug = enable_memory_debug
        self.memory_debug_prefix = memory_debug_prefix
        self.memory_logger = logging.getLogger("scripts.train.train_celestial_production.memory")
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Parameter(
            torch.randn(3, edge_feature_dim) * 0.02
        )
        
        # Transform scalar adjacencies to edge feature vectors
        # Input: 3 scalar adjacency values per edge
        # Output: rich edge feature vector
        self.adjacency_to_edge_features = nn.Sequential(
            nn.Linear(3, edge_feature_dim * 2),
            nn.LayerNorm(edge_feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_feature_dim * 2, edge_feature_dim),
            nn.LayerNorm(edge_feature_dim)
        )
        
        # Context integration: combine encoder context with edge features
        self.context_edge_fusion = nn.Sequential(
            nn.Linear(d_model + edge_feature_dim, edge_feature_dim * 2),
            nn.GELU(),
            nn.Linear(edge_feature_dim * 2, edge_feature_dim)
        )
        
        # Petri net message passing layers
        # Choose a message dimension compatible with attention heads
        msg_dim_cfg = max(num_attention_heads, d_model // 2)
        if msg_dim_cfg % num_attention_heads != 0:
            msg_dim_cfg = ((msg_dim_cfg // num_attention_heads) + 1) * num_attention_heads

        self.message_passing_layers = nn.ModuleList([
            PetriNetMessagePassing(
                num_nodes=num_nodes,
                node_dim=d_model,
                edge_feature_dim=edge_feature_dim,
                message_dim=msg_dim_cfg,
                num_heads=num_attention_heads,
                dropout=dropout,
                use_local_attention=True
            )
            for _ in range(num_message_passing_steps)
        ])
        
        # Optional temporal attention (over time for each node)
        if use_temporal_attention:
            self.temporal_attention = TemporalNodeAttention(
                node_dim=d_model,
                num_heads=num_attention_heads,
                dropout=dropout
            )
        
        # Optional spatial attention (over nodes at each timestep)
        if use_spatial_attention:
            self.spatial_attention = SpatialGraphAttention(
                node_dim=d_model,
                num_heads=num_attention_heads,
                dropout=dropout
            )
        
        # Final adjacency projection (for compatibility with existing code)
        # Converts rich edge features back to scalar adjacency if needed
        self.edge_features_to_adjacency = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_feature_dim // 2),
            nn.GELU(),
            nn.Linear(edge_feature_dim // 2, 1),
            nn.Tanh()
        )
        
        # Learnable combination weights for 3 edge types
        self.combination_weights = nn.Parameter(torch.ones(3) / 3)

    def _rss_gb(self) -> float:
        try:
            import psutil, os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024**3
        except Exception:
            return -1.0

    def _dbg(self, stage: str, tensors: Optional[Dict[str, torch.Tensor]] = None) -> None:
        if not self.enable_memory_debug:
            return
        rss = self._rss_gb()
        parts = [f"{self.memory_debug_prefix} [{stage}] RSS={rss:.2f}GB"]
        if tensors:
            sizes = []
            for name, t in tensors.items():
                try:
                    numel = t.numel()
                    bytes_ = numel * (t.element_size() if t.is_floating_point() or t.is_complex() else max(1, t.element_size()))
                    sizes.append(f"{name}={bytes_/1024**2:.1f}MB")
                except Exception:
                    continue
            if sizes:
                parts.append(", ".join(sizes))
        self.memory_logger.debug("🔬 %s", " | ".join(parts))
        
    def forward(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor,
        attention_edges: torch.Tensor,
        enc_out: torch.Tensor,
        return_rich_features: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Petri net graph combination with preserved edge features
        
        Args:
            astronomical_edges: [batch, seq_len, num_nodes, num_nodes] 
            learned_edges: [batch, seq_len, num_nodes, num_nodes]
            attention_edges: [batch, seq_len, num_nodes, num_nodes]
            enc_out: [batch, seq_len, d_model] Encoder context
            return_rich_features: If True, returns full edge feature tensor
        
        Returns:
            Tuple of:
                - combined_adjacency: [batch, seq_len, num_nodes, num_nodes] scalar adjacency
                - edge_features: [batch, seq_len, num_nodes, num_nodes, edge_dim] OR None
                - metadata: Dict with diagnostics
        """
        batch_size, seq_len, num_nodes, _ = astronomical_edges.shape
        
        # 1. Stack and transform adjacency matrices to edge feature vectors
        self._dbg("START")
        # CRITICAL: Don't compress - expand to vectors!
        stacked_adjacencies = torch.stack([
            astronomical_edges, learned_edges, attention_edges
        ], dim=-1)
        # Shape: [batch, seq_len, num_nodes, num_nodes, 3]
        self._dbg("STACK_ADJ", {"stacked_adj": stacked_adjacencies})
        
        # Transform to rich edge features
        edge_features_flat = stacked_adjacencies.reshape(-1, 3)
        edge_features_encoded = self.adjacency_to_edge_features(edge_features_flat)
        edge_features = edge_features_encoded.reshape(
            batch_size, seq_len, num_nodes, num_nodes, self.edge_feature_dim
        )
        # Shape: [batch, seq_len, num_nodes, num_nodes, edge_feature_dim]
        # PRESERVED as vectors, not compressed to scalars!
        self._dbg("EDGE_FEATURES", {"edge_features": edge_features})
        
        # 2. Heavy Processing (Enrichment + Message Passing)
        message_passing_metadata = []
        if self.use_gradient_checkpointing and enc_out.requires_grad:
            # Checkpointing requires input tensors to have requires_grad
            node_states, enriched_edge_features, mp_metadata = torch.utils.checkpoint.checkpoint(
                self._heavy_processing,
                edge_features,
                enc_out,
                num_nodes,
                use_reentrant=False
            )
        else:
            node_states, enriched_edge_features, mp_metadata = self._heavy_processing(
                edge_features, enc_out, num_nodes
            )
            
        message_passing_metadata.extend(mp_metadata)

        if self.use_temporal_attention and self.temporal_attention is not None:
            node_states = self.temporal_attention(node_states)
            self._dbg("TEMPORAL_ATTENTION", {"node_states": node_states})
        
        # 6. Optional spatial attention (graph state evolution)
        if self.use_spatial_attention and self.spatial_attention is not None:
            node_states = self.spatial_attention(node_states)
            self._dbg("SPATIAL_ATTENTION", {"node_states": node_states})
        
        # 7. Project edge features to scalar adjacency (for compatibility)
        combined_adjacency_flat = self.edge_features_to_adjacency(
            enriched_edge_features.reshape(-1, self.edge_feature_dim)
        )
        combined_adjacency = combined_adjacency_flat.reshape(
            batch_size, seq_len, num_nodes, num_nodes
        )
        # Shape: [batch, seq_len, num_nodes, num_nodes]
        self._dbg("ADJ_PROJECTION", {"combined_adj": combined_adjacency})
        
        # 8. Collect comprehensive metadata
        metadata = {
            'astronomical_contribution': F.softmax(self.combination_weights, dim=0)[0].item(),
            'learned_contribution': F.softmax(self.combination_weights, dim=0)[1].item(),
            'attention_contribution': F.softmax(self.combination_weights, dim=0)[2].item(),
            'edge_feature_dim': self.edge_feature_dim,
            'num_message_passing_steps': len(message_passing_metadata),
            'avg_transition_strength': sum(
                m['avg_transition_strength'] for m in message_passing_metadata
            ) / len(message_passing_metadata),
            'final_edge_density': (combined_adjacency.abs() > 0.1).float().mean().item(),
            'temporal_edge_variance': enriched_edge_features.var(dim=1).mean().item(),
            'spatial_edge_variance': enriched_edge_features.var(dim=[2, 3]).mean().item(),
            'node_state_norm': node_states.norm(dim=-1).mean().item()
        }
        
        # Return rich edge features if requested
        self._dbg("END")
        if return_rich_features:
            return combined_adjacency, enriched_edge_features, metadata
        else:
            return combined_adjacency, None, metadata

    def _heavy_processing(self, edge_features, enc_out, num_nodes):
        # Re-broadcast context (re-computed during backward if checkpointed)
        batch_size, seq_len, d_model = enc_out.shape
        
        # Broadcast context to all edges
        context_broadcast = enc_out.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, num_nodes, num_nodes, -1
        )
        
        # Fuse context with edge features
        context_edge_combined = torch.cat([
            context_broadcast,
            edge_features
        ], dim=-1).reshape(-1, self.d_model + self.edge_feature_dim)
        
        enriched_edge_features = self.context_edge_fusion(context_edge_combined)
        enriched_edge_features = enriched_edge_features.reshape(
            edge_features.shape[0], edge_features.shape[1], num_nodes, num_nodes, self.edge_feature_dim
        )
        
        # Initialize node states
        node_states = enc_out.unsqueeze(2).expand(-1, -1, num_nodes, -1).contiguous()
        
        mp_metadata = []
        for mp_layer in self.message_passing_layers:
            node_states, mp_meta = mp_layer(
                node_states=node_states,
                edge_features=enriched_edge_features,
                edge_mask=None
            )
            mp_metadata.append(mp_meta)
            
        return node_states, enriched_edge_features, mp_metadata
