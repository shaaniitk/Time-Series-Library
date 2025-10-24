"""
Petri Net Message Passing Layer - Preserves Rich Edge Features

This module implements true Petri net dynamics for celestial graph processing:
- Nodes represent "places" (celestial bodies with state/tokens)
- Edges represent "transitions" (phase relationships, velocity ratios, etc.)
- Message passing represents "token flow" weighted by edge features
- NO information loss - all edge features preserved as vectors

Key Features:
1. Edge features stored as vectors [theta_diff, phi_diff, velocity_diff, ...]
2. Local message aggregation (no global 169×169 attention)
3. Learnable transition functions (how edges control token flow)
4. Memory efficient: O(nodes × neighbors) not O(edges²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from layers.modular.graph.registry import GraphComponentRegistry


@GraphComponentRegistry.register("petri_net_message_passing")
class PetriNetMessagePassing(nn.Module):
    """
    Petri Net Message Passing Layer
    
    Implements token flow through celestial graph where:
    - Nodes = celestial bodies (places with tokens/state)
    - Edges = phase relationships (transitions with firing conditions)
    - Messages = tokens transferred based on edge features
    
    Args:
        num_nodes: Number of celestial bodies (13)
        node_dim: Dimension of node state vectors
        edge_feature_dim: Dimension of edge feature vectors (6+ features)
        message_dim: Dimension of messages passed between nodes
        num_heads: Number of attention heads for local aggregation
        dropout: Dropout probability
        use_local_attention: Whether to use attention for message aggregation
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        edge_feature_dim: int = 6,
        message_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_local_attention: bool = True
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_feature_dim = edge_feature_dim
        self.message_dim = message_dim
        self.num_heads = num_heads
        self.use_local_attention = use_local_attention
        
        # Transition function: determines HOW MUCH information flows
        # Input: edge features [theta_diff, phi_diff, velocity_diff, ...]
        # Output: transition strength (scalar per edge)
        self.transition_strength_net = nn.Sequential(
            nn.Linear(edge_feature_dim, message_dim),
            nn.LayerNorm(message_dim),
            nn.GELU(),
            nn.Linear(message_dim, message_dim // 2),
            nn.GELU(),
            nn.Linear(message_dim // 2, 1),
            nn.Sigmoid()  # Bounded [0, 1] for stability
        )
        
        # Message content function: determines WHAT information flows
        # Input: source node state + edge features
        # Output: message vector
        self.message_content_net = nn.Sequential(
            nn.Linear(node_dim + edge_feature_dim, message_dim),
            nn.LayerNorm(message_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(message_dim, message_dim)
        )
        
        # Edge feature encoder: transforms raw edge features
        # Preserves all information but learns better representation
        self.edge_feature_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, message_dim),
            nn.LayerNorm(message_dim),
            nn.GELU(),
            nn.Linear(message_dim, edge_feature_dim)  # Same dimension out
        )
        
        # Local attention for message aggregation (optional)
        if use_local_attention:
            self.message_attention = nn.MultiheadAttention(
                embed_dim=message_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Aggregation function: combines incoming messages
        self.aggregation_net = nn.Sequential(
            nn.Linear(message_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim)
        )
        
        # Update gate: controls how much new information to accept
        self.update_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),  # old_state + new_info
            nn.Sigmoid()
        )
        
        # Layer norms for stability
        self.node_norm = nn.LayerNorm(node_dim)
        self.message_norm = nn.LayerNorm(message_dim)
        
    def forward(
        self,
        node_states: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Petri net message passing (token flow through graph)
        
        Args:
            node_states: [batch, seq_len, num_nodes, node_dim]
                Current state of each celestial body
            edge_features: [batch, seq_len, num_nodes, num_nodes, edge_feature_dim]
                Rich edge features [theta_diff, phi_diff, velocity_diff, radius_ratio, ...]
            edge_mask: [batch, seq_len, num_nodes, num_nodes] optional
                Mask for valid edges (1 = valid, 0 = invalid)
        
        Returns:
            Tuple of:
                - Updated node states [batch, seq_len, num_nodes, node_dim]
                - Metadata dictionary with diagnostic information
        """
        batch_size, seq_len, num_nodes, node_dim = node_states.shape
        
        # Encode edge features (preserves dimensionality)
        encoded_edge_features = self.edge_feature_encoder(edge_features)
        # Shape: [batch, seq_len, num_nodes, num_nodes, edge_feature_dim]
        
        # Process each target node
        new_node_states = []
        message_strengths_all = []
        
        for target_node_idx in range(num_nodes):
            # Get incoming edges to this target node
            # Shape: [batch, seq_len, num_nodes (sources), edge_feature_dim]
            incoming_edge_features = encoded_edge_features[:, :, :, target_node_idx, :]
            
            # Compute transition strengths (how much each edge fires)
            # Shape: [batch, seq_len, num_nodes (sources), 1]
            transition_strengths = self.transition_strength_net(incoming_edge_features)
            
            # Apply edge mask if provided
            if edge_mask is not None:
                mask = edge_mask[:, :, :, target_node_idx].unsqueeze(-1)
                transition_strengths = transition_strengths * mask
            
            # Compute message content from each source node
            # Concatenate source states with edge features
            source_states_expanded = node_states  # [batch, seq_len, num_nodes, node_dim]
            
            # Expand edge features to match
            edge_feat_for_msg = incoming_edge_features
            # Shape: [batch, seq_len, num_nodes, edge_feature_dim]
            
            # Concatenate for message computation
            # Need to broadcast to align dimensions
            batch_seq_nodes = batch_size * seq_len * num_nodes
            source_states_flat = source_states_expanded.reshape(batch_seq_nodes, node_dim)
            edge_feat_flat = edge_feat_for_msg.reshape(batch_seq_nodes, self.edge_feature_dim)
            
            message_input = torch.cat([source_states_flat, edge_feat_flat], dim=-1)
            # Shape: [batch * seq * num_nodes, node_dim + edge_feature_dim]
            
            # Compute message content
            messages = self.message_content_net(message_input)
            # Shape: [batch * seq * num_nodes, message_dim]
            
            # Reshape back
            messages = messages.reshape(batch_size, seq_len, num_nodes, self.message_dim)
            # Shape: [batch, seq_len, num_nodes (sources), message_dim]
            
            # Apply transition strengths (token flow controlled by edge features)
            weighted_messages = messages * transition_strengths
            # Shape: [batch, seq_len, num_nodes, message_dim]
            
            # Normalize messages
            weighted_messages = self.message_norm(weighted_messages)
            
            # Aggregate messages (local operation, only num_nodes neighbors)
            if self.use_local_attention:
                # Attention over source nodes (13 neighbors, not 169 edges!)
                # Reshape for attention
                msgs_for_attn = weighted_messages.reshape(
                    batch_size * seq_len, num_nodes, self.message_dim
                )
                
                # Self-attention over source nodes
                aggregated_msgs, attn_weights = self.message_attention(
                    msgs_for_attn, msgs_for_attn, msgs_for_attn
                )
                # Attention matrix: [batch*seq, num_nodes, num_nodes] = [2000, 13, 13]
                # Much smaller than [2000, 169, 169]!
                
                # Take mean across sources (or could use attention-weighted sum)
                aggregated_msgs = aggregated_msgs.mean(dim=1)  # [batch*seq, message_dim]
                aggregated_msgs = aggregated_msgs.reshape(batch_size, seq_len, self.message_dim)
            else:
                # Simple aggregation (even more efficient)
                aggregated_msgs = weighted_messages.mean(dim=2)
                # Shape: [batch, seq_len, message_dim]
            
            # Transform aggregated messages to node dimension
            new_information = self.aggregation_net(aggregated_msgs)
            # Shape: [batch, seq_len, node_dim]
            
            # Get current node state
            current_state = node_states[:, :, target_node_idx, :]
            # Shape: [batch, seq_len, node_dim]
            
            # Compute update gate (how much to incorporate new info)
            gate_input = torch.cat([current_state, new_information], dim=-1)
            update_gate = self.update_gate(gate_input)
            # Shape: [batch, seq_len, node_dim]
            
            # Update node state (Petri net token update)
            updated_state = update_gate * new_information + (1 - update_gate) * current_state
            updated_state = self.node_norm(updated_state)
            
            new_node_states.append(updated_state)
            message_strengths_all.append(transition_strengths.mean().item())
        
        # Stack all updated node states
        new_node_states = torch.stack(new_node_states, dim=2)
        # Shape: [batch, seq_len, num_nodes, node_dim]
        
        # Collect metadata
        metadata = {
            'avg_transition_strength': sum(message_strengths_all) / len(message_strengths_all),
            'num_nodes': num_nodes,
            'message_passing_type': 'local_aggregation',
            'edge_features_preserved': True,
            'edge_feature_dim': self.edge_feature_dim
        }
        
        return new_node_states, metadata


@GraphComponentRegistry.register("temporal_node_attention")
class TemporalNodeAttention(nn.Module):
    """
    Temporal attention over each node's history
    
    Captures delayed effects: "Moon's state 10 timesteps ago affects today"
    Attention over time (seq_len), not over edges
    """
    
    def __init__(
        self,
        node_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_states: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal self-attention to each node independently
        
        Args:
            node_states: [batch, seq_len, num_nodes, node_dim]
        
        Returns:
            Temporally attended states [batch, seq_len, num_nodes, node_dim]
        """
        batch_size, seq_len, num_nodes, node_dim = node_states.shape
        
        # Process each node's temporal history
        attended_states = []
        
        for node_idx in range(num_nodes):
            node_history = node_states[:, :, node_idx, :]
            # Shape: [batch, seq_len, node_dim]
            
            # Self-attention over time
            attended, _ = self.temporal_attention(
                node_history, node_history, node_history
            )
            # Attention matrix: [batch, seq_len, seq_len]
            # e.g., [8, 250, 250] = 500K elements (manageable!)
            
            # Residual + norm
            attended = self.norm(node_history + self.dropout(attended))
            attended_states.append(attended)
        
        # Stack back
        attended_states = torch.stack(attended_states, dim=2)
        # Shape: [batch, seq_len, num_nodes, node_dim]
        
        return attended_states


@GraphComponentRegistry.register("spatial_graph_attention")
class SpatialGraphAttention(nn.Module):
    """
    Spatial attention over nodes (graph state)
    
    Captures global patterns: "Mars's current state modulates Venus-Jupiter relationship"
    Attention over nodes (13), not edges (169)
    """
    
    def __init__(
        self,
        node_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_states: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial self-attention across nodes at each timestep
        
        Args:
            node_states: [batch, seq_len, num_nodes, node_dim]
        
        Returns:
            Spatially attended states [batch, seq_len, num_nodes, node_dim]
        """
        batch_size, seq_len, num_nodes, node_dim = node_states.shape
        
        # Reshape for attention over nodes
        states_reshaped = node_states.reshape(batch_size * seq_len, num_nodes, node_dim)
        
        # Self-attention over nodes
        attended, _ = self.spatial_attention(
            states_reshaped, states_reshaped, states_reshaped
        )
        # Attention matrix: [batch*seq, num_nodes, num_nodes]
        # e.g., [2000, 13, 13] = 338K elements (tiny!)
        
        # Residual + norm
        attended = self.norm(states_reshaped + self.dropout(attended))
        
        # Reshape back
        attended = attended.reshape(batch_size, seq_len, num_nodes, node_dim)
        
        return attended
