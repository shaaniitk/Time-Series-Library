import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from layers.modular.graph.conv import CrossAttentionGNNConv, MessagePassing

class DynamicEdgeWeightComputation(nn.Module):
    """
    Learnable edge weight computation for dynamic graph attention.
    Computes edge weights based on node features and graph structure.
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        # Edge weight computation networks
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_heads),
            nn.Sigmoid()  # Ensure weights are in [0, 1]
        )
        
        # Attention-based edge weight computation
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Structural importance weighting
        self.structural_weight = nn.Parameter(torch.ones(1))
        self.feature_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, source_features, target_features, edge_index):
        """
        Compute dynamic edge weights based on node features.
        
        Args:
            source_features: Source node features [num_source_nodes, d_model]
            target_features: Target node features [num_target_nodes, d_model]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            edge_weights: Computed edge weights [num_edges, num_heads]
        """
        # Get source and target node features for each edge
        source_idx, target_idx = edge_index[0], edge_index[1]
        edge_source_features = source_features[source_idx]  # [num_edges, d_model]
        edge_target_features = target_features[target_idx]  # [num_edges, d_model]
        
        # Concatenate source and target features
        edge_features = torch.cat([edge_source_features, edge_target_features], dim=-1)  # [num_edges, 2*d_model]
        
        # Compute MLP-based edge weights
        mlp_weights = self.edge_weight_mlp(edge_features)  # [num_edges, num_heads]
        
        # Compute attention-based edge weights
        queries = self.query_proj(edge_source_features)  # [num_edges, d_model]
        keys = self.key_proj(edge_target_features)  # [num_edges, d_model]
        
        # Reshape for multi-head attention
        queries = queries.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
        keys = keys.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
        
        # Compute attention scores
        attention_scores = torch.sum(queries * keys, dim=-1) / (self.head_dim ** 0.5)  # [num_edges, num_heads]
        attention_weights = torch.sigmoid(attention_scores)  # [num_edges, num_heads]
        
        # Combine MLP and attention weights
        combined_weights = (self.structural_weight * mlp_weights + 
                          self.feature_weight * attention_weights)
        
        # Normalize weights
        combined_weights = F.softmax(combined_weights, dim=0)
        
        return combined_weights

class EnhancedCrossAttentionGNNConv(MessagePassing):
    """
    Enhanced CrossAttentionGNNConv with dynamic edge weights.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__(aggr='add')
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Original attention components
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_t = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)
        
        # Dynamic edge weight computation
        self.edge_weight_computer = DynamicEdgeWeightComputation(d_model, num_heads)
        
    def forward(self, x, t, edge_index):
        x_source, x_target = x
        t_source, t_target = t
        
        # Compute dynamic edge weights
        edge_weights = self.edge_weight_computer(x_source, x_target, edge_index)
        
        # Propagate with dynamic edge weights
        x_out, t_out = self.propagate(
            edge_index, 
            x=(x_source, x_target), 
            t=(t_source, t_target),
            edge_weights=edge_weights
        )
        return x_out, t_out
    
    def message(self, x_i, x_j, t_i, t_j, index, ptr, size_i, edge_weights):
        # Compute queries, keys, values
        q = self.W_q(x_i).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(x_j).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(x_j).view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.sum(q * k, dim=-1) / (self.head_dim ** 0.5)  # [num_edges, num_heads]
        
        # Apply dynamic edge weights
        weighted_scores = scores * edge_weights  # [num_edges, num_heads]
        
        # Apply softmax normalization
        from layers.modular.graph.conv import scatter_softmax
        attn_weights = scatter_softmax(weighted_scores, index, dim=0, dim_size=size_i)
        
        # Apply attention to values
        attn_weights = attn_weights.unsqueeze(-1)  # [num_edges, num_heads, 1]
        weighted_v = attn_weights * v  # [num_edges, num_heads, head_dim]
        
        # Reshape and project
        weighted_v = weighted_v.view(-1, self.d_model)  # [num_edges, d_model]
        x_message = self.W_out(weighted_v)
        
        # Topology message
        t_message = self.W_t(t_j)
        
        return x_message, t_message
    
    def update(self, aggr_out):
        return aggr_out

class EnhancedPGAT_CrossAttn_Layer(nn.Module):
    """
    Enhanced PGAT Cross Attention Layer with dynamic edge weights.
    """
    
    def __init__(self, d_model, num_heads=4, use_dynamic_weights=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_dynamic_weights = use_dynamic_weights
        
        if use_dynamic_weights:
            # Use enhanced convolutions with dynamic edge weights
            self.conv_wave_to_trans = EnhancedCrossAttentionGNNConv(d_model, num_heads)
            self.conv_trans_to_target = EnhancedCrossAttentionGNNConv(d_model, num_heads)
        else:
            # Fallback to original convolutions
            self.conv_wave_to_trans = CrossAttentionGNNConv(d_model)
            self.conv_trans_to_target = CrossAttentionGNNConv(d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
        # Additional learnable parameters for edge weight adaptation
        self.edge_weight_adaptation = nn.Parameter(torch.ones(1))
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x_dict, t_dict, edge_index_dict):
        """
        Forward pass with enhanced dynamic edge weight computation.
        
        Args:
            x_dict: Node features dictionary
            t_dict: Topology features dictionary  
            edge_index_dict: Edge connectivity dictionary
            
        Returns:
            Updated node and topology features
        """
        # 1. Message passing from waves to transitions with dynamic weights
        x_trans_update, t_trans_update = self.conv_wave_to_trans(
            x=(x_dict['wave'], x_dict['transition']),
            t=(t_dict['wave'], t_dict['transition']),
            edge_index=edge_index_dict[('wave', 'interacts_with', 'transition')]
        )
        
        # Apply adaptive residual connections
        x_trans = self.norm(
            self.residual_weight * x_dict['transition'] + 
            self.edge_weight_adaptation * F.relu(x_trans_update)
        )
        t_trans = self.norm(
            self.residual_weight * t_dict['transition'] + 
            self.edge_weight_adaptation * F.relu(t_trans_update)
        )
        
        # 2. Message passing from updated transitions to targets with dynamic weights
        x_target_update, t_target_update = self.conv_trans_to_target(
            x=(x_trans, x_dict['target']),
            t=(t_trans, t_dict['target']),
            edge_index=edge_index_dict[('transition', 'influences', 'target')]
        )
        
        # Apply adaptive residual connections
        x_target = self.norm(
            self.residual_weight * x_dict['target'] + 
            self.edge_weight_adaptation * F.relu(x_target_update)
        )
        t_target = self.norm(
            self.residual_weight * t_dict['target'] + 
            self.edge_weight_adaptation * F.relu(t_target_update)
        )
        
        return {'wave': x_dict['wave'], 'transition': x_trans, 'target': x_target}, \
               {'wave': t_dict['wave'], 'transition': t_trans, 'target': t_target}
    
    def get_edge_weights(self, x_dict, edge_index_dict):
        """
        Get the computed dynamic edge weights for analysis.
        
        Returns:
            Dictionary of edge weights for each edge type
        """
        edge_weights = {}
        
        if self.use_dynamic_weights:
            # Get edge weights for wave-to-transition edges
            wave_to_trans_weights = self.conv_wave_to_trans.edge_weight_computer(
                x_dict['wave'], x_dict['transition'],
                edge_index_dict[('wave', 'interacts_with', 'transition')]
            )
            edge_weights[('wave', 'interacts_with', 'transition')] = wave_to_trans_weights
            
            # Get edge weights for transition-to-target edges
            trans_to_target_weights = self.conv_trans_to_target.edge_weight_computer(
                x_dict['transition'], x_dict['target'],
                edge_index_dict[('transition', 'influences', 'target')]
            )
            edge_weights[('transition', 'influences', 'target')] = trans_to_target_weights
        
        return edge_weights