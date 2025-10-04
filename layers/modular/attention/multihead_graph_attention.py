import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-Head Graph Attention mechanism for heterogeneous graphs
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Separate projections for each node type
        self.node_projections = nn.ModuleDict({
            'wave': nn.ModuleDict({
                'query': nn.Linear(d_model, d_model),
                'key': nn.Linear(d_model, d_model),
                'value': nn.Linear(d_model, d_model)
            }),
            'transition': nn.ModuleDict({
                'query': nn.Linear(d_model, d_model),
                'key': nn.Linear(d_model, d_model),
                'value': nn.Linear(d_model, d_model)
            }),
            'target': nn.ModuleDict({
                'query': nn.Linear(d_model, d_model),
                'key': nn.Linear(d_model, d_model),
                'value': nn.Linear(d_model, d_model)
            })
        })
        
        # Edge type specific attention weights
        self.edge_attention = nn.ModuleDict({
            'wave_to_transition': nn.Linear(d_model * 2, num_heads),
            'transition_to_target': nn.Linear(d_model * 2, num_heads)
        })
        
        # Output projections
        self.output_projections = nn.ModuleDict({
            'wave': nn.Linear(d_model, d_model),
            'transition': nn.Linear(d_model, d_model),
            'target': nn.Linear(d_model, d_model)
        })
        
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.ModuleDict({
            'wave': nn.LayerNorm(d_model),
            'transition': nn.LayerNorm(d_model),
            'target': nn.LayerNorm(d_model)
        })
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_weights: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of multi-head graph attention
        
        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each edge type
            edge_weights: Optional edge weights
            
        Returns:
            Updated node features
        """
        batch_size = 1  # Assuming single batch for graph processing
        
        # Store original features for residual connections
        residual_dict = {node_type: x.clone() for node_type, x in x_dict.items()}
        
        # Process each edge type
        updated_features = {node_type: [] for node_type in x_dict.keys()}
        
        # Wave to Transition attention
        if ('wave', 'interacts_with', 'transition') in edge_index_dict:
            wave_to_trans_output = self._compute_attention(
                x_dict['wave'], x_dict['transition'],
                edge_index_dict[('wave', 'interacts_with', 'transition')],
                'wave', 'transition', 'wave_to_transition',
                edge_weights.get(('wave', 'interacts_with', 'transition')) if edge_weights else None
            )
            updated_features['transition'].append(wave_to_trans_output)
        
        # Transition to Target attention
        if ('transition', 'influences', 'target') in edge_index_dict:
            trans_to_target_output = self._compute_attention(
                x_dict['transition'], x_dict['target'],
                edge_index_dict[('transition', 'influences', 'target')],
                'transition', 'target', 'transition_to_target',
                edge_weights.get(('transition', 'influences', 'target')) if edge_weights else None
            )
            updated_features['target'].append(trans_to_target_output)
        
        # Aggregate and apply residual connections
        output_dict = {}
        for node_type, features_list in updated_features.items():
            if features_list:
                # Aggregate multiple attention outputs
                aggregated = torch.stack(features_list, dim=0).mean(dim=0)
                # Apply output projection
                projected = self.output_projections[node_type](aggregated)
                # Residual connection and layer norm
                output_dict[node_type] = self.layer_norm[node_type](
                    residual_dict[node_type] + self.dropout_layer(projected)
                )
            else:
                # No incoming edges, just apply layer norm to original features
                output_dict[node_type] = self.layer_norm[node_type](residual_dict[node_type])
        
        return output_dict
    
    def _compute_attention(self, source_x: torch.Tensor, target_x: torch.Tensor, 
                          edge_index: torch.Tensor, source_type: str, target_type: str,
                          edge_type: str, edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention between source and target nodes
        """
        # CRITICAL FIX: Correct edge index convention - edge_index[0] = target, edge_index[1] = source
        target_idx, source_idx = edge_index[0], edge_index[1]
        
        # Get queries, keys, values
        queries = self.node_projections[target_type]['query'](target_x)  # Target nodes as queries
        keys = self.node_projections[source_type]['key'](source_x)       # Source nodes as keys
        values = self.node_projections[source_type]['value'](source_x)   # Source nodes as values
        
        # Reshape for multi-head attention
        queries = queries.view(-1, self.num_heads, self.d_k)  # [num_target_nodes, num_heads, d_k]
        keys = keys.view(-1, self.num_heads, self.d_k)        # [num_source_nodes, num_heads, d_k]
        values = values.view(-1, self.num_heads, self.d_k)    # [num_source_nodes, num_heads, d_k]
        
        # Gather features for edges
        edge_queries = queries[target_idx]  # [num_edges, num_heads, d_k]
        edge_keys = keys[source_idx]        # [num_edges, num_heads, d_k]
        edge_values = values[source_idx]    # [num_edges, num_heads, d_k]
        
        # Compute attention scores
        attention_scores = torch.sum(edge_queries * edge_keys, dim=-1) / math.sqrt(self.d_k)  # [num_edges, num_heads]
        
        # Add edge-specific attention bias
        edge_features = torch.cat([source_x[source_idx], target_x[target_idx]], dim=-1)
        edge_bias = self.edge_attention[edge_type](edge_features)  # [num_edges, num_heads]
        attention_scores = attention_scores + edge_bias
        
        # Apply edge weights if provided
        if edge_weights is not None:
            attention_scores = attention_scores * edge_weights.unsqueeze(-1)
        
        # Softmax over source nodes for each target node
        attention_weights = self._edge_softmax(attention_scores, target_idx, target_x.size(0))
        
        # Apply attention to values
        attended_values = attention_weights.unsqueeze(-1) * edge_values  # [num_edges, num_heads, d_k]
        
        # Aggregate by target nodes
        output = torch.zeros(target_x.size(0), self.num_heads, self.d_k, device=target_x.device)
        output.index_add_(0, target_idx, attended_values)
        
        # Reshape back to original dimensions
        output = output.view(target_x.size(0), -1)  # [num_target_nodes, d_model]
        
        return output
    
    def _edge_softmax(self, attention_scores: torch.Tensor, target_idx: torch.Tensor, num_targets: int) -> torch.Tensor:
        """
        Apply softmax over edges grouped by target nodes
        """
        # Create a large negative value for masking
        max_val = attention_scores.max().item()
        
        # Initialize output
        softmax_scores = torch.zeros_like(attention_scores)
        
        # Apply softmax for each target node
        for target_id in range(num_targets):
            mask = target_idx == target_id
            if mask.any():
                target_scores = attention_scores[mask]
                target_softmax = F.softmax(target_scores, dim=0)
                softmax_scores[mask] = target_softmax
        
        return softmax_scores


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer combining multi-head attention with feed-forward networks
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadGraphAttention(d_model, num_heads, dropout)
        
        # Feed-forward networks for each node type
        self.feed_forward = nn.ModuleDict({
            'wave': self._create_ffn(d_model, d_ff, dropout),
            'transition': self._create_ffn(d_model, d_ff, dropout),
            'target': self._create_ffn(d_model, d_ff, dropout)
        })
        
        self.layer_norm = nn.ModuleDict({
            'wave': nn.LayerNorm(d_model),
            'transition': nn.LayerNorm(d_model),
            'target': nn.LayerNorm(d_model)
        })
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_ffn(self, d_model: int, d_ff: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_weights: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of graph transformer layer
        """
        # Runtime validations: node features and edge indices
        required_nodes = ['wave', 'transition', 'target']
        for nt in required_nodes:
            if nt not in x_dict:
                raise ValueError(f"Missing node type '{nt}' in x_dict")
        # Validate shapes and d_model consistency
        d_model = getattr(self.multi_head_attention, 'd_model', None)
        for node_type, x in x_dict.items():
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"x_dict['{node_type}'] must be a Tensor, got {type(x)}")
            if x.dim() != 2:
                raise ValueError(f"x_dict['{node_type}'] must be 2D [num_nodes, d_model], got shape {tuple(x.shape)}")
            if d_model is not None and x.size(1) != d_model:
                raise ValueError(f"x_dict['{node_type}'] last dim must equal d_model={d_model}, got {x.size(1)}")
        
        def _validate_edges(key, src_len, tgt_len):
            if key in edge_index_dict:
                ei = edge_index_dict[key]
                if not isinstance(ei, torch.Tensor):
                    raise ValueError(f"edge_index for {key} must be a Tensor, got {type(ei)}")
                if ei.dtype not in (torch.long, torch.int64):
                    raise ValueError(f"edge_index for {key} must be of dtype torch.long, got {ei.dtype}")
                if ei.dim() != 2 or ei.size(0) != 2:
                    raise ValueError(f"edge_index for {key} must have shape [2, E], got {tuple(ei.shape)}")
                if ei.numel() > 0:
                    max_src = int(ei[0].max().item())
                    max_tgt = int(ei[1].max().item())
                    min_src = int(ei[0].min().item())
                    min_tgt = int(ei[1].min().item())
                    if min_src < 0 or min_tgt < 0:
                        raise ValueError(f"edge_index for {key} contains negative indices")
                    if max_src >= src_len:
                        raise ValueError(f"edge_index source index out of bounds for {key}: max {max_src} >= {src_len}")
                    if max_tgt >= tgt_len:
                        raise ValueError(f"edge_index target index out of bounds for {key}: max {max_tgt} >= {tgt_len}")
        
        _validate_edges(('wave', 'interacts_with', 'transition'), x_dict['wave'].size(0), x_dict['transition'].size(0))
        _validate_edges(('transition', 'influences', 'target'), x_dict['transition'].size(0), x_dict['target'].size(0))        # Multi-head attention
        attention_output = self.multi_head_attention(x_dict, edge_index_dict, edge_weights)
        
        # Feed-forward networks with residual connections
        output_dict = {}
        for node_type, features in attention_output.items():
            # Feed-forward
            ff_output = self.feed_forward[node_type](features)
            # Residual connection and layer norm
            output_dict[node_type] = self.layer_norm[node_type](features + self.dropout(ff_output))
        
        return output_dict
