import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Simplified scatter_softmax implementation for Windows compatibility
def scatter_softmax(src, index, dim=0, dim_size=None):
    """Simplified scatter softmax implementation"""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output tensor
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    
    # Apply softmax per group
    for i in range(dim_size):
        mask = (index == i)
        if mask.any():
            group_src = src[mask]
            group_softmax = F.softmax(group_src, dim=0)
            out[i] = group_softmax.sum(dim=0)
    
    return out

# Simplified MessagePassing base class
class MessagePassing(nn.Module):
    """Simplified MessagePassing implementation"""
    def __init__(self, aggr='add'):
        super().__init__()
        self.aggr = aggr
    
    def propagate(self, edge_index, **kwargs):
        """Simplified propagate method"""
        row, col = edge_index
        
        # Prepare message arguments
        msg_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, tuple) and len(value) == 2:
                msg_kwargs[f'{key}_i'] = value[1][row]  # target nodes
                msg_kwargs[f'{key}_j'] = value[0][col]  # source nodes
        
        msg_kwargs['index'] = row
        msg_kwargs['ptr'] = None
        msg_kwargs['size_i'] = None
        
        # Compute messages
        messages = self.message(**msg_kwargs)
        
        # Aggregate messages
        if self.aggr == 'add':
            aggr_out = torch.zeros(kwargs['x'][1].size(0), messages.size(-1), 
                                 dtype=messages.dtype, device=messages.device)
            aggr_out.index_add_(0, row, messages)
        
        return self.update(aggr_out)
    
    def message(self, **kwargs):
        raise NotImplementedError
    
    def update(self, aggr_out):
        return aggr_out

class CrossAttentionGNNConv(MessagePassing):
    """The custom GNN layer that performs cross-attention between feature (x) and topology (t) modalities."""
    def __init__(self, d_model: int):
        super().__init__(aggr='add')
        self.d_model = d_model
        
        # Layers for the FEATURE modality update, weighted by TOPOLOGY attention
        self.W_x = nn.Linear(d_model, d_model, bias=False)
        self.K_alpha = nn.Linear(d_model, d_model); self.Q_alpha = nn.Linear(d_model, d_model)
        
        # Layers for the TOPOLOGY modality update, weighted by FEATURE attention
        self.W_t = nn.Linear(d_model, d_model, bias=False)
        self.K_beta = nn.Linear(d_model, d_model); self.Q_beta = nn.Linear(d_model, d_model)
        
    def forward(self, x, t, edge_index):
        # x and t are tuples: (source_nodes_data, target_nodes_data)
        return self.propagate(edge_index, x=x, t=t)

    def message(self, x_i, x_j, t_i, t_j, index, ptr, size_i):
        # Feature Attention (alpha), computed from Topology vectors
        q_alpha = self.Q_alpha(t_i); k_alpha = self.K_alpha(t_j)
        alpha_scores = torch.sum(q_alpha * k_alpha, dim=-1) / (self.d_model**0.5)
        alpha = scatter_softmax(alpha_scores, index, dim=0)

        # Topology Attention (beta), computed from Feature vectors
        q_beta = self.Q_beta(x_i); k_beta = self.K_beta(x_j)
        beta_scores = torch.sum(q_beta * k_beta, dim=-1) / (self.d_model**0.5)
        beta = scatter_softmax(beta_scores, index, dim=0)

        # Create two distinct messages, weighted by the cross-attention scores
        msg_for_t = alpha.unsqueeze(-1) * self.W_t(t_j)
        msg_for_x = beta.unsqueeze(-1) * self.W_x(x_j)
        
        return torch.cat([msg_for_x, msg_for_t], dim=-1)

    def update(self, aggr_out):
        # The update is simple as aggregation is handled by propagate
        return torch.chunk(aggr_out, 2, dim=-1)


class PGAT_CrossAttn_Layer(nn.Module):
    """A wrapper to apply the cross-attention GNN to our heterogeneous Petri Net graph."""
    def __init__(self, d_model):
        super().__init__()
        self.conv_wave_to_trans = CrossAttentionGNNConv(d_model)
        self.conv_trans_to_target = CrossAttentionGNNConv(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_dict, t_dict, edge_index_dict):
        # 1. Message passing from waves to transitions
        x_trans_update, t_trans_update = self.conv_wave_to_trans(
            x=(x_dict['wave'], x_dict['transition']),
            t=(t_dict['wave'], t_dict['transition']),
            edge_index=edge_index_dict[('wave', 'interacts_with', 'transition')]
        )
        x_trans = self.norm(x_dict['transition'] + F.relu(x_trans_update))
        t_trans = self.norm(t_dict['transition'] + F.relu(t_trans_update))
        
        # 2. Message passing from updated transitions to targets
        x_target_update, t_target_update = self.conv_trans_to_target(
            x=(x_trans, x_dict['target']),
            t=(t_trans, t_dict['target']),
            edge_index=edge_index_dict[('transition', 'influences', 'target')]
        )
        x_target = self.norm(x_dict['target'] + F.relu(x_target_update))
        t_target = self.norm(t_dict['target'] + F.relu(t_target_update))
        
        return {'wave': x_dict['wave'], 'transition': x_trans, 'target': x_target}, \
               {'wave': t_dict['wave'], 'transition': t_trans, 'target': t_target}