"""Graph attention mechanisms for time series analysis.

This module provides graph attention layers and utility functions for constructing
various types of graphs used in time series modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import kneighbors_graph


class MultiGraphAttention(nn.Module):
    """Multi-graph attention mechanism.
    
    Applies attention over multiple graph structures simultaneously.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, n_graphs: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_graphs = n_graphs
        self.d_k = d_model // n_heads
        
        # Separate projections for each graph type
        self.graph_projections = nn.ModuleList([
            nn.ModuleDict({
                'query': nn.Linear(d_model, d_model),
                'key': nn.Linear(d_model, d_model),
                'value': nn.Linear(d_model, d_model)
            }) for _ in range(n_graphs)
        ])
        
        self.output_projection = nn.Linear(d_model * n_graphs, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, adjacency_matrices: list) -> torch.Tensor:
        """Forward pass with multiple adjacency matrices.
        
        Args:
            x: Input tensor [B, N, D]
            adjacency_matrices: List of adjacency matrices [B, N, N]
        """
        B, N, D = x.shape
        residual = x
        
        graph_outputs = []
        
        for i, adj_matrix in enumerate(adjacency_matrices[:self.n_graphs]):
            # Get projections for this graph
            proj = self.graph_projections[i]
            
            q = proj['query'](x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            k = proj['key'](x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            v = proj['value'](x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
            
            # Apply graph structure mask
            if adj_matrix is not None:
                mask = adj_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Apply attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply to values
            graph_out = torch.matmul(attn_weights, v)
            graph_out = graph_out.transpose(1, 2).contiguous().view(B, N, D)
            
            graph_outputs.append(graph_out)
        
        # Combine outputs from all graphs
        combined = torch.cat(graph_outputs, dim=-1)
        output = self.output_projection(combined)
        
        return self.layer_norm(output + residual)


def construct_correlation_graph(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Construct correlation-based adjacency matrix.
    
    Args:
        x: Input tensor [B, N, T, D] where N is number of variables, T is time steps
        threshold: Correlation threshold for edge creation
        
    Returns:
        Adjacency matrix [B, N, N]
    """
    B, N, T, D = x.shape
    
    # Flatten time and feature dimensions for correlation computation
    x_flat = x.view(B, N, -1)  # [B, N, T*D]
    
    # Compute correlation matrix
    x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
    x_norm = F.normalize(x_centered, p=2, dim=-1)
    
    # Correlation matrix
    corr_matrix = torch.bmm(x_norm, x_norm.transpose(-2, -1))  # [B, N, N]
    
    # Apply threshold to create binary adjacency matrix
    adj_matrix = (torch.abs(corr_matrix) > threshold).float()
    
    # Remove self-loops
    eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
    adj_matrix = adj_matrix * (1 - eye)
    
    return adj_matrix


def construct_knn_graph(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Construct k-nearest neighbor graph.
    
    Args:
        x: Input tensor [B, N, D] where N is number of nodes
        k: Number of nearest neighbors
        
    Returns:
        Adjacency matrix [B, N, N]
    """
    B, N, D = x.shape
    
    # Compute pairwise distances
    x_expanded_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
    x_expanded_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
    
    distances = torch.norm(x_expanded_i - x_expanded_j, p=2, dim=-1)  # [B, N, N]
    
    # Find k nearest neighbors for each node
    _, knn_indices = torch.topk(distances, k + 1, dim=-1, largest=False)  # +1 to exclude self
    
    # Create adjacency matrix
    adj_matrix = torch.zeros_like(distances)
    
    for b in range(B):
        for i in range(N):
            # Skip the first index (self)
            neighbors = knn_indices[b, i, 1:k+1]
            adj_matrix[b, i, neighbors] = 1.0
    
    # Make symmetric
    adj_matrix = torch.max(adj_matrix, adj_matrix.transpose(-2, -1))
    
    return adj_matrix


def construct_temporal_correlation_graph(x: torch.Tensor, lag: int = 1, threshold: float = 0.3) -> torch.Tensor:
    """Construct temporal correlation graph based on lagged correlations.
    
    Args:
        x: Input tensor [B, N, T, D] where N is variables, T is time steps
        lag: Time lag for correlation computation
        threshold: Correlation threshold for edge creation
        
    Returns:
        Adjacency matrix [B, N, N]
    """
    B, N, T, D = x.shape
    
    if T <= lag:
        # If not enough time steps, return empty graph
        return torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
    
    # Compute lagged correlations
    x_current = x[:, :, lag:, :].reshape(B, N, -1)  # [B, N, (T-lag)*D]
    x_lagged = x[:, :, :-lag, :].reshape(B, N, -1)  # [B, N, (T-lag)*D]
    
    # Normalize
    x_current = F.normalize(x_current - x_current.mean(dim=-1, keepdim=True), p=2, dim=-1)
    x_lagged = F.normalize(x_lagged - x_lagged.mean(dim=-1, keepdim=True), p=2, dim=-1)
    
    # Compute cross-correlation matrix between current and lagged variables
    corr_matrix = torch.bmm(x_current, x_lagged.transpose(-2, -1))  # [B, N, N]
    
    # Apply threshold
    adj_matrix = (torch.abs(corr_matrix) > threshold).float()
    
    return adj_matrix


# Export all functions and classes
__all__ = [
    'MultiGraphAttention',
    'construct_correlation_graph',
    'construct_knn_graph', 
    'construct_temporal_correlation_graph'
]