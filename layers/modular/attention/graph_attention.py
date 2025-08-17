"""
Graph Attention Network (GAT) implementation for time series data.
Integrates graph attention mechanisms with temporal sequence processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementing GAT mechanism for time series data.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        alpha: LeakyReLU negative slope
        concat: Whether to concatenate or average multi-head outputs
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, 
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.concat = concat
        
        if concat:
            self.d_k = d_model // n_heads
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads when concat=True"
        else:
            self.d_k = d_model
            
        # Linear transformations for each head
        self.W = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.d_k, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        if not concat:
            self.out_proj = nn.Linear(self.d_k, d_model)
            
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Graph Attention Layer.
        
        Args:
            x: Input tensor [B, L, D] or [B, N, D]
            adj_matrix: Adjacency matrix [B, N, N] or [N, N]
            attn_mask: Attention mask
            
        Returns:
            output: Transformed features
            attention: Attention weights
        """
        # Handle both 3D [B, N, D] and 4D [B, L, N, D] inputs
        if x.dim() == 4:
            B_orig, L, N, D = x.shape
            # Reshape to [B*L, N, D] for processing
            x = x.view(B_orig * L, N, D)
            B = B_orig * L
            process_4d = True
        else:
            B, N, D = x.shape
            B_orig, L = B, 1
            process_4d = False
        
        # Linear transformation
        h = self.W(x).view(B, N, self.n_heads, self.d_k)  # [B, N, H, D_k]
        
        # Compute attention coefficients
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, N, N, H, D_k]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, N, H, D_k]
        
        # Concatenate for attention computation
        edge_h = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, H, 2*D_k]
        
        # Apply attention mechanism
        # edge_h: [B, N, N, H, 2*D_k], self.a: [H, 2*D_k, 1]
        edge_e = torch.einsum('bijhd,hd->bijh', edge_h, self.a.squeeze(-1))  # [B, N, N, H]
        edge_e = self.leakyrelu(edge_e)
        
        # Apply adjacency matrix mask if provided
        if adj_matrix is not None:
            if adj_matrix.dim() == 2:
                adj_matrix = adj_matrix.unsqueeze(0).expand(B, -1, -1)
            elif adj_matrix.dim() == 3 and process_4d:
                # Expand adjacency matrix for all time steps
                adj_matrix = adj_matrix.unsqueeze(1).expand(-1, L, -1, -1).contiguous().view(B, N, N)
            
            # Ensure adjacency matrix matches the batch size and node dimensions
            if adj_matrix.size(0) != B or adj_matrix.size(1) != N or adj_matrix.size(2) != N:
                # Reshape or expand adjacency matrix to match current tensor dimensions
                adj_matrix = adj_matrix[:B, :N, :N] if adj_matrix.size(0) >= B and adj_matrix.size(1) >= N and adj_matrix.size(2) >= N else adj_matrix
                if adj_matrix.size(0) < B:
                    adj_matrix = adj_matrix.expand(B, -1, -1)
                if adj_matrix.size(1) < N or adj_matrix.size(2) < N:
                    # Pad or truncate to match N
                    adj_matrix = F.pad(adj_matrix, (0, max(0, N - adj_matrix.size(2)), 0, max(0, N - adj_matrix.size(1))), value=0)
                    adj_matrix = adj_matrix[:, :N, :N]
            
            # Apply mask: edge_e [B, N, N, H], adj_matrix [B, N, N] -> [B, N, N, 1]
            edge_e = edge_e.masked_fill(adj_matrix.unsqueeze(-1) == 0, -1e9)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            edge_e = edge_e.masked_fill(attn_mask.unsqueeze(-1) == 0, -1e9)
            
        # Compute attention weights
        attention = F.softmax(edge_e, dim=2)  # [B, N, N, H]
        attention = self.dropout(attention)
        
        # Apply attention to features
        # attention: [B, N, N, H], h: [B, N, H, D_k]
        h_prime = torch.einsum('bijh,bjhd->bihd', attention, h)  # [B, N, H, D_k]
        
        if self.concat:
            output = h_prime.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, H*D_k]
        else:
            output = h_prime.mean(dim=1)  # [B, N, D_k]
            output = self.out_proj(output)  # [B, N, D]
        
        # Reshape back to original format if needed
        if process_4d:
            output = output.view(B_orig, L, N, -1)
            # attention: [B, N, N, H] -> [B_orig, L, N, N, H] -> [B_orig, L, N, N]
            attention = attention.mean(dim=-1)  # Average over heads first: [B, N, N]
            attention = attention.view(B_orig, L, N, N)  # Then reshape
        else:
            attention = attention.mean(dim=-1)  # Return mean attention across heads
            
        return output, attention


class MultiGraphAttention(nn.Module):
    """
    Multi-layer Graph Attention with residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.gat = GraphAttentionLayer(d_model, n_heads, dropout, concat=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connection and normalization."""
        residual = x
        x, attention = self.gat(x, adj_matrix, attn_mask)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x, attention


def construct_correlation_graph(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Construct adjacency matrix based on feature correlations.
    
    Args:
        x: Input tensor [B, L, D] (Batch, Sequence Length, Features)
        threshold: Correlation threshold for edge creation
        
    Returns:
        adj_matrix: Adjacency matrix [B, D, D] - correlations between features
    """
    B, L, D = x.shape
    
    # Compute correlation matrix between features (D x D)
    # x: [B, L, D] -> transpose to [B, D, L] for correlation computation
    x_transposed = x.transpose(1, 2)  # [B, D, L]
    x_centered = x_transposed - x_transposed.mean(dim=2, keepdim=True)  # Center along time dimension
    
    # Compute correlation matrix: [B, D, D]
    corr_matrix = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (L - 1)
    
    # Normalize to get correlation coefficients
    std = torch.sqrt(torch.diagonal(corr_matrix, dim1=1, dim2=2)).unsqueeze(-1)  # [B, D, 1]
    corr_matrix = corr_matrix / (std * std.transpose(1, 2) + 1e-8)
    
    # Create adjacency matrix based on threshold
    adj_matrix = (torch.abs(corr_matrix) > threshold).float()
    
    # Zero out the diagonal to avoid self-loops
    identity = torch.eye(D, device=x.device).unsqueeze(0)
    adj_matrix = adj_matrix * (1 - identity)

    return adj_matrix


def construct_temporal_correlation_graph(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Construct adjacency matrix based on temporal correlations between time steps.
    
    Args:
        x: Input tensor [B, L, D] (Batch, Sequence Length, Features)
        threshold: Correlation threshold for edge creation
        
    Returns:
        adj_matrix: Adjacency matrix [B, L, L] - correlations between time steps
    """
    B, L, D = x.shape
    
    # Compute correlation matrix between time steps (L x L)
    # For each time step, compute correlation with other time steps based on feature vectors
    
    # Reshape for correlation computation: [B, L, D]
    x_centered = x - x.mean(dim=2, keepdim=True)  # Center along feature dimension
    
    # Compute correlation matrix: [B, L, L]
    # bmm([B, L, D], [B, D, L]) -> [B, L, L]
    corr_matrix = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (D - 1)
    
    # Normalize to get correlation coefficients
    # Get standard deviation for each time step
    std = torch.sqrt(torch.diagonal(corr_matrix, dim1=1, dim2=2)).unsqueeze(-1)  # [B, L, 1]
    corr_matrix = corr_matrix / (std * std.transpose(1, 2) + 1e-8)
    
    # Create adjacency matrix based on threshold
    adj_matrix = (torch.abs(corr_matrix) > threshold).float()
    
    # Zero out the diagonal to avoid self-loops
    identity = torch.eye(L, device=x.device).unsqueeze(0)
    adj_matrix = adj_matrix * (1 - identity)

    return adj_matrix


def construct_knn_graph(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Construct k-nearest neighbor graph based on feature similarity.
    This version creates a "mutual" k-NN graph, which is symmetric by definition.
    
    Args:
        x: Input tensor [B, N, D]
        k: Number of nearest neighbors
        
    Returns:
        adj_matrix: Adjacency matrix [B, N, N]
    """
    B, N, D = x.shape
    
    # Compute pairwise distances
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    distances = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    
    # Find k nearest neighbors
    _, indices = torch.topk(distances, k + 1, dim=-1, largest=False)  # +1 to exclude self
    indices = indices[:, :, 1:]  # Remove self-connections
    
    # Create directed adjacency matrix
    adj_matrix_directed = torch.zeros(B, N, N, device=x.device)
    batch_indices = torch.arange(B).view(-1, 1, 1).expand(-1, N, k)
    node_indices = torch.arange(N).view(1, -1, 1).expand(B, -1, k)
    
    adj_matrix_directed[batch_indices, node_indices, indices] = 1.0
    
    # Make symmetric by taking the intersection of edges (mutual k-NN)
    adj_matrix = adj_matrix_directed * adj_matrix_directed.transpose(1, 2)
    
    return adj_matrix