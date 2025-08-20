"""
Graph-aware embedding components for time series data.
Combines temporal and spatial (graph) embeddings for enhanced representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GraphPositionalEncoding(nn.Module):
    """
    Positional encoding that incorporates graph structure information.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        seq_len = x.size(0)
        # pe is [max_len, 1, d_model], x is [seq_len, batch_size, d_model]
        pe_slice = self.pe[:seq_len, :, :x.size(-1)]  # Match the feature dimension
        x = x + pe_slice
        return self.dropout(x)


class SpatialEmbedding(nn.Module):
    """
    Spatial embedding that learns node-specific representations.
    """
    
    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        self.spatial_emb = nn.Parameter(torch.randn(num_nodes, d_model))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.spatial_emb)
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate spatial embeddings for batch."""
        return self.dropout(self.spatial_emb.unsqueeze(0).expand(batch_size, -1, -1))


class GraphTimeSeriesEmbedding(nn.Module):
    """
    Combined temporal and spatial embedding for graph-based time series.
    
    Args:
        c_in: Input feature dimension
        d_model: Model dimension
        num_nodes: Number of nodes in graph (optional)
        max_len: Maximum sequence length
        dropout: Dropout rate
        embed_type: Type of temporal embedding ('fixed', 'learned')
    """
    
    def __init__(self, c_in: int, d_model: int, num_nodes: Optional[int] = None,
                 max_len: int = 5000, dropout: float = 0.1, embed_type: str = 'fixed'):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.embed_type = embed_type
        
        # Value embedding
        self.value_embedding = nn.Linear(c_in, d_model)
        
        # Temporal embedding
        if embed_type == 'fixed':
            self.temporal_embed = GraphPositionalEncoding(d_model, max_len, dropout)
        elif embed_type == 'learned':
            self.temporal_embed = nn.Embedding(max_len, d_model)
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}")
            
        # Spatial embedding (if num_nodes provided)
        if num_nodes is not None:
            self.spatial_embed = SpatialEmbedding(num_nodes, d_model, dropout)
        else:
            self.spatial_embed = None
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, temporal_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of graph time series embedding.
        
        Args:
            x: Input tensor [B, L, N, C] or [B, L, C]
            temporal_idx: Temporal indices for learned embedding
            
        Returns:
            embedded: Embedded tensor [B, L, N, D] or [B, L, D]
        """
        if x.dim() == 3:  # [B, L, C]
            B, L, C = x.shape
            N = 1
            x = x.unsqueeze(2)  # [B, L, 1, C]
        else:  # [B, L, N, C]
            B, L, N, C = x.shape
            
        # Value embedding
        x = self.value_embedding(x)  # [B, L, N, D]
        
        # Temporal embedding
        if self.embed_type == 'fixed':
            # Apply positional encoding along the sequence dimension
            # x: [B, L, N, D] -> apply PE to each (B, N) independently
            temp_emb = torch.zeros_like(x)
            for i in range(B):
                for j in range(N):
                    # x[i, :, j, :] has shape [L, D]
                    seq_data = x[i, :, j, :].unsqueeze(1)  # [L, 1, D]
                    temp_emb[i, :, j, :] = self.temporal_embed(seq_data).squeeze(1)  # [L, D]
        else:  # learned
            if temporal_idx is None:
                temporal_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            temp_emb = self.temporal_embed(temporal_idx).unsqueeze(2).expand(-1, -1, N, -1)
            
        x = x + temp_emb
        
        # Spatial embedding
        if self.spatial_embed is not None:
            spatial_emb = self.spatial_embed(B).unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, N, D]
            x = x + spatial_emb
            
        x = self.dropout(x)
        
        # Return original shape
        if N == 1:
            x = x.squeeze(2)  # [B, L, D]
            
        return x


class AdaptiveGraphEmbedding(nn.Module):
    """
    Adaptive embedding that learns to weight temporal vs spatial information.
    """
    
    def __init__(self, c_in: int, d_model: int, num_nodes: Optional[int] = None,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.base_embed = GraphTimeSeriesEmbedding(
            c_in, d_model, num_nodes, max_len, dropout, 'fixed'
        )
        
        # Adaptive weighting
        self.temporal_weight = nn.Parameter(torch.ones(1))
        self.spatial_weight = nn.Parameter(torch.ones(1)) if num_nodes else None
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2 if num_nodes else 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, temporal_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with adaptive weighting."""
        embedded = self.base_embed(x, temporal_idx)
        
        # Compute adaptive weights
        # embedded: [B, L, N, D] -> mean over L: [B, N, D] -> mean over N: [B, D]
        gate_input = embedded.mean(dim=(1, 2))  # [B, D]
        gate_weights = self.gate(gate_input)  # [B, 2] or [B, 1]
        
        if self.spatial_weight is not None:
            temp_weight = gate_weights[:, 0:1] * self.temporal_weight  # [B, 1]
            spat_weight = gate_weights[:, 1:2] * self.spatial_weight   # [B, 1]
            
            # Apply weights (simplified - in practice would separate temporal/spatial components)
            total_weight = temp_weight + spat_weight  # [B, 1]
            # Broadcast to match embedded shape: [B, 1] -> [B, 1, 1, 1]
            embedded = embedded * total_weight.view(-1, 1, 1, 1)
        else:
            temp_weight = gate_weights * self.temporal_weight
            embedded = embedded * temp_weight.unsqueeze(1).unsqueeze(-1)
            
        return embedded


class GraphFeatureEmbedding(nn.Module):
    """
    Feature embedding that incorporates graph structure into feature representation.
    """
    
    def __init__(self, c_in: int, d_model: int, use_graph_conv: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_graph_conv = use_graph_conv
        
        if use_graph_conv:
            self.graph_conv = nn.Conv1d(c_in, d_model, kernel_size=1)
            self.norm = nn.BatchNorm1d(d_model)
        else:
            self.linear = nn.Linear(c_in, d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional graph convolution.
        
        Args:
            x: Input features [B, L, N, C] or [B, L, C]
            adj_matrix: Adjacency matrix [B, N, N] or [N, N]
            
        Returns:
            embedded: Feature embeddings
        """
        if self.use_graph_conv and adj_matrix is not None and x.dim() == 4:
            B, L, N, C = x.shape
            
            # Apply graph convolution
            x_reshaped = x.view(B * L, N, C)
            
            if adj_matrix.dim() == 2:
                adj_matrix = adj_matrix.unsqueeze(0).expand(B * L, -1, -1)
            elif adj_matrix.dim() == 3 and adj_matrix.size(0) == B:
                adj_matrix = adj_matrix.unsqueeze(1).expand(-1, L, -1, -1).contiguous().view(B * L, N, N)
                
            # Graph convolution: X' = AXW
            x_conv = torch.bmm(adj_matrix, x_reshaped)  # [B*L, N, C]
            x_conv = self.graph_conv(x_conv.transpose(1, 2)).transpose(1, 2)  # [B*L, N, D]
            x_conv = self.norm(x_conv.transpose(1, 2)).transpose(1, 2)
            
            x = x_conv.view(B, L, N, -1)
        else:
            x = self.linear(x)
            
        return self.dropout(x)