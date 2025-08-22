"""
Graph-based encoder implementations for time series analysis.
Integrates graph attention mechanisms with temporal sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from .base_encoder import BaseEncoder
from ..attention.graph_attention import MultiGraphAttention, construct_correlation_graph, construct_knn_graph, construct_temporal_correlation_graph
from ..layers.common import FeedForward
import torch.nn.functional as F


class GraphEncoderLayer(nn.Module):
    """
    Single layer of graph-based encoder with attention and feed-forward components.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        graph_type: Type of graph construction ('correlation', 'knn', 'learned', 'provided')
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: Optional[int] = None,
                 dropout: float = 0.1, activation: str = "relu", 
                 graph_type: str = "correlation", **graph_kwargs):
        super().__init__()
        
        self.d_model = d_model
        self.graph_type = graph_type
        self.graph_kwargs = graph_kwargs
        
        # Graph attention mechanism
        self.graph_attention = MultiGraphAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        d_ff = d_ff or 4 * d_model
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable adjacency matrix (if graph_type == 'learned')
        if graph_type == 'learned':
            num_nodes = graph_kwargs.get('num_nodes', 100)  # Default fallback
            self.adj_matrix = nn.Parameter(torch.randn(num_nodes, num_nodes))
            nn.init.xavier_uniform_(self.adj_matrix)
            
    def construct_graph(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Construct adjacency matrix based on graph_type.
        
        Args:
            x: Input tensor [B, L, N] or [B, N, D]
            
        Returns:
            adj_matrix: Adjacency matrix or None
        """
        if self.graph_type == "correlation":
            threshold = self.graph_kwargs.get('threshold', 0.5)
            # For time series data, use temporal correlation (between time steps)
            return construct_temporal_correlation_graph(x, threshold)
        elif self.graph_type == "knn":
            k = self.graph_kwargs.get('k', 5)
            return construct_knn_graph(x, k)
        elif self.graph_type == "learned":
            B = x.size(0)
            adj = torch.sigmoid(self.adj_matrix)
            return adj.unsqueeze(0).expand(B, -1, -1)
        elif self.graph_type == "provided":
            return None  # Will be provided externally
        else:
            return None
            
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of graph encoder layer.
        
        Args:
            x: Input tensor [B, L, D] or [B, N, D]
            adj_matrix: Optional adjacency matrix
            attn_mask: Attention mask
            
        Returns:
            output: Transformed features
            attention: Attention weights
        """
        # Construct graph if not provided
        if adj_matrix is None and self.graph_type != "provided":
            adj_matrix = self.construct_graph(x)
            
        # Graph attention with residual connection
        residual = x
        x, attention = self.graph_attention(x, adj_matrix, attn_mask)
        x = self.norm1(x + residual)
        
        # Feed-forward with residual connection
        residual = x
        x = self.feed_forward(x)
        x = self.norm2(x + residual)
        
        return x, attention


class GraphTimeSeriesEncoder(BaseEncoder):
    """
    Multi-layer graph encoder for time series data.
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        graph_type: Type of graph construction
        norm_layer: Optional normalization layer
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int = 8,
                 d_ff: Optional[int] = None, dropout: float = 0.1,
                 activation: str = "relu", graph_type: str = "correlation",
                 norm_layer: Optional[nn.Module] = None, **graph_kwargs):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.graph_type = graph_type
        
        # Create encoder layers
        self.layers = nn.ModuleList([
            GraphEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                graph_type=graph_type,
                **graph_kwargs
            ) for _ in range(num_layers)
        ])
        
        # Optional normalization
        self.norm = norm_layer
        
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input tensor [B, L, D] or [B, N, D]
            adj_matrix: Optional adjacency matrix
            attn_mask: Attention mask
            
        Returns:
            output: Final encoded features
            attentions: List of attention weights from each layer
        """
        attentions = []
        
        for layer in self.layers:
            x, attention = layer(x, adj_matrix, attn_mask, **kwargs)
            attentions.append(attention)
            
        if self.norm is not None:
            x = self.norm(x)
            
        return x, attentions


class HybridGraphEncoder(BaseEncoder):
    """
    Hybrid encoder that combines standard temporal attention with graph attention.
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        graph_layers: Indices of layers to use graph attention (e.g., [0, 2, 4])
        graph_type: Type of graph construction
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int = 8,
                 d_ff: Optional[int] = None, dropout: float = 0.1,
                 activation: str = "relu", graph_layers: Optional[list] = None,
                 graph_type: str = "correlation", **graph_kwargs):
        super().__init__()
        
    # Use existing attention components from registry (not required here; using built-ins)
        
        self.num_layers = num_layers
        self.graph_layers = graph_layers or [i for i in range(0, num_layers, 2)]  # Every other layer
        
        # Create mixed layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i in self.graph_layers:
                # Graph attention layer
                layer = GraphEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    graph_type=graph_type,
                    **graph_kwargs
                )
            else:
                # Standard temporal attention layer - use a simple attention mechanism
                # For now, create a basic multi-head attention
                attention_comp = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
                
                # Create a wrapper layer that matches the expected interface
                class StandardLayer(nn.Module):
                    def __init__(self, attention, d_model, d_ff, dropout, activation):
                        super().__init__()
                        self.attention = attention
                        self.feed_forward = FeedForward(d_model, d_ff or 4*d_model, dropout, activation)
                        self.norm1 = nn.LayerNorm(d_model)
                        self.norm2 = nn.LayerNorm(d_model)
                        self.dropout = nn.Dropout(dropout)
                        
                    def forward(self, x, attn_mask=None, **kwargs):
                        # Handle 4D input by reshaping
                        original_shape = x.shape
                        if x.dim() == 4:
                            B, L, N, D = x.shape
                            x = x.view(B * L, N, D)
                            
                        # Self-attention
                        residual = x
                        attn_out, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
                        x = self.norm1(residual + self.dropout(attn_out))
                        
                        # Feed-forward
                        residual = x
                        x = self.feed_forward(x)
                        x = self.norm2(residual + x)
                        
                        # Reshape back if needed
                        if len(original_shape) == 4:
                            x = x.view(original_shape)
                            # Reshape attention weights too
                            attn_weights = attn_weights.view(B, L, N, N)
                        
                        return x, attn_weights
                        
                layer = StandardLayer(attention_comp, d_model, d_ff, dropout, activation)
            self.layers.append(layer)
            
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, list]:
        """Forward pass through hybrid layers."""
        attentions = []
        
        for i, layer in enumerate(self.layers):
            if i in self.graph_layers:
                # Graph layer
                x, attention = layer(x, adj_matrix, attn_mask, **kwargs)
            else:
                # Standard layer
                x, attention = layer(x, attn_mask=attn_mask, **kwargs)
            attentions.append(attention)
            
        return x, attentions


class AdaptiveGraphEncoder(BaseEncoder):
    """
    Adaptive encoder that learns to weight graph vs temporal attention.
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int = 8,
                 d_ff: Optional[int] = None, dropout: float = 0.1,
                 activation: str = "relu", graph_type: str = "correlation", **graph_kwargs):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Dual attention mechanisms
        self.graph_layers = nn.ModuleList([
            GraphEncoderLayer(d_model, n_heads, d_ff, dropout, activation, graph_type, **graph_kwargs)
            for _ in range(num_layers)
        ])
        
        from ..layers.standard_layers import StandardEncoderLayer
        # Use PyTorch's built-in attention for temporal layers
        
        # Create temporal layers using PyTorch's MultiheadAttention
        class TemporalLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout, activation):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
                self.feed_forward = FeedForward(d_model, d_ff or 4*d_model, dropout, activation)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x, attn_mask=None, **kwargs):
                # Handle 4D input by reshaping
                original_shape = x.shape
                if x.dim() == 4:
                    B, L, N, D = x.shape
                    x = x.view(B * L, N, D)
                    
                # Self-attention
                residual = x
                attn_out, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
                x = self.norm1(residual + self.dropout(attn_out))
                
                # Feed-forward
                residual = x
                x = self.feed_forward(x)
                x = self.norm2(residual + x)
                
                # Reshape back if needed
                if len(original_shape) == 4:
                    x = x.view(original_shape)
                    # Reshape attention weights too
                    attn_weights = attn_weights.view(B, L, N, N)
                
                return x, attn_weights
        
        self.temporal_layers = nn.ModuleList([
            TemporalLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Adaptive weighting
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, list]:
        """Forward pass with adaptive weighting."""
        attentions = []
        
        for graph_layer, temporal_layer in zip(self.graph_layers, self.temporal_layers):
            # Compute both attention types
            x_graph, attn_graph = graph_layer(x, adj_matrix, attn_mask, **kwargs)
            x_temporal, attn_temporal = temporal_layer(x, attn_mask=attn_mask, **kwargs)
            
            # Compute adaptive weight
            # x: [B, L, N, D] -> mean over L and N: [B, D]
            if x.dim() == 4:
                weight_input = x.mean(dim=(1, 2))  # [B, D]
            else:
                weight_input = x.mean(dim=1)  # [B, D]
            weight = self.weight_predictor(weight_input)  # [B, 1]
            
            # Reshape weight for broadcasting
            if x.dim() == 4:
                weight = weight.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            else:
                weight = weight.view(-1, 1, 1)  # [B, 1, 1]
            
            # Weighted combination
            x = weight * x_graph + (1 - weight) * x_temporal
            
            # Handle attention combination - need to match dimensions
            weight_for_attn = weight.squeeze(-1) if x.dim() == 4 else weight.squeeze(-1)
            if attn_graph.dim() != attn_temporal.dim():
                # If dimensions don't match, just use graph attention
                attention = attn_graph
            else:
                # Ensure weight broadcasting works for attention
                while weight_for_attn.dim() < attn_graph.dim():
                    weight_for_attn = weight_for_attn.unsqueeze(-1)
                attention = weight_for_attn * attn_graph + (1 - weight_for_attn) * attn_temporal
            
            attentions.append(attention)
            
        return x, attentions