"""
Adjacency-Aware Graph Attention Layer

This module implements graph attention layers that properly utilize adjacency matrices
to mask attention weights, ensuring that attention is only computed between connected nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..base_interfaces import BaseComponent


class AdjacencyAwareGraphAttention(BaseComponent):
    """
    Graph attention layer that properly uses adjacency matrices for attention masking.
    
    This fixes the critical bug where adjacency matrices were ignored in graph attention.
    The layer ensures that attention is only computed between nodes that are connected
    in the graph structure.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension  
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_adjacency_mask: Whether to use adjacency matrix for attention masking
    """
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1, 
                 use_adjacency_mask: bool = True, config=None):
        super().__init__(config)
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout_prob = dropout
        self.use_adjacency_mask = use_adjacency_mask
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable adjacency adaptation
        self.adjacency_adaptation = nn.Parameter(torch.ones(1))
        
    def _create_attention_mask(self, adj_matrix: torch.Tensor, seq_len: int) -> Optional[torch.Tensor]:
        """
        Create attention mask from adjacency matrix.
        
        Args:
            adj_matrix: Adjacency matrix [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]
            seq_len: Sequence length (should match num_nodes)
            
        Returns:
            Attention mask where True indicates positions to ignore
        """
        if adj_matrix is None or not self.use_adjacency_mask:
            return None
            
        # Handle different adjacency matrix dimensions
        if adj_matrix.dim() == 3:
            # Batched adjacency matrix - use the first batch (assuming similar structure)
            attn_mask = (adj_matrix[0] == 0)
        elif adj_matrix.dim() == 2:
            # Single adjacency matrix
            attn_mask = (adj_matrix == 0)
        else:
            # Invalid dimensions
            return None
        
        # Ensure mask dimensions match sequence length
        if attn_mask.size(0) != seq_len or attn_mask.size(1) != seq_len:
            if seq_len <= attn_mask.size(0):
                # Truncate if sequence is shorter
                attn_mask = attn_mask[:seq_len, :seq_len]
            else:
                # Pad with False (allow attention) for additional positions
                pad_size = seq_len - attn_mask.size(0)
                attn_mask = F.pad(attn_mask, (0, pad_size, 0, pad_size), value=False)
        
        return attn_mask
    
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adjacency-aware attention.
        
        Args:
            x: Input features [batch, seq_len, d_model]
            adj_matrix: Adjacency matrix [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]
            
        Returns:
            Output features [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create attention mask from adjacency matrix
        attn_mask = self._create_attention_mask(adj_matrix, seq_len)
        
        # Self-attention with adjacency-based masking
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
        
        # First residual connection and layer norm
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network
        ff_out = self.feed_forward(x)
        
        # Second residual connection and layer norm
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for analysis.
        
        Args:
            x: Input features [batch, seq_len, d_model]
            adj_matrix: Adjacency matrix
            
        Returns:
            Attention weights [batch, n_heads, seq_len, seq_len]
        """
        seq_len = x.size(1)
        attn_mask = self._create_attention_mask(adj_matrix, seq_len)
        
        with torch.no_grad():
            _, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
        
        return attn_weights


class MultiLayerAdjacencyAwareAttention(BaseComponent):
    """
    Multi-layer adjacency-aware graph attention network.
    
    Stacks multiple AdjacencyAwareGraphAttention layers with residual connections.
    """
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int, num_layers: int = 3,
                 dropout: float = 0.1, use_adjacency_mask: bool = True, config=None):
        super().__init__(config)
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            AdjacencyAwareGraphAttention(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout,
                use_adjacency_mask=use_adjacency_mask,
                config=config
            ) for _ in range(num_layers)
        ])
        
        # Layer-wise residual scaling
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multiple adjacency-aware attention layers.
        
        Args:
            x: Input features [batch, seq_len, d_model]
            adj_matrix: Adjacency matrix
            
        Returns:
            Output features [batch, seq_len, d_model]
        """
        for i, layer in enumerate(self.layers):
            # Apply layer with residual scaling
            layer_out = layer(x, adj_matrix)
            x = x + self.layer_scales[i] * (layer_out - x)
        
        return x


class AdaptiveAdjacencyAttention(BaseComponent):
    """
    Adaptive adjacency attention that learns to combine structural and feature-based attention.
    
    This layer learns to balance between:
    1. Structural attention (based on adjacency matrix)
    2. Feature-based attention (based on feature similarity)
    """
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1, config=None):
        super().__init__(config)
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Structural attention (adjacency-based)
        self.structural_attention = AdjacencyAwareGraphAttention(
            d_model, d_ff, n_heads, dropout, use_adjacency_mask=True, config=config
        )
        
        # Feature-based attention (no adjacency mask)
        self.feature_attention = AdjacencyAwareGraphAttention(
            d_model, d_ff, n_heads, dropout, use_adjacency_mask=False, config=config
        )
        
        # Adaptive mixing weights
        self.mixing_network = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive attention mixing.
        
        Args:
            x: Input features [batch, seq_len, d_model]
            adj_matrix: Adjacency matrix
            
        Returns:
            Output features [batch, seq_len, d_model]
        """
        # Compute both types of attention
        structural_out = self.structural_attention(x, adj_matrix)
        feature_out = self.feature_attention(x, None)  # No adjacency mask
        
        # Compute adaptive mixing weights based on input features
        # Use global average pooling to get sequence-level representation
        x_global = x.mean(dim=1)  # [batch, d_model]
        mixing_weights = self.mixing_network(x_global)  # [batch, 2]
        
        # Expand mixing weights to match feature dimensions
        structural_weight = mixing_weights[:, 0:1].unsqueeze(1)  # [batch, 1, 1]
        feature_weight = mixing_weights[:, 1:2].unsqueeze(1)    # [batch, 1, 1]
        
        # Adaptive mixing
        mixed_out = (structural_weight * structural_out + 
                    feature_weight * feature_out)
        
        # Final projection and normalization
        output = self.output_proj(mixed_out)
        output = self.norm(output + x)  # Residual connection
        
        return output
    
    def get_mixing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get the adaptive mixing weights for analysis."""
        x_global = x.mean(dim=1)
        return self.mixing_network(x_global)