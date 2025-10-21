"""
FIXED Celestial Graph Combiner - Batch Processing for Memory Efficiency

This version fixes the critical memory issues by:
1. Processing all timesteps in parallel instead of sequentially
2. Using efficient batch operations
3. Implementing gradient checkpointing
4. Reducing intermediate tensor accumulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math
from torch.utils.checkpoint import checkpoint
from layers.modular.graph.registry import GraphComponentRegistry

@GraphComponentRegistry.register("celestial_graph_combiner_fixed")
class CelestialGraphCombinerFixed(nn.Module):
    """
    FIXED: Memory-efficient graph combiner using batch processing
    
    Key fixes:
    1. Parallel processing of all timesteps
    2. Reduced intermediate tensor creation
    3. Gradient checkpointing support
    4. Memory-efficient attention patterns
    """
    
    def __init__(
        self, 
        num_nodes: int,
        d_model: int, 
        num_attention_heads: int = 8,
        fusion_layers: int = 2,  # Reduced from 3
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.fusion_layers = fusion_layers
        self.dropout = dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Simplified edge transformation (reduced complexity)
        self.edge_projection = nn.Linear(3, d_model)  # Direct projection from 3 edge types
        
        # Memory-efficient fusion layers
        self.fusion_layers_net = nn.ModuleList([
            MemoryEfficientFusionLayer(d_model, num_attention_heads, dropout)
            for _ in range(fusion_layers)
        ])
        
        # Simplified output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor, 
        attention_edges: torch.Tensor,
        enc_out: torch.Tensor,
        market_regime: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        FIXED: Batch processing for all timesteps simultaneously
        
        Args:
            astronomical_edges: [batch, seq_len, num_nodes, num_nodes]
            learned_edges: [batch, seq_len, num_nodes, num_nodes]
            attention_edges: [batch, seq_len, num_nodes, num_nodes]
            enc_out: [batch, seq_len, d_model]
            
        Returns:
            Tuple of (combined_edges, metadata)
        """
        batch_size, seq_len, num_nodes, _ = astronomical_edges.shape
        
        # FIXED: Process all timesteps in parallel
        if self.use_gradient_checkpointing and self.training:
            combined_edges = checkpoint(
                self._batch_process_edges,
                astronomical_edges, learned_edges, attention_edges, enc_out
            )
        else:
            combined_edges = self._batch_process_edges(
                astronomical_edges, learned_edges, attention_edges, enc_out
            )
        
        # Generate lightweight metadata
        metadata = self._generate_lightweight_metadata(
            astronomical_edges, learned_edges, attention_edges, combined_edges
        )
        
        return combined_edges, metadata
    
    def _batch_process_edges(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor,
        attention_edges: torch.Tensor,
        enc_out: torch.Tensor
    ) -> torch.Tensor:
        """Process all edges in parallel for memory efficiency."""
        batch_size, seq_len, num_nodes, _ = astronomical_edges.shape
        
        # Stack edge types: [batch, seq_len, num_nodes, num_nodes, 3]
        stacked_edges = torch.stack([
            astronomical_edges, learned_edges, attention_edges
        ], dim=-1)
        
        # Flatten for batch processing: [batch * seq_len * num_nodes * num_nodes, 3]
        edges_flat = stacked_edges.view(-1, 3)
        
        # Project to feature space: [batch * seq_len * num_nodes * num_nodes, d_model]
        edge_features = self.edge_projection(edges_flat)
        
        # Reshape back: [batch, seq_len, num_nodes, num_nodes, d_model]
        edge_features = edge_features.view(batch_size, seq_len, num_nodes, num_nodes, self.d_model)
        
        # Apply fusion layers with batch processing
        current_features = edge_features
        for fusion_layer in self.fusion_layers_net:
            current_features = fusion_layer(current_features, enc_out)
        
        # Final projection: [batch, seq_len, num_nodes, num_nodes]
        combined_edges = self.output_projection(current_features).squeeze(-1)
        
        return combined_edges
    
    def _generate_lightweight_metadata(
        self,
        astronomical_edges: torch.Tensor,
        learned_edges: torch.Tensor,
        attention_edges: torch.Tensor,
        combined_edges: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate minimal metadata to reduce memory overhead."""
        
        # Compute statistics without creating large intermediate tensors
        with torch.no_grad():
            astro_strength = astronomical_edges.abs().mean()
            learned_strength = learned_edges.abs().mean()
            attention_strength = attention_edges.abs().mean()
            
            total_strength = astro_strength + learned_strength + attention_strength + 1e-8
            
            metadata = {
                'astronomical_contribution': astro_strength / total_strength,
                'learned_contribution': learned_strength / total_strength,
                'attention_contribution': attention_strength / total_strength,
                'final_edge_density': (combined_edges.abs() > 0.1).float().mean(),
                'combination_weights': F.softmax(self.combination_weights, dim=0)
            }
        
        return metadata


class MemoryEfficientFusionLayer(nn.Module):
    """Memory-efficient fusion layer with reduced complexity."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Simplified attention (no cross-attention to reduce memory)
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Reduced feed-forward dimension
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # Reduced from 4x to 2x
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        edge_features: torch.Tensor, 
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Memory-efficient fusion processing.
        
        Args:
            edge_features: [batch, seq_len, num_nodes, num_nodes, d_model]
            context: [batch, seq_len, d_model]
        """
        batch_size, seq_len, num_nodes, _, d_model = edge_features.shape
        
        # Flatten spatial dimensions for attention
        features_flat = edge_features.view(batch_size, seq_len, -1, d_model)
        # [batch, seq_len, num_nodes*num_nodes, d_model]
        
        # Reshape for batch attention processing
        features_for_attention = features_flat.view(batch_size * seq_len, -1, d_model)
        
        # Self-attention
        attended, _ = self.self_attention(
            features_for_attention, features_for_attention, features_for_attention
        )
        attended = attended.view(batch_size, seq_len, -1, d_model)
        
        # Residual connection and norm
        features_flat = self.norm1(features_flat + self.dropout(attended))
        
        # Feed-forward
        ff_output = self.feed_forward(features_flat)
        features_flat = self.norm2(features_flat + self.dropout(ff_output))
        
        # Reshape back to original format
        output = features_flat.view(batch_size, seq_len, num_nodes, num_nodes, d_model)
        
        return output