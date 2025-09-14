import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class JointSpatioTemporalEncoding(nn.Module):
    """
    Joint Spatial-Temporal Encoding that processes spatial and temporal information simultaneously
    instead of sequentially, enabling better interaction between spatial and temporal patterns.
    """
    
    def __init__(self, d_model: int, seq_len: int, num_nodes: int, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Spatial-temporal interaction matrices
        self.spatial_temporal_interaction = nn.Parameter(
            torch.randn(num_heads, d_model // num_heads, d_model // num_heads)
        )
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]  # Different temporal scales
        ])
        
        # Spatial graph convolutions
        self.spatial_convs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(3)  # Multi-hop spatial convolutions
        ])
        
        # Cross-attention between spatial and temporal features
        self.spatial_temporal_cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion mechanisms
        self.spatial_gate = nn.Linear(d_model * 2, d_model)
        self.temporal_gate = nn.Linear(d_model * 2, d_model)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        
        # Positional encodings
        self.spatial_pos_encoding = nn.Parameter(torch.randn(1, num_nodes, d_model))
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of joint spatial-temporal encoding
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, d_model]
            adjacency_matrix: Spatial adjacency matrix [num_nodes, num_nodes]
            temporal_mask: Optional temporal attention mask
            
        Returns:
            Encoded features with joint spatial-temporal information
        """
        batch_size, seq_len, num_nodes, d_model = x.shape
        
        # Add positional encodings
        x = x + self.spatial_pos_encoding.unsqueeze(1)  # Add spatial positions
        x = x + self.temporal_pos_encoding.unsqueeze(2)  # Add temporal positions
        
        # 1. Multi-scale temporal processing
        temporal_features = self._multi_scale_temporal_processing(x)
        
        # 2. Multi-hop spatial processing
        spatial_features = self._multi_hop_spatial_processing(x, adjacency_matrix)
        
        # 3. Spatial-temporal interaction
        interaction_features = self._spatial_temporal_interaction(temporal_features, spatial_features)
        
        # 4. Cross-attention between spatial and temporal
        cross_attention_features = self._cross_attention_processing(temporal_features, spatial_features)
        
        # 5. Adaptive fusion
        fused_features = self._adaptive_fusion(interaction_features, cross_attention_features, x)
        
        return self.layer_norm3(fused_features)
    
    def _multi_scale_temporal_processing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal information at multiple scales
        """
        batch_size, seq_len, num_nodes, d_model = x.shape
        
        # Reshape for temporal convolution: [batch_size * num_nodes, d_model, seq_len]
        x_temp = x.permute(0, 2, 3, 1).contiguous().view(-1, d_model, seq_len)
        
        # Apply multi-scale temporal convolutions
        temporal_outputs = []
        for conv in self.temporal_convs:
            temp_out = F.relu(conv(x_temp))  # [batch_size * num_nodes, d_model, seq_len]
            temporal_outputs.append(temp_out)
        
        # Combine multi-scale features
        combined_temporal = torch.stack(temporal_outputs, dim=0).mean(dim=0)
        
        # Reshape back: [batch_size, seq_len, num_nodes, d_model]
        combined_temporal = combined_temporal.view(batch_size, num_nodes, d_model, seq_len)
        combined_temporal = combined_temporal.permute(0, 3, 1, 2).contiguous()
        
        return self.layer_norm1(combined_temporal + x)
    
    def _multi_hop_spatial_processing(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Process spatial information with multi-hop graph convolutions
        """
        batch_size, seq_len, num_nodes, d_model = x.shape
        
        # Normalize adjacency matrix
        degree = adjacency_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        normalized_adj = adjacency_matrix / degree
        
        # Multi-hop spatial convolutions
        spatial_features = x
        adj_power = normalized_adj
        
        spatial_outputs = []
        for i, conv in enumerate(self.spatial_convs):
            # Apply spatial convolution at different hops
            # Reshape for matrix multiplication: [batch_size * seq_len, num_nodes, d_model]
            x_spatial = spatial_features.view(-1, num_nodes, d_model)
            
            # Graph convolution: X' = A^k * X * W
            conv_out = torch.bmm(
                adj_power.unsqueeze(0).expand(batch_size * seq_len, -1, -1),
                x_spatial
            )  # [batch_size * seq_len, num_nodes, d_model]
            
            conv_out = conv(conv_out.view(-1, d_model)).view(batch_size * seq_len, num_nodes, d_model)
            conv_out = conv_out.view(batch_size, seq_len, num_nodes, d_model)
            
            spatial_outputs.append(F.relu(conv_out))
            
            # Update adjacency matrix for next hop
            if i < len(self.spatial_convs) - 1:
                adj_power = torch.mm(adj_power, normalized_adj)
        
        # Combine multi-hop features
        combined_spatial = torch.stack(spatial_outputs, dim=0).mean(dim=0)
        
        return self.layer_norm2(combined_spatial + x)
    
    def _spatial_temporal_interaction(self, temporal_features: torch.Tensor, 
                                    spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Compute interaction between spatial and temporal features
        """
        batch_size, seq_len, num_nodes, d_model = temporal_features.shape
        
        # Reshape for multi-head processing
        temp_reshaped = temporal_features.view(batch_size, seq_len, num_nodes, self.num_heads, -1)
        spatial_reshaped = spatial_features.view(batch_size, seq_len, num_nodes, self.num_heads, -1)
        
        # Apply interaction matrices
        interaction_output = torch.zeros_like(temp_reshaped)
        for h in range(self.num_heads):
            # Bilinear interaction: temp^T * W * spatial
            temp_h = temp_reshaped[:, :, :, h, :]  # [batch, seq, nodes, d_k]
            spatial_h = spatial_reshaped[:, :, :, h, :]  # [batch, seq, nodes, d_k]
            
            # Compute bilinear interaction
            interaction = torch.einsum('bsni,ij,bsnj->bsni', temp_h, self.spatial_temporal_interaction[h], spatial_h)
            interaction_output[:, :, :, h, :] = interaction
        
        # Reshape back
        interaction_output = interaction_output.view(batch_size, seq_len, num_nodes, d_model)
        
        return interaction_output
    
    def _cross_attention_processing(self, temporal_features: torch.Tensor, 
                                  spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between spatial and temporal features
        """
        batch_size, seq_len, num_nodes, d_model = temporal_features.shape
        
        # Flatten spatial and temporal dimensions for cross-attention
        # Temporal as query, spatial as key-value
        temp_flat = temporal_features.view(batch_size, seq_len * num_nodes, d_model)
        spatial_flat = spatial_features.view(batch_size, seq_len * num_nodes, d_model)
        
        # Cross-attention: temporal attends to spatial
        cross_attn_output, _ = self.spatial_temporal_cross_attention(
            temp_flat, spatial_flat, spatial_flat
        )
        
        # Reshape back
        cross_attn_output = cross_attn_output.view(batch_size, seq_len, num_nodes, d_model)
        
        return cross_attn_output
    
    def _adaptive_fusion(self, interaction_features: torch.Tensor, 
                        cross_attention_features: torch.Tensor,
                        original_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptively fuse different types of features
        """
        # Compute gating weights
        interaction_gate = torch.sigmoid(self.spatial_gate(
            torch.cat([interaction_features, original_features], dim=-1)
        ))
        
        cross_attn_gate = torch.sigmoid(self.temporal_gate(
            torch.cat([cross_attention_features, original_features], dim=-1)
        ))
        
        # Weighted combination
        combined_features = (
            interaction_gate * interaction_features + 
            cross_attn_gate * cross_attention_features
        )
        
        # Final fusion gate
        fusion_gate = torch.sigmoid(self.fusion_gate(
            torch.cat([combined_features, original_features], dim=-1)
        ))
        
        # Final output with residual connection
        output = fusion_gate * combined_features + (1 - fusion_gate) * original_features
        
        return self.dropout_layer(output)


class AdaptiveSpatioTemporalEncoder(nn.Module):
    """
    Adaptive encoder that can handle varying spatial and temporal scales
    """
    
    def __init__(self, d_model: int, max_seq_len: int, max_nodes: int,
                 num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack of joint spatial-temporal encoding layers
        self.encoding_layers = nn.ModuleList([
            JointSpatioTemporalEncoding(
                d_model, max_seq_len, max_nodes, num_heads, dropout
            ) for _ in range(num_layers)
        ])
        
        # Adaptive pooling for variable sequence lengths
        self.adaptive_temporal_pool = nn.AdaptiveAvgPool1d(max_seq_len)
        
        # Final projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multiple joint spatial-temporal encoding layers
        """
        current_features = x
        
        # Apply each encoding layer
        for layer in self.encoding_layers:
            current_features = layer(current_features, adjacency_matrix, temporal_mask)
        
        # Final projection
        output = self.output_projection(current_features)
        
        return output