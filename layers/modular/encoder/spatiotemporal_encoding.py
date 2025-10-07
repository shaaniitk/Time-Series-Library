import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.modular.graph.registry import GraphComponentRegistry

class JointSpatioTemporalEncoding(nn.Module):
    """Jointly encode spatial and temporal patterns with optional token budgeting.

    The encoder processes spatial and temporal information simultaneously instead of
    sequentially, enabling richer cross-domain interactions. When ``max_tokens`` is
    provided, the cross-attention sub-module will subsample tokens adaptively to
    avoid quadratic memory spikes on large spatio-temporal grids.
    """
    
    def __init__(self, d_model: int, seq_len: int, num_nodes: int, 
                 num_heads: int = 8, dropout: float = 0.1, max_tokens: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.dropout = dropout
        default_cap = int(os.environ.get("PGAT_MAX_ATTENTION_TOKENS", "4096"))
        if max_tokens is None or max_tokens <= 0:
            self.max_tokens = default_cap
        else:
            self.max_tokens = min(max_tokens, default_cap)
        self._token_budget_warned = False
        
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
            x: Input tensor [batch_size, seq_len, num_nodes, d_model] or [batch_size, seq_len, d_model]
            adjacency_matrix: Spatial adjacency matrix [num_nodes, num_nodes]
            temporal_mask: Optional temporal attention mask
            
        Returns:
            Encoded features with joint spatial-temporal information
        """
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            # 3D input: [batch_size, seq_len, d_model] - aggregated features case
            return self._forward_aggregated(x, adjacency_matrix, temporal_mask)
        elif x.dim() == 4:
            # 4D input: [batch_size, seq_len, num_nodes, d_model] - original case
            return self._forward_original(x, adjacency_matrix, temporal_mask)
        else:
            raise ValueError(f"Expected 3D or 4D input tensor, got {x.dim()}D")
    
    def _forward_original(self, x: torch.Tensor, adjacency_matrix: torch.Tensor,
                         temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Original forward pass for 4D input [batch_size, seq_len, num_nodes, d_model]"""
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
    
    def _forward_aggregated(self, x: torch.Tensor, adjacency_matrix: torch.Tensor,
                           temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for 3D aggregated input [batch_size, seq_len, d_model]"""
        batch_size, seq_len, d_model = x.shape
        num_nodes = adjacency_matrix.size(0)
        
        # Method 1: Treat the aggregated features as a single "super-node"
        # and apply temporal processing only
        
        # Apply temporal convolutions to capture temporal patterns
        x_temp = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        
        # Multi-scale temporal processing
        temporal_outputs = []
        for conv in self.temporal_convs:
            temp_out = F.relu(conv(x_temp))  # [batch, d_model, seq_len]
            temporal_outputs.append(temp_out)
        
        # Combine multi-scale features
        combined_temporal = torch.stack(temporal_outputs, dim=0).mean(dim=0)
        combined_temporal = combined_temporal.permute(0, 2, 1)  # [batch, seq_len, d_model]
        
        # Add residual connection and layer norm
        temporal_features = self.layer_norm1(combined_temporal + x)
        
        # Apply self-attention for global temporal dependencies
        attn_output, _ = self.spatial_temporal_cross_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # Final layer norm and residual
        output = self.layer_norm3(attn_output + temporal_features)
        
        return output
    
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
    
    def _cross_attention_processing(
        self,
        temporal_features: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention between spatial and temporal features.

        The method can optionally subsample tokens before attention to
        respect ``max_tokens`` limits, then upsamples the attended output to
        match the original spatial-temporal resolution for downstream fusion.
        """
        batch_size, original_seq_len, original_num_nodes, d_model = temporal_features.shape

        # Prepare potentially downsampled representations for attention.
        temp_tokens = temporal_features
        spatial_tokens = spatial_features
        seq_len = original_seq_len
        num_nodes = original_num_nodes

        if self.max_tokens is not None:
            token_cap = max(1, self.max_tokens)
            total_tokens = seq_len * num_nodes

            if total_tokens > token_cap:
                if not self._token_budget_warned:
                    logging.warning(
                        "JointSpatioTemporalEncoding downsampling tokens from %s to %s",
                        total_tokens,
                        token_cap,
                    )
                    self._token_budget_warned = True
                # Preferentially reduce the node dimension.
                max_nodes = max(1, token_cap // seq_len) if seq_len > 0 else num_nodes
                if max_nodes < num_nodes:
                    node_stride = max(1, math.ceil(num_nodes / max_nodes))
                    node_indices = torch.arange(
                        0,
                        num_nodes,
                        node_stride,
                        device=temp_tokens.device,
                        dtype=torch.long,
                    )
                    if node_indices[-1] != num_nodes - 1:
                        node_indices = torch.cat(
                            [node_indices, node_indices.new_tensor([num_nodes - 1])]
                        )
                    node_indices = node_indices[:max_nodes]
                    temp_tokens = temp_tokens.index_select(2, node_indices)
                    spatial_tokens = spatial_tokens.index_select(2, node_indices)
                    num_nodes = temp_tokens.size(2)
                    total_tokens = seq_len * num_nodes

                # Reduce the temporal dimension if tokens remain above the cap.
                if total_tokens > token_cap and seq_len > 0:
                    max_seq = max(1, token_cap // num_nodes) if num_nodes > 0 else seq_len
                    if max_seq < seq_len:
                        time_stride = max(1, math.ceil(seq_len / max_seq))
                        time_indices = torch.arange(
                            0,
                            seq_len,
                            time_stride,
                            device=temp_tokens.device,
                            dtype=torch.long,
                        )
                        if time_indices[-1] != seq_len - 1:
                            time_indices = torch.cat(
                                [time_indices, time_indices.new_tensor([seq_len - 1])]
                            )
                        time_indices = time_indices[:max_seq]
                        temp_tokens = temp_tokens.index_select(1, time_indices)
                        spatial_tokens = spatial_tokens.index_select(1, time_indices)
                        seq_len = temp_tokens.size(1)

        seq_len = max(1, seq_len)
        num_nodes = max(1, num_nodes)

        temp_flat = temp_tokens.contiguous().view(batch_size, seq_len * num_nodes, d_model)
        spatial_flat = spatial_tokens.contiguous().view(batch_size, seq_len * num_nodes, d_model)

        cross_attn_output, _ = self.spatial_temporal_cross_attention(
            temp_flat,
            spatial_flat,
            spatial_flat,
        )

        cross_attn_output = cross_attn_output.view(batch_size, seq_len, num_nodes, d_model)

        if seq_len != original_seq_len or num_nodes != original_num_nodes:
            cross_attn_output = cross_attn_output.permute(0, 3, 1, 2).contiguous()
            target_size = (original_seq_len, original_num_nodes)
            if seq_len > 1 and num_nodes > 1:
                cross_attn_output = F.interpolate(
                    cross_attn_output,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                cross_attn_output = F.interpolate(
                    cross_attn_output,
                    size=target_size,
                    mode="nearest",
                )
            cross_attn_output = cross_attn_output.permute(0, 2, 3, 1).contiguous()

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


@GraphComponentRegistry.register("adaptive_spatiotemporal_encoder")
class AdaptiveSpatioTemporalEncoder(nn.Module):
    """Stacked joint spatial-temporal encoder with optional token budgeting.

    Args:
        d_model: Embedding dimension for model representations.
        max_seq_len: Maximum sequence length expected by the encoder.
        max_nodes: Maximum number of spatial nodes handled by the encoder.
        num_layers: Number of joint spatial-temporal encoding layers to stack.
        num_heads: Number of attention heads in each joint layer.
        dropout: Dropout probability applied within each encoding layer.
        max_tokens: Optional cap on the total tokens processed during cross-attention.
            When provided, each encoding layer enforces the limit to avoid quadratic
            memory blow-ups.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        max_nodes: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_tokens = max_tokens

        self.encoding_layers = nn.ModuleList(
            [
                JointSpatioTemporalEncoding(
                    d_model,
                    max_seq_len,
                    max_nodes,
                    num_heads,
                    dropout,
                    max_tokens,
                )
                for _ in range(num_layers)
            ]
        )

        self.adaptive_temporal_pool = nn.AdaptiveAvgPool1d(max_seq_len)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run stacked encoding layers with optional temporal masking."""
        current_features = x

        for layer in self.encoding_layers:
            current_features = layer(current_features, adjacency_matrix, temporal_mask)

        output = self.output_projection(current_features)
        return output