
import torch
import torch.nn as nn

class HierarchicalTemporalSpatialMapper(nn.Module):
    """
    Maps a sequence of temporal patch embeddings to a spatial representation for a set of nodes.
    This is achieved through a hierarchical process involving temporal convolution and self-attention.
    """
    def __init__(self, d_model: int, num_nodes: int, n_heads: int, num_attention_layers: int = 2, conv_kernel_size: int = 3):
        """
        Args:
            d_model (int): The dimensionality of the model.
            num_nodes (int): The number of spatial nodes to map to.
            n_heads (int): The number of attention heads.
            num_attention_layers (int): The number of self-attention layers to stack.
            conv_kernel_size (int): The kernel size for the initial temporal convolution.
        """
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.n_heads = n_heads

        # 1. Temporal Convolution to capture local patterns
        self.temporal_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_activation = nn.ReLU()

        # 2. Stacked Self-Attention layers to capture global dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            activation='relu',
            batch_first=True
        )
        self.attention_stack = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)

        # 3. Final mapping to spatial nodes
        # This layer maintains d_model dimension while ensuring proper spatial structure
        self.spatial_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_nodes, d_model].
        """
        batch_size, num_patches, _ = x.shape

        # 1. Apply temporal convolution
        # Conv1d expects [B, C, L], so we permute
        x_conv = x.permute(0, 2, 1)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1) # Permute back to [B, L, C]
        x = self.conv_norm(x + x_conv) # Residual connection
        x = self.conv_activation(x)

        # 2. Apply self-attention stack
        x_attended = self.attention_stack(x)

        # 3. Project to spatial nodes
        # Fixed: Use adaptive pooling instead of dynamic linear layers
        # This avoids recreating weights and handles variable patch counts
        
        # Method 1: Adaptive average pooling (most stable)
        if num_patches >= self.num_nodes:
            # Downsample patches to nodes using adaptive pooling
            x_pooled = nn.functional.adaptive_avg_pool1d(
                x_attended.permute(0, 2, 1),  # [B, d_model, num_patches]
                self.num_nodes
            ).permute(0, 2, 1)  # [B, num_nodes, d_model]
        else:
            # Upsample patches to nodes using interpolation
            x_interpolated = nn.functional.interpolate(
                x_attended.permute(0, 2, 1),  # [B, d_model, num_patches]
                size=self.num_nodes,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [B, num_nodes, d_model]
            x_pooled = x_interpolated
        
        # Apply final spatial projection to ensure proper dimensionality
        spatial_output = self.spatial_projection(x_pooled)
        
        return spatial_output
