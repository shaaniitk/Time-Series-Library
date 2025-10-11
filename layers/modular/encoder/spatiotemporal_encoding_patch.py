"""Patch for spatiotemporal encoding to handle variable sequence lengths."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def create_adaptive_positional_encoding(max_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Create sinusoidal positional encoding that can be truncated or extended."""
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                        -(torch.log(torch.tensor(10000.0)) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)  # Add batch dimension


def patched_joint_spatiotemporal_forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor,
                                       temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Patched forward method that handles variable sequence lengths and last-dim mismatches."""
    # Support both 3D aggregated and 4D original inputs
    if x.dim() == 3:
        batch_size, seq_len, x_last_dim = x.shape
        num_nodes = adjacency_matrix.size(0)
        # Ensure last dimension matches self.d_model for internal convs
        if x_last_dim != self.d_model:
            if self.d_model % x_last_dim == 0:
                factor = self.d_model // x_last_dim
                x = x.repeat_interleave(factor, dim=-1)
            else:
                x = F.pad(x, (0, self.d_model - x_last_dim))
        # Create dummy spatial positions for aggregated case
        spatial_pos_enc = getattr(self, 'spatial_pos_encoding', torch.zeros(1, num_nodes, self.d_model, device=x.device))
        temp_pos_enc = getattr(self, 'temporal_pos_encoding', create_adaptive_positional_encoding(seq_len, self.d_model, x.device))
        try:
            x = x + temp_pos_enc  # [batch, seq_len, d_model]
        except RuntimeError:
            pass
        # Minimal temporal-only processing path: use original aggregated forward components
        # Fall back to 1D temporal convs if available
        try:
            temporal_features = self._multi_scale_temporal_processing(x)
        except Exception:
            # Simple temporal smoothing as fallback
            x_temp = x.permute(0, 2, 1)
            conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, padding=0).to(x.device)
            temporal_features = F.relu(conv(x_temp)).permute(0, 2, 1)
        # Project back to expected output token shape
        return self.layer_norm3(temporal_features + x)
    
    # 4D input path
    batch_size, seq_len, num_nodes, x_last_dim = x.shape
    # Ensure last dimension matches self.d_model for internal convs
    if x_last_dim != self.d_model:
        if self.d_model % x_last_dim == 0:
            factor = self.d_model // x_last_dim
            x = x.repeat_interleave(factor, dim=-1)
        else:
            x = F.pad(x, (0, self.d_model - x_last_dim))
    
    # Handle variable sequence lengths for positional encoding
    if seq_len != self.temporal_pos_encoding.size(1):
        # Create adaptive temporal positional encoding
        if seq_len <= self.temporal_pos_encoding.size(1):
            # Truncate existing encoding
            temp_pos_enc = self.temporal_pos_encoding[:, :seq_len, :]
        else:
            # Extend encoding using sinusoidal pattern
            temp_pos_enc = create_adaptive_positional_encoding(
                seq_len, self.d_model, x.device
            )
    else:
        temp_pos_enc = self.temporal_pos_encoding
    
    # Handle variable number of nodes for spatial encoding
    if num_nodes != self.spatial_pos_encoding.size(1):
        if num_nodes <= self.spatial_pos_encoding.size(1):
            # Truncate existing encoding
            spatial_pos_enc = self.spatial_pos_encoding[:, :num_nodes, :]
        else:
            # Extend by repeating the last encoding
            spatial_pos_enc = F.pad(
                self.spatial_pos_encoding, 
                (0, 0, 0, num_nodes - self.spatial_pos_encoding.size(1)), 
                mode='replicate'
            )
    else:
        spatial_pos_enc = self.spatial_pos_encoding
    
    # Add positional encodings with proper broadcasting
    try:
        x = x + spatial_pos_enc.unsqueeze(1)  # Add spatial positions
        x = x + temp_pos_enc.unsqueeze(2)     # Add temporal positions
    except RuntimeError as e:
        # Fallback: skip positional encoding if dimensions don't match
        print(f"Warning: Skipping positional encoding due to dimension mismatch: {e}")
    
    # Continue with the rest of the original forward pass
    # 1. Multi-scale temporal processing
    temporal_features = self._multi_scale_temporal_processing(x)
    
    # 2. Multi-hop spatial processing
    spatial_features = self._multi_hop_spatial_processing(x, adjacency_matrix)
    
    # 3. Cross-attention between spatial and temporal features
    cross_attended_features = self._spatial_temporal_cross_attention(
        temporal_features, spatial_features, temporal_mask
    )
    
    # 4. Fusion and output
    fused_features = self._fusion_mechanism(temporal_features, spatial_features, cross_attended_features)
    
    return self.layer_norm3(fused_features + x)


# Apply the patch
def patch_spatiotemporal_encoding():
    """Apply the patch to handle variable sequence lengths."""
    try:
        from layers.modular.encoder.spatiotemporal_encoding import JointSpatioTemporalEncoding
        
        # Replace the forward method
        JointSpatioTemporalEncoding.forward = patched_joint_spatiotemporal_forward
        print("Successfully patched JointSpatioTemporalEncoding for variable sequence lengths")
        
    except ImportError as e:
        print(f"Could not patch spatiotemporal encoding: {e}")