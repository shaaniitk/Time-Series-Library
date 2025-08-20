"""
Learnable Wavelet Decomposition with trainable filters and orthogonality constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnableWaveletDecomposition(nn.Module):
    """Wavelet decomposition with learnable filters and orthogonality constraints"""
    
    def __init__(self, d_model: int, levels: int = 3, wavelet_length: int = 8, 
                 orthogonality_weight: float = 0.01, padding_mode: str = 'reflect'):
        super().__init__()
        self.levels = levels
        self.wavelet_length = wavelet_length
        self.d_model = d_model
        self.orthogonality_weight = orthogonality_weight
        self.padding_mode = padding_mode
        
        # Initialize with Haar wavelet coefficients for better starting point
        haar_low = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.float32)
        haar_high = torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)], dtype=torch.float32)
        
        # Pad or truncate to desired length
        if wavelet_length > 2:
            haar_low = F.pad(haar_low, (0, wavelet_length - 2))
            haar_high = F.pad(haar_high, (0, wavelet_length - 2))
        elif wavelet_length == 2:
            pass  # Perfect fit
        else:
            haar_low = haar_low[:wavelet_length]
            haar_high = haar_high[:wavelet_length]
        
        # Learnable wavelet filters initialized with Haar
        self.low_pass_filters = nn.ParameterList([
            nn.Parameter(haar_low.clone() + torch.randn(wavelet_length) * 0.01)
            for _ in range(levels)
        ])
        
        self.high_pass_filters = nn.ParameterList([
            nn.Parameter(haar_high.clone() + torch.randn(wavelet_length) * 0.01)
            for _ in range(levels)
        ])
        
        # Projection layers for each decomposition level
        self.level_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(levels + 1)
        ])
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signal [B, L, 1]
        Returns:
            Multi-level decomposition [B, L, d_model]
        """
        B, L, _ = x.shape
        x_signal = x.squeeze(-1)  # [B, L]
        
        decompositions = []
        current_signal = x_signal
        
        for level in range(self.levels):
            # Apply learnable filters with reflection padding for better boundary handling
            padding = self.wavelet_length // 2
            
            # Apply reflection padding
            padded_signal = F.pad(current_signal.unsqueeze(1), (padding, padding), mode=self.padding_mode)
            
            low_pass = F.conv1d(
                padded_signal, 
                self.low_pass_filters[level].view(1, 1, -1)
            ).squeeze(1)
            
            high_pass = F.conv1d(
                padded_signal,
                self.high_pass_filters[level].view(1, 1, -1)
            ).squeeze(1)
            
            # Downsample by 2
            if low_pass.size(1) >= 2:
                low_pass = low_pass[:, ::2]
            if high_pass.size(1) >= 2:
                high_pass = high_pass[:, ::2]
            
            # Store high-frequency component
            decompositions.append(high_pass)
            current_signal = low_pass
            
        # Add final low-frequency component
        decompositions.append(current_signal)
        
        # Project each level to d_model dimensions
        projected_decomps = []
        weights = F.softmax(self.combination_weights, dim=0)
        
        for i, decomp in enumerate(decompositions):
            # Interpolate to original length if needed (using cubic for better reconstruction)
            if decomp.size(1) != L:
                if decomp.size(1) > 1:
                    decomp = F.interpolate(
                        decomp.unsqueeze(1), size=L, mode='linear', align_corners=False
                    ).squeeze(1)
                else:
                    # Handle edge case of single sample
                    decomp = decomp.expand(-1, L)
            
            # Project to d_model
            proj_decomp = self.level_projections[i](decomp.unsqueeze(-1))  # [B, L, d_model]
            
            # Apply learnable weight
            proj_decomp = proj_decomp * weights[i]
            projected_decomps.append(proj_decomp)
        
        # Combine all levels
        combined = torch.stack(projected_decomps, dim=-1).sum(dim=-1)
        
        return combined
    
    def compute_orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization loss for filter constraints"""
        total_loss = 0.0
        
        for level in range(self.levels):
            low_filter = self.low_pass_filters[level]
            high_filter = self.high_pass_filters[level]
            
            # Orthogonality constraint: low and high filters should be orthogonal
            orthogonality_loss = torch.abs(torch.dot(low_filter, high_filter))
            
            # Normalization constraint: filters should have unit norm
            low_norm_loss = torch.abs(torch.norm(low_filter) - 1.0)
            high_norm_loss = torch.abs(torch.norm(high_filter) - 1.0)
            
            total_loss += orthogonality_loss + low_norm_loss + high_norm_loss
        
        return total_loss * self.orthogonality_weight