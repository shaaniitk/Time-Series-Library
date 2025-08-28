import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class WaveletDecomposition(nn.Module):
    """
    Learnable 1D wavelet decomposition. This module acts as a filter bank,
    separating a signal into low-pass and high-pass components at multiple levels.
    """
    def __init__(self, d_model: int, levels: int = 3, kernel_size: int = 4):
        super().__init__()
        self.levels = levels
        self.kernel_size = kernel_size
        
        # Filters for low-pass and high-pass components at each level
        self.low_pass = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size, stride=2, padding=kernel_size // 2, groups=d_model)
            for _ in range(levels)
        ])
        self.high_pass = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size, stride=2, padding=kernel_size // 2, groups=d_model)
            for _ in range(levels)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Decomposes the input tensor into a list of components.
        
        Args:
            x (torch.Tensor): Input of shape [B, L, D].
        
        Returns:
            List[torch.Tensor]: A list of high-pass components from finest to
                                coarsest, followed by the final low-pass component.
        """
        x_conv = x.transpose(1, 2) # [B, D, L]
        components: List[torch.Tensor] = []
        current = x_conv
        
        for level in range(self.levels):
            low = self.low_pass[level](current)
            high = self.high_pass[level](current)
            components.append(high.transpose(1, 2)) # Store high-pass details
            current = low # Continue decomposing the low-pass approximation
        
        components.append(current.transpose(1, 2)) # Add final low-pass component
        
        return components