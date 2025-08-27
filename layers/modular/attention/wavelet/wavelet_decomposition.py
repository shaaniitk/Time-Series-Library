"""WaveletDecomposition component (split from monolithic wavelet_attention module)."""
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.modular.core.logger import logger  # type: ignore


class WaveletDecomposition(nn.Module):
    """Learnable wavelet decomposition producing multi-resolution components."""

    def __init__(self, input_dim: int, levels: int = 3, kernel_size: int = 4) -> None:
        super().__init__()
        logger.info("Initializing WaveletDecomposition: input_dim=%s levels=%s", input_dim, levels)
        self.levels = levels
        self.kernel_size = kernel_size
        self.low_pass = nn.ModuleList(
            [
                nn.Conv1d(
                    input_dim,
                    input_dim,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    groups=input_dim,
                )
                for _ in range(levels)
            ]
        )
        self.high_pass = nn.ModuleList(
            [
                nn.Conv1d(
                    input_dim,
                    input_dim,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    groups=input_dim,
                )
                for _ in range(levels)
            ]
        )
        self.recon_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        self.output_dim_multiplier = 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)
        components: List[torch.Tensor] = []
        current = x_conv
        for level in range(self.levels):
            low = self.low_pass[level](current)
            high = self.high_pass[level](current)
            components.append(high.transpose(1, 2))
            current = low
        components.append(current.transpose(1, 2))
        weights = F.softmax(self.recon_weights, dim=0)
        reconstructed = torch.zeros_like(x)
        for comp, weight in zip(components, weights):
            if comp.size(1) < L:
                comp_upsampled = F.interpolate(
                    comp.transpose(1, 2), size=L, mode="linear", align_corners=False
                ).transpose(1, 2)
            else:
                comp_upsampled = comp
            reconstructed += comp_upsampled * weight
        return reconstructed, components

__all__ = ["WaveletDecomposition"]
