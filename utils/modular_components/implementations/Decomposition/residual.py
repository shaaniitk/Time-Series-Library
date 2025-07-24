import torch
import torch.nn as nn
from .base import BaseDecomposition
from .series import SeriesDecomposition
from ..config_schemas import ComponentConfig

class ResidualDecomposition(BaseDecomposition):
    """Decomposition with explicit residual component"""
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.base_decomp = SeriesDecomposition(config)
        self.d_model = config.d_model
        self.residual_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model)
        )
    def forward(self, x: torch.Tensor):
        trend, seasonal = self.base_decomp(x)
        residual = x - trend - seasonal
        residual = self.residual_net(residual)
        adjusted_trend = trend + 0.5 * residual
        adjusted_seasonal = seasonal + 0.5 * residual
        return adjusted_trend, adjusted_seasonal
