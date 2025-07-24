import torch
import torch.nn as nn
from .base import BaseDecomposition
from ..config_schemas import ComponentConfig

class LearnableDecomposition(BaseDecomposition):
    """Learnable decomposition with neural networks"""
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.hidden_dim = config.custom_params.get('hidden_dim', 64)
        self.trend_net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.d_model)
        )
        self.seasonal_net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.d_model)
        )
    def forward(self, x: torch.Tensor):
        trend = self.trend_net(x)
        seasonal = self.seasonal_net(x)
        residual = x - (trend + seasonal)
        trend = trend + 0.5 * residual
        seasonal = seasonal + 0.5 * residual
        return trend, seasonal
