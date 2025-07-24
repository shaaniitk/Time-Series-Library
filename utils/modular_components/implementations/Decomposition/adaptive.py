import torch
import torch.nn as nn
from .base import BaseDecomposition
from .moving_average import MovingAverageDecomposition
from .learnable import LearnableDecomposition
from .fourier import FourierDecomposition
from ..config_schemas import ComponentConfig

class AdaptiveDecomposition(BaseDecomposition):
    """Adaptive decomposition that selects best method"""
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.decomposition_methods = nn.ModuleList([
            MovingAverageDecomposition(config),
            LearnableDecomposition(config),
            FourierDecomposition(config)
        ])
        self.method_selector = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.decomposition_methods)),
            nn.Softmax(dim=-1)
        )
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x_summary = x.mean(dim=1)
        method_weights = self.method_selector(x_summary)
        trend_total = torch.zeros_like(x)
        seasonal_total = torch.zeros_like(x)
        for i, decomp_method in enumerate(self.decomposition_methods):
            trend, seasonal = decomp_method(x)
            trend_total += trend * method_weights[:, i].view(-1, 1, 1)
            seasonal_total += seasonal * method_weights[:, i].view(-1, 1, 1)
        return trend_total, seasonal_total
