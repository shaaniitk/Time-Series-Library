import torch
import torch.nn.functional as F
from .base import BaseDecomposition
from ..config_schemas import ComponentConfig

class MovingAverageDecomposition(BaseDecomposition):
    """Moving average based decomposition"""
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.kernel_size = config.custom_params.get('kernel_size', 25)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.padding = self.kernel_size // 2
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, features = x.shape
        x_permuted = x.permute(0, 2, 1)
        trend = F.avg_pool1d(
            F.pad(x_permuted, (self.padding, self.padding), mode='replicate'),
            kernel_size=self.kernel_size,
            stride=1
        )
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal
