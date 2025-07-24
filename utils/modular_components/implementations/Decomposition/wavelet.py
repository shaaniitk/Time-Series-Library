import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDecomposition
from ..config_schemas import ComponentConfig

class WaveletDecomposition(BaseDecomposition):
    """Wavelet-based decomposition for multi-scale analysis"""
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.levels = config.custom_params.get('levels', 3)
        self.d_model = config.d_model
        self.wavelet_filters = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_model, 4)) for _ in range(self.levels)
        ])
    def _wavelet_transform(self, x: torch.Tensor, level: int):
        batch_size, seq_len, features = x.shape
        filter_weights = self.wavelet_filters[level]
        x_conv = F.conv1d(
            x.transpose(1, 2),
            filter_weights.unsqueeze(1),
            padding=2,
            groups=features
        )
        approx = x_conv[:, :, ::2]
        detail = x_conv[:, :, 1::2]
        return approx.transpose(1, 2), detail.transpose(1, 2)
    def forward(self, x: torch.Tensor):
        current = x
        details = []
        for level in range(self.levels):
            approx, detail = self._wavelet_transform(current, level)
            details.append(detail)
            current = approx
        trend = current
        if details:
            seasonal = sum(F.interpolate(d, size=x.shape[1], mode='linear', align_corners=False) for d in details)
        else:
            seasonal = torch.zeros_like(x)
        if trend.shape[1] != x.shape[1]:
            trend = F.interpolate(trend, size=x.shape[1], mode='linear', align_corners=False)
        return trend, seasonal
