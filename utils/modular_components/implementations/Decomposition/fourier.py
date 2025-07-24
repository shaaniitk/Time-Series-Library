import torch
from .base import BaseDecomposition
from ..config_schemas import ComponentConfig

class FourierDecomposition(BaseDecomposition):
    """Fourier-based decomposition for periodic components"""
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.top_k = config.custom_params.get('top_k', 10)
        self.low_freq_threshold = config.custom_params.get('low_freq_threshold', 0.1)
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, features = x.shape
        x_fft = torch.fft.rfft(x, dim=1)
        freq_dim = x_fft.shape[1]
        freq_indices = torch.arange(freq_dim, device=x.device)
        normalized_freq = freq_indices.float() / freq_dim
        trend_mask = (normalized_freq <= self.low_freq_threshold).float()
        seasonal_mask = 1.0 - trend_mask
        trend_fft = x_fft * trend_mask.unsqueeze(0).unsqueeze(-1)
        seasonal_fft = x_fft * seasonal_mask.unsqueeze(0).unsqueeze(-1)
        trend = torch.fft.irfft(trend_fft, n=seq_len, dim=1)
        seasonal = torch.fft.irfft(seasonal_fft, n=seq_len, dim=1)
        return trend, seasonal
