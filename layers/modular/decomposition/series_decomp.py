import torch
import torch.nn as nn


class series_decomp(nn.Module):
    """
    Series decomposition block that separates a time series into trend and seasonal components.
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)
        # Add learnable parameters for the decomposition
        self.trend_weight = nn.Parameter(torch.ones(1))
        self.seasonal_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        # Transpose for AvgPool1d: [batch_size, features, seq_len]
        x_transposed = x.transpose(1, 2)
        
        # Apply moving average to get trend
        trend = self.moving_avg(x_transposed)
        
        # Transpose back: [batch_size, seq_len, features]
        trend = trend.transpose(1, 2)
        
        # Apply learnable weights
        trend = trend * self.trend_weight
        
        # Seasonal component is the residual
        seasonal = (x - trend) * self.seasonal_weight
        
        return seasonal, trend