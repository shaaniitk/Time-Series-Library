
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDecomposition
from layers.modular.core.logger import logger

class LearnableSeriesDecomposition(BaseDecomposition):
    """
    Enhanced series decomposition with learnable parameters.
    
    Improvements over standard moving average:
    1. Learnable trend extraction weights
    2. Adaptive kernel size selection
    3. Feature-specific decomposition parameters
    """
    
    def __init__(self, input_dim, init_kernel_size=25, max_kernel_size=50):
        super(LearnableSeriesDecomposition, self).__init__()
        logger.info("Initializing LearnableSeriesDecomp")
        
        self.input_dim = input_dim
        self.max_kernel_size = max_kernel_size
        
        # Learnable trend extraction weights: one set of weights per input feature/channel
        self.feature_specific_trend_weights = nn.Parameter(torch.randn(input_dim, 1, max_kernel_size))
        
        # Adaptive kernel size predictor
        self.kernel_predictor = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 4), 1),
            nn.Sigmoid()
        )
        
        self.init_kernel_size = init_kernel_size
        if self.init_kernel_size > 0 and self.init_kernel_size <= self.max_kernel_size:
            nn.init.constant_(self.feature_specific_trend_weights[:, :, :self.init_kernel_size], 1.0 / self.init_kernel_size)
        
    def forward(self, x):
        B, L, D = x.shape
        assert D == self.input_dim, f"Input feature dimension mismatch. Expected {self.input_dim}, got {D}"

        x_global = x.mean(dim=1)
        kernel_logits = self.kernel_predictor(x_global)
        predicted_k_float = (kernel_logits * (self.max_kernel_size - 5) + 5)

        k_float = predicted_k_float.squeeze(-1)
        kernel_sizes = torch.round(k_float).int()
        kernel_sizes = torch.clamp(kernel_sizes, 3, min(self.max_kernel_size, L // 2))
        kernel_sizes = kernel_sizes - (kernel_sizes % 2 == 0).int()
        kernel_sizes = torch.clamp(kernel_sizes, min=3)

        trend = torch.zeros_like(x)
        
        unique_kernels = torch.unique(kernel_sizes)

        for k_val in unique_kernels:
            k = k_val.item()
            indices = torch.where(kernel_sizes == k)[0]
            batch_subset = x[indices]
            
            padding = k // 2
            subset_permuted = batch_subset.transpose(1, 2)
            subset_padded = F.pad(subset_permuted, (padding, padding), mode='replicate')
            
            current_weights_unnormalized = self.feature_specific_trend_weights[:, :, :k]
            normalized_weights = F.softmax(current_weights_unnormalized, dim=2)
            
            trend_conv = F.conv1d(
                subset_padded,
                normalized_weights,
                groups=self.input_dim,
                padding=0
            )
            
            trend[indices] = trend_conv.transpose(1, 2)

        seasonal = x - trend
        return seasonal, trend
