from .base import BaseOutputHead
import torch
import torch.nn as nn

class LevelOutputHead(BaseOutputHead):
    def __init__(self, d_model: int, c_out: int, kernel_size: int = 3, use_smoothing: bool = True):
        super(LevelOutputHead, self).__init__()
        self.d_model = d_model
        self.c_out = c_out
        self.use_smoothing = use_smoothing
        
        # Temporal smoothing layer using depthwise separable convolution
        if use_smoothing:
            self.temporal_smoother = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, 
                         padding=kernel_size//2, groups=d_model),  # Depthwise
                nn.Conv1d(d_model, d_model, kernel_size=1),  # Pointwise
                nn.ReLU()
            )
        
        # Final projection layer
        self.projection = nn.Linear(d_model, c_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for level output head.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, c_out)
        """
        # Apply temporal smoothing if enabled
        if self.use_smoothing:
            # Transpose for conv1d: (batch, d_model, seq_len)
            x_transposed = x.transpose(1, 2)
            smoothed = self.temporal_smoother(x_transposed)
            # Transpose back: (batch, seq_len, d_model)
            x = smoothed.transpose(1, 2)
        
        # Apply final projection
        output = self.projection(x)
        return output