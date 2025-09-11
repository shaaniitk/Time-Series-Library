"""TemporalBlock for TCN-based attention mechanisms."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """A temporal convolution block with residual connections and causal padding.
    
    Parameters
    ----------
    n_inputs : int
        Number of input channels
    n_outputs : int
        Number of output channels
    kernel_size : int
        Convolution kernel size
    stride : int
        Convolution stride
    dilation : int
        Dilation factor
    padding : int
        Padding size
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self) -> None:
        """Initialize weights using normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns
        -------
        torch.Tensor
            Output tensor of same shape
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Removes padding from the end of a sequence to ensure causality."""
    
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding from the end of sequence.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Chomped tensor
        """
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


__all__ = ["TemporalBlock", "Chomp1d"]