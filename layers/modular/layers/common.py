import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    A simple feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(FeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        """
        Forward pass for the FeedForward network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Handle both 3D [B, L, D] and 4D [B, L, N, D] inputs
        original_shape = x.shape
        if x.dim() == 4:
            B, L, N, D = x.shape
            x = x.view(B * L, N, D)  # Reshape to 3D for processing
            
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        result = x + y
        
        # Reshape back to original format if needed
        if len(original_shape) == 4:
            result = result.view(original_shape)
            
        return result
