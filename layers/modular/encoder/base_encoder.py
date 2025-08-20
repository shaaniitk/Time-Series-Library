import torch.nn as nn
from typing import List, Optional


class BaseEncoder(nn.Module):
    """
    Abstract base class for all encoder implementations.
    """
    def __init__(self):
        super(BaseEncoder, self).__init__()
        
    def forward(self, x, attn_mask=None, **kwargs):
        """
        Forward pass for the encoder.
        
        Args:
            x: Input tensor
            attn_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            tuple: (output, attentions)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class ModularEncoder(nn.Module):
    """
    A modular encoder that takes a list of encoder layers and an optional normalization layer.
    """
    def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None):
        super(ModularEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Forward pass for the ModularEncoder.

        Args:
            x (torch.Tensor): The input tensor.
            attn_mask (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - torch.Tensor: The encoder output.
                   - torch.Tensor or None: The attention weights, if available.
        """
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
