
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoder components.
    """
    def __init__(self):
        super(BaseEncoder, self).__init__()

    @abstractmethod
    def forward(self, x: nn.Module, attn_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        """
        The forward pass for the encoder.

        Args:
            x (torch.Tensor): The input tensor.
            attn_mask (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - torch.Tensor: The encoder output.
                   - torch.Tensor or None: The attention weights, if available.
        """
        pass
