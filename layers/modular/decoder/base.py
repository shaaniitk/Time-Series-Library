
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for all decoder components.
    """
    def __init__(self):
        super(BaseDecoder, self).__init__()

    @abstractmethod
    def forward(self, 
                x: torch.Tensor, 
                cross: torch.Tensor, 
                x_mask: Optional[torch.Tensor] = None, 
                cross_mask: Optional[torch.Tensor] = None, 
                trend: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the decoder.

        Args:
            x (torch.Tensor): The input tensor for the decoder.
            cross (torch.Tensor): The output from the encoder.
            x_mask (torch.Tensor, optional): The self-attention mask for the decoder. Defaults to None.
            cross_mask (torch.Tensor, optional): The cross-attention mask. Defaults to None.
            trend (torch.Tensor, optional): The trend component. Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - torch.Tensor: The seasonal component output.
                   - torch.Tensor: The trend component output.
        """
        pass
