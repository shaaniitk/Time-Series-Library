
import torch

import torch.nn as nn
from abc import ABC, abstractmethod

class BaseOutputHead(nn.Module, ABC):
    """
    Abstract base class for all output head components.
    """
    def __init__(self):
        super(BaseOutputHead, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the output head.

        Args:
            x (torch.Tensor): The input tensor from the decoder.

        Returns:
            torch.Tensor: The final output tensor.
        """
        pass
