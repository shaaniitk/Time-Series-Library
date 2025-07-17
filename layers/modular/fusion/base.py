
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List

class BaseFusion(nn.Module, ABC):
    """
    Abstract base class for all fusion components.
    """
    def __init__(self):
        super(BaseFusion, self).__init__()

    @abstractmethod
    def forward(self, multi_res_features: List[nn.Module], target_length: int = None) -> nn.Module:
        """
        Fuses the input features.

        Args:
            multi_res_features (List[torch.Tensor]): A list of tensors to fuse.
            target_length (int, optional): The target length of the output. Defaults to None.

        Returns:
            torch.Tensor: The fused output.
        """
        pass
