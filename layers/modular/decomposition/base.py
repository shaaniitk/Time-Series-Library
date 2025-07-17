
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDecomposition(nn.Module, ABC):
    """
    Abstract base class for all series decomposition components.
    """
    def __init__(self):
        super(BaseDecomposition, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Decomposes the input time series.

        Args:
            x (torch.Tensor): The input time series of shape [Batch, Seq_Len, Dims].

        Returns:
            tuple: A tuple containing two tensors:
                   - seasonal (torch.Tensor): The seasonal component.
                   - trend (torch.Tensor): The trend component.
        """
        pass
