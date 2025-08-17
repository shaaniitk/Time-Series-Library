
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class BaseEncoderLayer(nn.Module, ABC):
    """
    Abstract base class for all encoder layer components.
    """
    def __init__(self):
        super(BaseEncoderLayer, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

class BaseDecoderLayer(nn.Module, ABC):
    """
    Abstract base class for all decoder layer components.
    """
    def __init__(self):
        super(BaseDecoderLayer, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Optional[torch.Tensor] = None, cross_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
