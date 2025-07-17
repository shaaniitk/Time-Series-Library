
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
    def forward(self, x: nn.Module, attn_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        pass

class BaseDecoderLayer(nn.Module, ABC):
    """
    Abstract base class for all decoder layer components.
    """
    def __init__(self):
        super(BaseDecoderLayer, self).__init__()

    @abstractmethod
    def forward(self, x: nn.Module, cross: nn.Module, x_mask: Optional[nn.Module] = None, cross_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, nn.Module]:
        pass
