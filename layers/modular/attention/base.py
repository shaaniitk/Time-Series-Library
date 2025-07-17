
import torch.nn as nn
from typing import Optional, Tuple

class BaseAttention(nn.Module):
    """
    Base class for all attention components.
    This is not an abstract class, as we are reusing components
    from another framework that do not share a common abstract base class.
    """
    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, 
                query: nn.Module, 
                key: nn.Module, 
                value: nn.Module, 
                attn_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        
        raise NotImplementedError("This method should be implemented by subclasses.")
