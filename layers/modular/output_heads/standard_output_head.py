
from .base import BaseOutputHead
import torch.nn as nn

class StandardOutputHead(BaseOutputHead):
    """
    A standard output head with a single linear layer.
    """
    def __init__(self, d_model, c_out):
        super(StandardOutputHead, self).__init__()
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        return self.projection(x)
