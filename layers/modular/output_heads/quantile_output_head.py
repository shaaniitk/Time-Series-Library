
from .base import BaseOutputHead
import torch.nn as nn

class QuantileOutputHead(BaseOutputHead):
    """
    An output head for quantile regression.
    """
    def __init__(self, d_model, c_out, num_quantiles):
        super(QuantileOutputHead, self).__init__()
        self.num_quantiles = num_quantiles
        self.projection = nn.Linear(d_model, c_out * num_quantiles, bias=True)

    def forward(self, x):
        return self.projection(x)
