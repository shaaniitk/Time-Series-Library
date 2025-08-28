
from __future__ import annotations

import torch.nn as nn

class StandardLossWrapper(nn.Module):
    """
    A wrapper for standard PyTorch loss functions like MSELoss.
    """
    def __init__(self, loss_class):
        super().__init__()
        self.loss = loss_class()
        # This component declares that it does not change the output dimension.
        self.output_dim_multiplier = 1
    
    def get_output_multiplier(self) -> int:
        """Losses like MSE/MAE don't change output dims."""
        return 1

    def forward(self, preds, target):
        return self.loss(preds, target)
