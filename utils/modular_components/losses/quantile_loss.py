"""
Quantile Loss implementation for modular architecture.
"""
import torch
from .loss_configs import QuantileConfig
from ..base_interfaces import BaseLoss

class QuantileLoss(BaseLoss):
    """Standard Quantile Loss"""
    def __init__(self, config: QuantileConfig):
        super().__init__()
        self.config = config

    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # Example quantile loss for tau=0.5
        tau = getattr(self.config, 'tau', 0.5)
        error = true - pred
        return torch.mean(torch.max(tau * error, (tau - 1) * error))

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)

    def get_loss_type(self) -> str:
        return "quantile"
