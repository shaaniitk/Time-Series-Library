"""
Mean Squared Error Loss implementation for modular architecture.
"""
import torch
from .loss_configs import LossConfig
from ..base_interfaces import BaseLoss

class MSELoss(BaseLoss):
    """Mean Squared Error loss for regression tasks"""
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - true) ** 2)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)

    def get_loss_type(self) -> str:
        return "mse"
