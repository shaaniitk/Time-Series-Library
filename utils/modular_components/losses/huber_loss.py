"""
Huber Loss implementation for modular architecture.
"""
import torch
from .loss_configs import LossConfig
from ..base_interfaces import BaseLoss

class HuberLoss(BaseLoss):
    """Huber loss for robust regression"""
    def __init__(self, config: LossConfig, delta: float = 1.0):
        super().__init__()
        self.config = config
        self.delta = delta

    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        error = pred - true
        abs_error = torch.abs(error)
        quadratic = torch.where(abs_error < self.delta, 0.5 * error ** 2, self.delta * (abs_error - 0.5 * self.delta))
        return torch.mean(quadratic)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)

    def get_loss_type(self) -> str:
        return "huber"
