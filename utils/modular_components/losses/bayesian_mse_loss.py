"""
Bayesian MSE Loss implementation for modular architecture.
"""
import torch
from .loss_configs import BayesianLossConfig
from ..base_interfaces import BaseLoss

class BayesianMSELoss(BaseLoss):
    """MSE Loss with KL divergence for Bayesian models"""
    def __init__(self, config: BayesianLossConfig):
        super().__init__()
        self.config = config

    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - true) ** 2)  # Add KL divergence logic as needed

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)

    def get_loss_type(self) -> str:
        return "bayesian_mse"
