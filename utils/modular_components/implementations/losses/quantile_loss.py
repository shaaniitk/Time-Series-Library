"""
Quantile and Pinball Losses for Modular Autoformer
"""
from ..base_interfaces import BaseLoss
from ..Losses.loss_configs import QuantileConfig
import torch
from typing import Union, Dict, Any

class QuantileLoss(BaseLoss):
    """Standard Quantile Loss"""
    def __init__(self, config: Union[QuantileConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        q = self.config.quantile
        return torch.mean(torch.max(q * (true - pred), (q - 1) * (true - pred)))
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "quantile"

class PinballLoss(BaseLoss):
    """Pinball Loss for quantile regression"""
    def __init__(self, config: Union[QuantileConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        q = self.config.quantile
        return torch.mean(torch.max(q * (true - pred), (q - 1) * (true - pred)))
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "pinball"
