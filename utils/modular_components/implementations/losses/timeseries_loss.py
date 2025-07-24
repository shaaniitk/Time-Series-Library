"""
Time Series Losses for Modular Autoformer
"""
from ..base_interfaces import BaseLoss
from ..Losses.loss_configs import LossConfig, DTWConfig
import torch
from typing import Union, Dict, Any, Optional

class MAPELoss(BaseLoss):
    """Mean Absolute Percentage Error Loss"""
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((true - pred) / (true + 1e-8)))
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "mape"

class SMAPELoss(BaseLoss):
    """Symmetric Mean Absolute Percentage Error Loss"""
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return torch.mean(2 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "smape"

class MASELoss(BaseLoss):
    """Mean Absolute Scaled Error Loss"""
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor, training_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Placeholder for scaled error logic
        return torch.mean(torch.abs(pred - true))
    def forward(self, pred: torch.Tensor, true: torch.Tensor, training_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.compute_loss(pred, true, training_data)
    def get_loss_type(self):
        return "mase"

class DTWLoss(BaseLoss):
    """Dynamic Time Warping Loss"""
    def __init__(self, config: Union[DTWConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # Placeholder for DTW logic
        return torch.mean(torch.abs(pred - true))
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "dtw"
