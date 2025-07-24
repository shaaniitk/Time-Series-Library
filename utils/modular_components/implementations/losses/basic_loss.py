"""
Basic Losses for Modular Autoformer
"""
from ..base_interfaces import BaseLoss
from ..Losses.loss_configs import LossConfig
import torch
import torch.nn as nn
from typing import Union, Dict, Any

class MSELoss(BaseLoss):
    """Mean Squared Error loss for regression tasks"""
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred, true)
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)
    def get_loss_type(self):
        return "mse"

class MAELoss(BaseLoss):
    """Mean Absolute Error loss for regression tasks"""
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(pred, true)
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)
    def get_loss_type(self):
        return "mae"

class HuberLoss(BaseLoss):
    """Huber loss for robust regression"""
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return nn.functional.huber_loss(pred, true)
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)
    def get_loss_type(self):
        return "huber"

class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for classification tasks"""
    def __init__(self, config: Union[LossConfig, Dict[str, Any]], num_classes=None):
        self.config = config
        self.num_classes = num_classes
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(pred, true)
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)
    def get_loss_type(self):
        return "cross_entropy"
