"""
Classification Losses for Modular Autoformer
"""
from ..base_interfaces import BaseLoss
from ..Losses.loss_configs import FocalLossConfig
import torch
from typing import Union, Dict, Any

class FocalLoss(BaseLoss):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, config: Union[FocalLossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        gamma = self.config.gamma
        alpha = self.config.alpha
        ce_loss = torch.nn.functional.cross_entropy(pred, true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "focal"
