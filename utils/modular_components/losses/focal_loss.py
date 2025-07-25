"""
Focal Loss implementation for modular architecture.
"""
import torch
from .loss_configs import FocalLossConfig
from ..base_interfaces import BaseLoss

class FocalLoss(BaseLoss):
    """Focal loss for classification tasks"""
    def __init__(self, config: FocalLossConfig, gamma: float = 2.0):
        super().__init__()
        self.config = config
        self.gamma = gamma

    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(pred, true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)

    def get_loss_type(self) -> str:
        return "focal"
