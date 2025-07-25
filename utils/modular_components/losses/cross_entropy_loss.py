"""
Cross-Entropy Loss implementation for modular architecture.
"""
import torch
from .loss_configs import LossConfig
from ..base_interfaces import BaseLoss

class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for classification tasks"""
    def __init__(self, config: LossConfig, num_classes: int = None):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(pred, true)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute_loss(predictions, targets)

    def get_loss_type(self) -> str:
        return "cross_entropy"
