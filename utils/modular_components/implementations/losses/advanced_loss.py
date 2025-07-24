"""
Advanced Losses for Modular Autoformer
"""
from ..base_interfaces import BaseLoss
from ..Losses.loss_configs import AdaptiveLossConfig, FrequencyLossConfig, StructuralLossConfig
import torch
from typing import Union, Dict, Any

class AdaptiveStructuralLoss(BaseLoss):
    """Adaptive Structural Loss for time series"""
    def __init__(self, config: Union[StructuralLossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # Placeholder for decomposition logic
        return torch.mean(torch.abs(pred - true))
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "adaptive_structural"

class FrequencyAwareLoss(BaseLoss):
    """Frequency-aware loss using FFT"""
    def __init__(self, config: Union[FrequencyLossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # Placeholder for FFT logic
        return torch.mean(torch.abs(pred - true))
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(pred, true)
    def get_loss_type(self):
        return "frequency_aware"
