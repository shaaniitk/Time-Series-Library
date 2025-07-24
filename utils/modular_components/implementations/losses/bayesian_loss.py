"""
Bayesian Losses for Modular Autoformer
"""
from ..base_interfaces import BaseLoss
from ..Losses.loss_configs import BayesianLossConfig
import torch
import torch.nn as nn
from typing import Union, Dict, Any, Optional

class BayesianMSELoss(BaseLoss):
    """MSE Loss with KL divergence for Bayesian models"""
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred, true)
    def forward(self, predictions, targets, model: Optional[nn.Module] = None):
        mse = self.compute_loss(predictions, targets)
        kl = model.kl_divergence() if model and hasattr(model, 'kl_divergence') else 0.0
        return mse + self.config.kl_weight * kl
    def get_loss_type(self):
        return "bayesian_mse"

class BayesianMAELoss(BaseLoss):
    """MAE Loss with KL divergence for Bayesian models"""
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(pred, true)
    def forward(self, predictions, targets, model: Optional[nn.Module] = None):
        mae = self.compute_loss(predictions, targets)
        kl = model.kl_divergence() if model and hasattr(model, 'kl_divergence') else 0.0
        return mae + self.config.kl_weight * kl
    def get_loss_type(self):
        return "bayesian_mae"

class BayesianLoss(BaseLoss):
    """Generic Bayesian loss with KL divergence support"""
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        self.config = config
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred, true)
    def forward(self, predictions, targets, model: Optional[nn.Module] = None):
        loss = self.compute_loss(predictions, targets)
        kl = model.kl_divergence() if model and hasattr(model, 'kl_divergence') else 0.0
        return loss + self.config.kl_weight * kl
    def get_loss_type(self):
        return "bayesian"
