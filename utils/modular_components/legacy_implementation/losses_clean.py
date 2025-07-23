"""
CLEAN LOSS COMPONENTS
Self-contained loss implementations for the new modular framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseLoss
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)


class MSELoss(BaseLoss):
    """Mean Squared Error Loss"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, true):
        return F.mse_loss(pred, true, reduction=self.reduction)
    
    def compute_loss(self, pred, true, **kwargs):
        return self.forward(pred, true)


class MAELoss(BaseLoss):
    """Mean Absolute Error Loss"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, true):
        return F.l1_loss(pred, true, reduction=self.reduction)
    
    def compute_loss(self, pred, true, **kwargs):
        return self.forward(pred, true)


class HuberLoss(BaseLoss):
    """Huber Loss (Smooth L1)"""
    
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, pred, true):
        return F.smooth_l1_loss(pred, true, reduction=self.reduction, beta=self.delta)
    
    def compute_loss(self, pred, true, **kwargs):
        return self.forward(pred, true)


class QuantileLoss(BaseLoss):
    """Quantile Loss for probabilistic forecasting"""
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9], reduction='mean'):
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction
        
    def forward(self, pred, true):
        """
        pred: [B, L, num_quantiles] predictions
        true: [B, L, 1] ground truth
        """
        if pred.shape[-1] != len(self.quantiles):
            # If pred doesn't match quantiles, use MSE fallback
            return F.mse_loss(pred[..., :1], true, reduction=self.reduction)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            error = true - pred[..., i:i+1]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        
        total_loss = torch.cat(losses, dim=-1).mean()
        return total_loss
    
    def compute_loss(self, pred, true, **kwargs):
        return self.forward(pred, true)


class FocalLoss(BaseLoss):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, true):
        # Convert to probability task if needed
        if pred.shape == true.shape:
            # Regression case - use MSE with focusing
            mse = F.mse_loss(pred, true, reduction='none')
            focal_weight = (1 - torch.exp(-mse)) ** self.gamma
            loss = self.alpha * focal_weight * mse
        else:
            # Classification case
            ce_loss = F.cross_entropy(pred, true, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    def compute_loss(self, pred, true, **kwargs):
        return self.forward(pred, true)


# Registry for clean loss components
LOSS_REGISTRY = {
    'mse': MSELoss,
    'mae': MAELoss,
    'huber': HuberLoss,
    'quantile': QuantileLoss,
    'focal': FocalLoss,
}


def get_loss_component(name: str, config: ComponentConfig = None, **kwargs):
    """Get loss component by name"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss component: {name}")
    
    component_class = LOSS_REGISTRY[name]
    
    if config is not None:
        # Use config parameters
        params = {
            **getattr(config, 'custom_params', {}),
            **kwargs
        }
    else:
        params = kwargs
    
    return component_class(**params)


def register_loss_components(registry):
    """Register all loss components with the registry"""
    for name, component_class in LOSS_REGISTRY.items():
        registry.register('loss', name, component_class)
    
    logger.info(f"Registered {len(LOSS_REGISTRY)} clean loss components")


def list_loss_components():
    """List all available loss components"""
    return list(LOSS_REGISTRY.keys())
