"""
UNIFIED LOSS COMPONENTS
All loss functions in one place - no duplicates, clean modular structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List, Union
import logging

# Import base interfaces from new framework
from ..base_interfaces import BaseComponent, BaseLoss
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# BASIC LOSS COMPONENTS
# =============================================================================

class MSELoss(BaseLoss):
    """Mean Squared Error Loss"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.mse_loss(predictions, targets, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MAELoss(BaseLoss):
    """Mean Absolute Error Loss"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.l1_loss(predictions, targets, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class HuberLoss(BaseLoss):
    """Huber Loss - robust to outliers"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.delta = getattr(config, 'delta', 1.0) if config else 1.0
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.huber_loss(predictions, targets, delta=self.delta, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# QUANTILE LOSS COMPONENTS
# =============================================================================

class QuantileLoss(BaseLoss):
    """Quantile Loss for probabilistic forecasting"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.quantiles = getattr(config, 'quantiles', [0.1, 0.5, 0.9]) if config else [0.1, 0.5, 0.9]
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        predictions: [batch_size, seq_len, num_quantiles]
        targets: [batch_size, seq_len, 1] or [batch_size, seq_len]
        """
        if targets.dim() == predictions.dim() - 1:
            targets = targets.unsqueeze(-1)
        
        errors = targets - predictions
        loss = torch.zeros_like(predictions)
        
        for i, q in enumerate(self.quantiles):
            loss[..., i] = torch.max(q * errors[..., i], (q - 1) * errors[..., i])
        
        loss = loss.mean(dim=-1)  # Average over quantiles
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PinballLoss(BaseLoss):
    """Pinball Loss (alternative name for Quantile Loss)"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.quantile = getattr(config, 'quantile', 0.5) if config else 0.5
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        errors = targets - predictions
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# DISTRIBUTIONAL LOSS COMPONENTS
# =============================================================================

class GaussianNLLLoss(BaseLoss):
    """Gaussian Negative Log-Likelihood Loss"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.eps = getattr(config, 'eps', 1e-6) if config else 1e-6
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                variance: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        predictions: mean predictions
        targets: ground truth
        variance: predicted variance (if None, assumes unit variance)
        """
        if variance is None:
            variance = torch.ones_like(predictions)
        
        variance = torch.clamp(variance, min=self.eps)
        
        loss = 0.5 * (torch.log(variance) + ((targets - predictions) ** 2) / variance)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class StudentTLoss(BaseLoss):
    """Student-t Distribution Loss for robust forecasting"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.df = getattr(config, 'df', 3.0) if config else 3.0  # degrees of freedom
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                scale: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        predictions: location parameter (mean)
        targets: ground truth
        scale: scale parameter
        """
        if scale is None:
            scale = torch.ones_like(predictions)
        
        scale = torch.clamp(scale, min=1e-6)
        
        # Student-t negative log-likelihood
        normalized_error = (targets - predictions) / scale
        
        loss = (torch.log(scale) + 
                0.5 * (self.df + 1) * torch.log(1 + normalized_error ** 2 / self.df))
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# ADVANCED LOSS COMPONENTS
# =============================================================================

class FocalLoss(BaseLoss):
    """Focal Loss for handling imbalanced data"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.alpha = getattr(config, 'alpha', 1.0) if config else 1.0
        self.gamma = getattr(config, 'gamma', 2.0) if config else 2.0
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ce_loss = F.mse_loss(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if mask is not None:
            focal_loss = focal_loss * mask
            if self.reduction == 'mean':
                return focal_loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return focal_loss.sum()
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SmoothedMAE(BaseLoss):
    """Smoothed Mean Absolute Error"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.beta = getattr(config, 'beta', 1.0) if config else 1.0
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.abs(predictions - targets)
        
        # Smooth transition between L2 and L1
        loss = torch.where(diff < self.beta,
                          0.5 * diff ** 2 / self.beta,
                          diff - 0.5 * self.beta)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BalancedMAE(BaseLoss):
    """Balanced MAE that handles different scales"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize by target magnitude to balance different scales
        target_magnitude = torch.abs(targets) + 1e-8
        normalized_error = torch.abs(predictions - targets) / target_magnitude
        
        if mask is not None:
            normalized_error = normalized_error * mask
            if self.reduction == 'mean':
                return normalized_error.sum() / mask.sum()
            elif self.reduction == 'sum':
                return normalized_error.sum()
        
        if self.reduction == 'mean':
            return normalized_error.mean()
        elif self.reduction == 'sum':
            return normalized_error.sum()
        return normalized_error


# =============================================================================
# TIME SERIES SPECIFIC LOSSES
# =============================================================================

class MAPELoss(BaseLoss):
    """Mean Absolute Percentage Error"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.eps = getattr(config, 'eps', 1e-8) if config else 1e-8
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        percentage_error = torch.abs((targets - predictions) / (torch.abs(targets) + self.eps))
        
        if mask is not None:
            percentage_error = percentage_error * mask
            if self.reduction == 'mean':
                return percentage_error.sum() / mask.sum()
            elif self.reduction == 'sum':
                return percentage_error.sum()
        
        if self.reduction == 'mean':
            return percentage_error.mean()
        elif self.reduction == 'sum':
            return percentage_error.sum()
        return percentage_error


class SMAPELoss(BaseLoss):
    """Symmetric Mean Absolute Percentage Error"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.eps = getattr(config, 'eps', 1e-8) if config else 1e-8
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        numerator = torch.abs(targets - predictions)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2 + self.eps
        smape = numerator / denominator
        
        if mask is not None:
            smape = smape * mask
            if self.reduction == 'mean':
                return smape.sum() / mask.sum()
            elif self.reduction == 'sum':
                return smape.sum()
        
        if self.reduction == 'mean':
            return smape.mean()
        elif self.reduction == 'sum':
            return smape.sum()
        return smape


class DTWLoss(BaseLoss):
    """Dynamic Time Warping Loss for sequence alignment"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.gamma = getattr(config, 'gamma', 1.0) if config else 1.0
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
    def _compute_dtw_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simplified differentiable DTW distance"""
        n, m = x.shape[1], y.shape[1]
        
        # Distance matrix
        dist_matrix = torch.cdist(x, y, p=2)
        
        # DP matrix
        D = torch.full((n + 1, m + 1), float('inf'), device=x.device)
        D[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i-1, j-1]
                D[i, j] = cost + torch.min(torch.stack([D[i-1, j], D[i, j-1], D[i-1, j-1]]))
        
        return D[n, m]
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = predictions.shape[0]
        losses = []
        
        for i in range(batch_size):
            pred_seq = predictions[i:i+1]
            target_seq = targets[i:i+1]
            
            if mask is not None:
                # Apply mask to sequences
                seq_mask = mask[i]
                valid_length = seq_mask.sum().int()
                pred_seq = pred_seq[:, :valid_length]
                target_seq = target_seq[:, :valid_length]
            
            dtw_dist = self._compute_dtw_distance(pred_seq, target_seq)
            losses.append(dtw_dist)
        
        loss_tensor = torch.stack(losses)
        
        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        return loss_tensor


# =============================================================================
# ADAPTIVE LOSS COMPONENTS
# =============================================================================

class AdaptiveLoss(BaseLoss):
    """Adaptive loss that combines multiple loss functions"""
    
    def __init__(self, config: ComponentConfig = None):
        super().__init__()
        self.loss_types = getattr(config, 'loss_types', ['mse', 'mae']) if config else ['mse', 'mae']
        self.reduction = getattr(config, 'reduction', 'mean') if config else 'mean'
        
        # Initialize sub-losses
        self.losses = nn.ModuleDict()
        for loss_type in self.loss_types:
            if loss_type == 'mse':
                self.losses[loss_type] = MSELoss(config)
            elif loss_type == 'mae':
                self.losses[loss_type] = MAELoss(config)
            elif loss_type == 'huber':
                self.losses[loss_type] = HuberLoss(config)
            # Add more as needed
        
        # Learnable weights for combining losses
        self.loss_weights = nn.Parameter(torch.ones(len(self.loss_types)))
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        total_loss = 0
        weights = F.softmax(self.loss_weights, dim=0)
        
        for i, (loss_name, loss_fn) in enumerate(self.losses.items()):
            loss_value = loss_fn(predictions, targets, mask)
            total_loss += weights[i] * loss_value
        
        return total_loss


# =============================================================================
# COMPATIBILITY LAYER - ADD compute_loss METHOD
# =============================================================================

def add_compute_loss_method(cls):
    """Add compute_loss method to loss classes for compatibility"""
    if not hasattr(cls, 'compute_loss'):
        def compute_loss(self, predictions, targets, mask=None, **kwargs):
            return self.forward(predictions, targets, mask)
        
        cls.compute_loss = compute_loss
    return cls

# Apply compatibility patches to all loss classes
MSELoss = add_compute_loss_method(MSELoss)
MAELoss = add_compute_loss_method(MAELoss)
HuberLoss = add_compute_loss_method(HuberLoss)
QuantileLoss = add_compute_loss_method(QuantileLoss)
PinballLoss = add_compute_loss_method(PinballLoss)
GaussianNLLLoss = add_compute_loss_method(GaussianNLLLoss)
StudentTLoss = add_compute_loss_method(StudentTLoss)
FocalLoss = add_compute_loss_method(FocalLoss)
SmoothedMAE = add_compute_loss_method(SmoothedMAE)
BalancedMAE = add_compute_loss_method(BalancedMAE)
MAPELoss = add_compute_loss_method(MAPELoss)
SMAPELoss = add_compute_loss_method(SMAPELoss)
DTWLoss = add_compute_loss_method(DTWLoss)
AdaptiveLoss = add_compute_loss_method(AdaptiveLoss)


# =============================================================================
# COMPONENT REGISTRY
# =============================================================================

LOSS_REGISTRY = {
    'mse': MSELoss,
    'mae': MAELoss,
    'huber': HuberLoss,
    'quantile': QuantileLoss,
    'pinball': PinballLoss,
    'gaussian_nll': GaussianNLLLoss,
    'student_t': StudentTLoss,
    'focal': FocalLoss,
    'smoothed_mae': SmoothedMAE,
    'balanced_mae': BalancedMAE,
    'mape': MAPELoss,
    'smape': SMAPELoss,
    'dtw': DTWLoss,
    'adaptive': AdaptiveLoss,
}


def get_loss_component(name: str, config: ComponentConfig = None, **kwargs):
    """Factory function to create loss components"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss component: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    
    component_class = LOSS_REGISTRY[name]
    
    if config is None and kwargs:
        # Create config from kwargs for backward compatibility
        config = ComponentConfig(
            component_name=name,
            **kwargs
        )
    
    return component_class(config)


def register_loss_components(registry):
    """Register all loss components with the main registry"""
    for name, component_class in LOSS_REGISTRY.items():
        registry.register('loss', name, component_class)
    
    logger.info(f"Registered {len(LOSS_REGISTRY)} unified loss components")


def list_loss_components():
    """List all available loss components"""
    return list(LOSS_REGISTRY.keys())


logger.info("✅ Unified loss components loaded successfully")
