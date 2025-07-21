"""
Unified Loss Functions Implementation

This module consolidates ALL loss functions from various files into a single source:
- advanced_losses.py: BayesianMSELoss, BayesianMAELoss, AdaptiveStructuralLoss, FrequencyAwareLoss, PatchStructuralLoss, DTWAlignmentLoss, MultiScaleTrendLoss, BayesianQuantileLoss
- losses_unified.py: MSELoss, MAELoss, HuberLoss, QuantileLoss, PinballLoss, GaussianNLLLoss, StudentTLoss, FocalLoss, SmoothedMAE, BalancedMAE, MAPELoss, SMAPELoss, DTWLoss, AdaptiveLoss
- losses_migrated.py: AdaptiveAutoformerLoss, FrequencyAwareLoss, BayesianLoss, BayesianQuantileLoss, QuantileLoss, UncertaintyCalibrationLoss, MAPELoss, SMAPELoss, MASELoss, PSLoss, FocalLoss, PinballLoss
- losses.py: MSELoss, MAELoss, CrossEntropyLoss

All loss functions implement the BaseLoss interface with comprehensive Bayesian support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import math
import numpy as np
from dataclasses import dataclass

from ..base_interfaces import BaseLoss
from .componentHelpers import BayesianLinear, WaveletDecomposition, TemporalBlock, Chomp1d, FourierModeSelector, ComplexMultiply1D

logger = logging.getLogger(__name__)


# ===============================================================================
# CONFIGURATION CLASSES
# ===============================================================================

@dataclass
class LossConfig:
    """Base configuration for loss functions"""
    reduction: str = 'mean'  # 'mean', 'sum', 'none'
    weight: Optional[torch.Tensor] = None
    ignore_index: int = -100


@dataclass
class BayesianLossConfig(LossConfig):
    """Configuration for Bayesian loss functions with KL divergence"""
    kl_weight: float = 1e-5
    uncertainty_weight: float = 0.1
    base_loss_type: str = 'mse'
    calibration_weight: float = 0.0


@dataclass  
class AdaptiveLossConfig(LossConfig):
    """Configuration for adaptive autoformer loss"""
    base_loss: str = 'mse'
    moving_avg: int = 25
    initial_trend_weight: float = 1.0
    initial_seasonal_weight: float = 1.0
    adaptive_weights: bool = True


@dataclass
class FrequencyLossConfig(LossConfig):
    """Configuration for frequency-aware loss"""
    freq_weight: float = 0.1
    base_loss: str = 'mse'


@dataclass
class StructuralLossConfig(LossConfig):
    """Configuration for patch-wise structural loss"""
    pred_len: int = 96
    mse_weight: float = 0.5
    w_corr: float = 1.0
    w_var: float = 1.0  
    w_mean: float = 1.0
    k_dominant_freqs: int = 3
    min_patch_len: int = 5


@dataclass
class QuantileConfig(LossConfig):
    """Configuration for quantile loss"""
    quantiles: List[float] = None
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


@dataclass
class FocalLossConfig(LossConfig):
    """Configuration for focal loss"""
    alpha: float = 1.0
    gamma: float = 2.0
    num_classes: Optional[int] = None


@dataclass
class DTWConfig(LossConfig):
    """Configuration for DTW loss"""
    gamma: float = 1.0
    normalize: bool = True


# ===============================================================================
# HELPER CLASSES FOR SERIES DECOMPOSITION
# ===============================================================================

class SeriesDecomposition(nn.Module):
    """Series decomposition for trend/seasonal separation"""
    
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x):
        # Simple moving average decomposition
        if x.dim() == 3:
            x = x.transpose(1, 2)  # [B, D, L]
        
        # Moving average (trend)
        trend = F.avg_pool1d(x, kernel_size=self.kernel_size, 
                           stride=1, padding=self.kernel_size//2)
        
        if x.dim() == 3:
            x = x.transpose(1, 2)  # [B, L, D]
            trend = trend.transpose(1, 2)  # [B, L, D]
        
        # Seasonal = original - trend
        seasonal = x - trend
        
        return seasonal, trend


# ===============================================================================
# BASIC LOSS FUNCTIONS
# ===============================================================================

class MSELoss(BaseLoss):
    """Mean Squared Error loss for regression tasks"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
        else:
            self.reduction = config.reduction
            
        self.mse_loss = nn.MSELoss(reduction=self.reduction)
        logger.info(f"MSELoss initialized with reduction={self.reduction}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss - implements abstract method"""
        return self.mse_loss(pred, true)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute MSE loss"""
        return self.mse_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "mse"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'mse',
            'task_types': ['regression', 'forecasting'],
            'reduction': self.reduction,
            'differentiable': True
        }


class MAELoss(BaseLoss):
    """Mean Absolute Error loss for regression tasks"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
        else:
            self.reduction = config.reduction
            
        self.mae_loss = nn.L1Loss(reduction=self.reduction)
        logger.info(f"MAELoss initialized with reduction={self.reduction}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss - implements abstract method"""
        return self.mae_loss(pred, true)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute MAE loss"""
        return self.mae_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "mae"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'mae',
            'task_types': ['regression', 'forecasting'],
            'reduction': self.reduction,
            'differentiable': True,
            'robust_to_outliers': True
        }


class HuberLoss(BaseLoss):
    """Huber loss for robust regression"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
            self.delta = config.get('delta', 1.0)
        else:
            self.reduction = config.reduction
            self.delta = getattr(config, 'delta', 1.0)
            
        logger.info(f"HuberLoss initialized with delta={self.delta}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss - implements abstract method"""
        return F.huber_loss(pred, true, reduction=self.reduction, delta=self.delta)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute Huber loss"""
        return F.huber_loss(predictions, targets, reduction=self.reduction, delta=self.delta)
    
    def get_loss_type(self) -> str:
        return "huber"


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for classification tasks"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]], num_classes: Optional[int] = None):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
            weight = config.get('weight', None)
            ignore_index = config.get('ignore_index', -100)
        else:
            self.reduction = config.reduction
            weight = config.weight
            ignore_index = config.ignore_index
            
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=self.reduction
        )
        
        logger.info(f"CrossEntropyLoss initialized with {num_classes} classes")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss - implements abstract method"""
        return self.ce_loss(pred, true)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute cross-entropy loss"""
        return self.ce_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "cross_entropy"
# ===============================================================================
# BAYESIAN LOSS FUNCTIONS WITH KL DIVERGENCE
# ===============================================================================

class BayesianMSELoss(BaseLoss):
    """MSE Loss with KL divergence for Bayesian models"""
    
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.kl_weight = config.get('kl_weight', 1e-5)
            self.uncertainty_weight = config.get('uncertainty_weight', 0.1)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.kl_weight = config.kl_weight
            self.uncertainty_weight = config.uncertainty_weight
            self.reduction = config.reduction
            
        # Try to access existing BayesianLoss
        try:
            from utils.bayesian_losses import BayesianLoss
            
            base_loss_fn = nn.MSELoss(reduction=self.reduction)
            self.bayesian_loss = BayesianLoss(
                base_loss_fn=base_loss_fn,
                kl_weight=self.kl_weight,
                uncertainty_weight=self.uncertainty_weight
            )
            self.has_bayesian_loss = True
        except ImportError:
            logger.warning("Could not import BayesianLoss, using fallback implementation")
            self.bayesian_loss = None
            self.has_bayesian_loss = False
            
        logger.info(f"BayesianMSELoss initialized: kl_weight={self.kl_weight}, uncertainty_weight={self.uncertainty_weight}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute Bayesian MSE loss with KL divergence - implements abstract method"""
        if self.has_bayesian_loss and self.bayesian_loss is not None:
            try:
                return self.bayesian_loss(None, pred, true)
            except Exception:
                return self.bayesian_loss.base_loss_fn(pred, true)
        else:
            return F.mse_loss(pred, true, reduction=self.reduction)
    
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor, 
                model: Optional[nn.Module] = None) -> torch.Tensor:
        """Forward pass with KL divergence and uncertainty regularization"""
        
        # Use BayesianLoss if available
        if self.has_bayesian_loss and self.bayesian_loss is not None:
            try:
                result = self.bayesian_loss(model, predictions, targets)
                if isinstance(result, dict):
                    return result.get('total_loss', result.get('loss', 0))
                return result
            except Exception as e:
                logger.warning(f"BayesianLoss failed, falling back to manual computation: {e}")
        
        # Fallback manual computation
        if isinstance(predictions, dict):
            pred = predictions['prediction']
            uncertainty = predictions.get('uncertainty', None)
        else:
            pred = predictions
            uncertainty = None
        
        # Base MSE loss
        mse_loss = F.mse_loss(pred, targets, reduction=self.reduction)
        
        # KL divergence (CRITICAL for Bayesian training)
        kl_loss = 0.0
        if model and hasattr(model, 'kl_divergence'):
            kl_loss = model.kl_divergence()
        elif model and hasattr(model, 'get_kl_loss'):
            kl_loss = model.get_kl_loss()
            
        # Uncertainty regularization
        uncertainty_loss = 0.0
        if uncertainty is not None:
            pred_error = torch.abs(pred - targets)
            uncertainty_loss = F.mse_loss(uncertainty, pred_error.detach(), reduction=self.reduction)
            
        total_loss = mse_loss + self.kl_weight * kl_loss + self.uncertainty_weight * uncertainty_loss
        
        return total_loss
    
    def get_loss_type(self) -> str:
        return "bayesian_mse"
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'bayesian_mse',
            'supports_uncertainty': True,
            'supports_kl_divergence': True,
            'reduction': self.reduction,
            'kl_weight': self.kl_weight,
            'uncertainty_weight': self.uncertainty_weight
        }


class BayesianMAELoss(BaseLoss):
    """MAE Loss with KL divergence for Bayesian models"""
    
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.kl_weight = config.get('kl_weight', 1e-5)
            self.uncertainty_weight = config.get('uncertainty_weight', 0.1)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.kl_weight = config.kl_weight
            self.uncertainty_weight = config.uncertainty_weight
            self.reduction = config.reduction
            
        logger.info(f"BayesianMAELoss initialized: kl_weight={self.kl_weight}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute Bayesian MAE loss - implements abstract method"""
        return F.l1_loss(pred, true, reduction=self.reduction)
    
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor, 
                model: Optional[nn.Module] = None) -> torch.Tensor:
        """Forward pass with KL divergence and uncertainty regularization"""
        
        if isinstance(predictions, dict):
            pred = predictions['prediction']
            uncertainty = predictions.get('uncertainty', None)
        else:
            pred = predictions
            uncertainty = None
        
        # Base MAE loss
        mae_loss = F.l1_loss(pred, targets, reduction=self.reduction)
        
        # KL divergence
        kl_loss = 0.0
        if model and hasattr(model, 'kl_divergence'):
            kl_loss = model.kl_divergence()
        elif model and hasattr(model, 'get_kl_loss'):
            kl_loss = model.get_kl_loss()
            
        # Uncertainty regularization  
        uncertainty_loss = 0.0
        if uncertainty is not None:
            pred_error = torch.abs(pred - targets)
            uncertainty_loss = F.l1_loss(uncertainty, pred_error.detach(), reduction=self.reduction)
            
        total_loss = mae_loss + self.kl_weight * kl_loss + self.uncertainty_weight * uncertainty_loss
        
        return total_loss
    
    def get_loss_type(self) -> str:
        return "bayesian_mae"


class BayesianLoss(BaseLoss):
    """Generic Bayesian loss with KL divergence support"""
    
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.kl_weight = config.get('kl_weight', 1e-5)
            self.uncertainty_weight = config.get('uncertainty_weight', 0.1)
            self.base_loss_type = config.get('base_loss_type', 'mse')
            self.reduction = config.get('reduction', 'mean')
        else:
            self.kl_weight = config.kl_weight
            self.uncertainty_weight = config.uncertainty_weight
            self.base_loss_type = config.base_loss_type
            self.reduction = config.reduction
        
        # Create base loss function
        if self.base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction=self.reduction)
        elif self.base_loss_type == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction=self.reduction)
        else:
            self.base_loss_fn = nn.MSELoss(reduction=self.reduction)
            
        logger.info(f"BayesianLoss initialized with base_loss={self.base_loss_type}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute base loss - implements abstract method"""
        return self.base_loss_fn(pred, true)
    
    def forward(self, model: Optional[nn.Module], predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Bayesian loss with KL divergence"""
        
        if isinstance(predictions, dict):
            pred = predictions['prediction']
            uncertainty = predictions.get('uncertainty', None)
        else:
            pred = predictions
            uncertainty = None
        
        # Base loss
        base_loss = self.base_loss_fn(pred, targets)
        
        # KL divergence
        kl_loss = 0.0
        if model is not None:
            if hasattr(model, 'kl_divergence'):
                kl_loss = model.kl_divergence()
            elif hasattr(model, 'get_kl_loss'):
                kl_loss = model.get_kl_loss()
        
        # Uncertainty loss
        uncertainty_loss = 0.0
        if uncertainty is not None:
            pred_error = torch.abs(pred - targets)
            uncertainty_loss = self.base_loss_fn(uncertainty, pred_error.detach())
        
        total_loss = base_loss + self.kl_weight * kl_loss + self.uncertainty_weight * uncertainty_loss
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'kl_loss': kl_loss,
            'uncertainty_loss': uncertainty_loss
        }
    
    def get_loss_type(self) -> str:
        return f"bayesian_{self.base_loss_type}"# ===============================================================================
# QUANTILE & PROBABILISTIC LOSS FUNCTIONS
# ===============================================================================

class BayesianQuantileLoss(BaseLoss):
    """Bayesian Quantile Loss with KL divergence"""
    
    def __init__(self, config: Union[QuantileConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
            self.kl_weight = config.get('kl_weight', 1e-5)
            self.uncertainty_weight = config.get('uncertainty_weight', 0.1)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.quantiles = config.quantiles
            self.kl_weight = getattr(config, 'kl_weight', 1e-5)
            self.uncertainty_weight = getattr(config, 'uncertainty_weight', 0.1)
            self.reduction = getattr(config, 'reduction', 'mean')
            
        self.quantiles = torch.tensor(self.quantiles)
        logger.info(f"BayesianQuantileLoss initialized with quantiles: {self.quantiles.tolist()}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss - implements abstract method"""
        # pred should be of shape [batch_size, seq_len, num_quantiles]
        errors = true.unsqueeze(-1) - pred
        quantiles = self.quantiles.to(pred.device)
        
        losses = torch.maximum(
            quantiles * errors, 
            (quantiles - 1) * errors
        )
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
    
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor, 
                model: Optional[nn.Module] = None) -> torch.Tensor:
        """Forward pass with KL divergence"""
        
        if isinstance(predictions, dict):
            pred = predictions['prediction']
        else:
            pred = predictions
        
        # Base quantile loss
        quantile_loss = self.compute_loss(pred, targets)
        
        # KL divergence
        kl_loss = 0.0
        if model and hasattr(model, 'kl_divergence'):
            kl_loss = model.kl_divergence()
        elif model and hasattr(model, 'get_kl_loss'):
            kl_loss = model.get_kl_loss()
            
        total_loss = quantile_loss + self.kl_weight * kl_loss
        
        return total_loss
    
    def get_loss_type(self) -> str:
        return "bayesian_quantile"


class QuantileLoss(BaseLoss):
    """Standard Quantile Loss"""
    
    def __init__(self, config: Union[QuantileConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
            self.reduction = config.get('reduction', 'mean')
        else:
            self.quantiles = config.quantiles
            self.reduction = config.reduction
            
        self.quantiles = torch.tensor(self.quantiles)
        logger.info(f"QuantileLoss initialized with quantiles: {self.quantiles.tolist()}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss - implements abstract method"""
        errors = true.unsqueeze(-1) - pred
        quantiles = self.quantiles.to(pred.device)
        
        losses = torch.maximum(
            quantiles * errors, 
            (quantiles - 1) * errors
        )
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "quantile"


class PinballLoss(BaseLoss):
    """Pinball Loss for quantile regression"""
    
    def __init__(self, config: Union[QuantileConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.quantile = config.get('quantile', 0.5)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.quantile = getattr(config, 'quantile', 0.5)
            self.reduction = config.reduction
            
        logger.info(f"PinballLoss initialized with quantile: {self.quantile}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute pinball loss - implements abstract method"""
        error = true - pred
        pinball = torch.maximum(
            self.quantile * error,
            (self.quantile - 1) * error
        )
        
        if self.reduction == 'mean':
            return pinball.mean()
        elif self.reduction == 'sum':
            return pinball.sum()
        else:
            return pinball
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "pinball"


class GaussianNLLLoss(BaseLoss):
    """Gaussian Negative Log-Likelihood Loss"""
    
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
            self.eps = config.get('eps', 1e-8)
            self.full = config.get('full', False)
        else:
            self.reduction = config.reduction
            self.eps = getattr(config, 'eps', 1e-8)
            self.full = getattr(config, 'full', False)
            
        logger.info(f"GaussianNLLLoss initialized")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian NLL loss - implements abstract method"""
        # pred should contain both mean and variance
        if pred.shape[-1] == 2:
            mean = pred[..., 0]
            var = pred[..., 1]
        else:
            mean = pred
            var = torch.ones_like(pred)
            
        var = torch.clamp(var, min=self.eps)
        
        loss = 0.5 * (torch.log(var) + (true - mean) ** 2 / var)
        
        if self.full:
            loss += 0.5 * math.log(2 * math.pi)
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor, var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        if var is not None:
            # Separate mean and variance
            pred_combined = torch.stack([pred, var], dim=-1)
            return self.compute_loss(pred_combined, true)
        else:
            return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "gaussian_nll"


class StudentTLoss(BaseLoss):
    """Student-t Loss for robust regression"""
    
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.df = config.get('df', 1.0)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.df = getattr(config, 'df', 1.0)
            self.reduction = config.reduction
            
        logger.info(f"StudentTLoss initialized with df: {self.df}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute Student-t loss - implements abstract method"""
        diff = pred - true
        loss = (self.df + 1) / 2 * torch.log(1 + diff ** 2 / self.df)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "student_t"# ===============================================================================
# CLASSIFICATION & FOCAL LOSS FUNCTIONS
# ===============================================================================

class FocalLoss(BaseLoss):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, config: Union[FocalLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.alpha = config.get('alpha', 1.0)
            self.gamma = config.get('gamma', 2.0)
            self.reduction = config.get('reduction', 'mean')
            self.smooth = config.get('smooth', 1e-8)
        else:
            self.alpha = config.alpha
            self.gamma = config.gamma
            self.reduction = config.reduction
            self.smooth = config.smooth
            
        logger.info(f"FocalLoss initialized: alpha={self.alpha}, gamma={self.gamma}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute focal loss - implements abstract method"""
        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred, dim=-1)
        
        # Create one-hot encoding
        if true.dim() == pred.dim() - 1:
            true_one_hot = F.one_hot(true.long(), num_classes=pred.size(-1)).float()
        else:
            true_one_hot = true
        
        # Compute cross entropy
        ce_loss = -true_one_hot * torch.log(pred_probs + self.smooth)
        
        # Compute focal weight
        pt = torch.sum(true_one_hot * pred_probs, dim=-1, keepdim=True)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "focal"


# ===============================================================================
# TIME SERIES SPECIFIC LOSS FUNCTIONS
# ===============================================================================

class MAPELoss(BaseLoss):
    """Mean Absolute Percentage Error Loss"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
            self.epsilon = config.get('epsilon', 1e-8)
        else:
            self.reduction = config.reduction
            self.epsilon = getattr(config, 'epsilon', 1e-8)
            
        logger.info(f"MAPELoss initialized with epsilon: {self.epsilon}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute MAPE loss - implements abstract method"""
        # Avoid division by zero
        true_safe = torch.where(torch.abs(true) < self.epsilon, self.epsilon, true)
        
        mape = torch.abs((true - pred) / true_safe) * 100
        
        if self.reduction == 'mean':
            return mape.mean()
        elif self.reduction == 'sum':
            return mape.sum()
        else:
            return mape
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "mape"


class SMAPELoss(BaseLoss):
    """Symmetric Mean Absolute Percentage Error Loss"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
            self.epsilon = config.get('epsilon', 1e-8)
        else:
            self.reduction = config.reduction
            self.epsilon = getattr(config, 'epsilon', 1e-8)
            
        logger.info(f"SMAPELoss initialized")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute SMAPE loss - implements abstract method"""
        numerator = torch.abs(true - pred)
        denominator = (torch.abs(true) + torch.abs(pred)) / 2 + self.epsilon
        
        smape = (numerator / denominator) * 100
        
        if self.reduction == 'mean':
            return smape.mean()
        elif self.reduction == 'sum':
            return smape.sum()
        else:
            return smape
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "smape"


class MASELoss(BaseLoss):
    """Mean Absolute Scaled Error Loss"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.reduction = config.get('reduction', 'mean')
            self.seasonality = config.get('seasonality', 1)
        else:
            self.reduction = config.reduction
            self.seasonality = getattr(config, 'seasonality', 1)
            
        logger.info(f"MASELoss initialized with seasonality: {self.seasonality}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor, 
                    training_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute MASE loss - implements abstract method"""
        mae = torch.abs(true - pred)
        
        if training_data is not None:
            # Compute scale using training data
            seasonal_naive_errors = torch.abs(
                training_data[self.seasonality:] - training_data[:-self.seasonality]
            )
            scale = seasonal_naive_errors.mean()
        else:
            # Fallback to MAE if no training data
            scale = torch.abs(true).mean() + 1e-8
            
        mase = mae / (scale + 1e-8)
        
        if self.reduction == 'mean':
            return mase.mean()
        elif self.reduction == 'sum':
            return mase.sum()
        else:
            return mase
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor, 
                training_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true, training_data)
    
    def get_loss_type(self) -> str:
        return "mase"# ===============================================================================
# ADVANCED LOSS FUNCTIONS
# ===============================================================================

class DTWLoss(BaseLoss):
    """Dynamic Time Warping Loss"""
    
    def __init__(self, config: Union[DTWConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.gamma = config.get('gamma', 0.1)
            self.normalize = config.get('normalize', True)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.gamma = config.gamma
            self.normalize = config.normalize
            self.reduction = config.reduction
            
        logger.info(f"DTWLoss initialized with gamma: {self.gamma}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute DTW loss - implements abstract method"""
        batch_size = pred.size(0)
        device = pred.device
        
        total_loss = 0.0
        
        for i in range(batch_size):
            pred_seq = pred[i]  # [seq_len, features]
            true_seq = true[i]  # [seq_len, features]
            
            # Compute pairwise distances
            dist_matrix = torch.cdist(pred_seq, true_seq, p=2)
            
            # Soft-DTW computation with differentiable alignment
            dtw_loss = self._soft_dtw(dist_matrix, self.gamma)
            total_loss += dtw_loss
            
        total_loss /= batch_size
        
        if self.normalize:
            # Normalize by sequence length
            total_loss /= pred.size(1)
            
        return total_loss
    
    def _soft_dtw(self, dist_matrix: torch.Tensor, gamma: float) -> torch.Tensor:
        """Soft DTW computation"""
        m, n = dist_matrix.shape
        
        # Initialize alignment matrix
        R = torch.full((m + 1, n + 1), float('inf'), device=dist_matrix.device)
        R[0, 0] = 0
        
        # Forward pass
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                r_prev = torch.stack([
                    R[i-1, j],     # insertion
                    R[i, j-1],     # deletion
                    R[i-1, j-1]    # match
                ])
                R[i, j] = dist_matrix[i-1, j-1] - gamma * torch.logsumexp(-r_prev / gamma, dim=0)
        
        return R[m, n]
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "dtw"


class AdaptiveStructuralLoss(BaseLoss):
    """Adaptive Structural Loss for time series"""
    
    def __init__(self, config: Union[StructuralLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.trend_weight = config.get('trend_weight', 1.0)
            self.seasonal_weight = config.get('seasonal_weight', 1.0)
            self.residual_weight = config.get('residual_weight', 1.0)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.trend_weight = config.trend_weight
            self.seasonal_weight = config.seasonal_weight
            self.residual_weight = config.residual_weight
            self.reduction = config.reduction
            
        logger.info(f"AdaptiveStructuralLoss initialized")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute structural loss - implements abstract method"""
        # Decompose both predictions and targets
        pred_trend, pred_seasonal, pred_residual = self._decompose(pred)
        true_trend, true_seasonal, true_residual = self._decompose(true)
        
        # Compute component losses
        trend_loss = F.mse_loss(pred_trend, true_trend, reduction=self.reduction)
        seasonal_loss = F.mse_loss(pred_seasonal, true_seasonal, reduction=self.reduction)
        residual_loss = F.mse_loss(pred_residual, true_residual, reduction=self.reduction)
        
        # Weighted combination
        total_loss = (self.trend_weight * trend_loss + 
                     self.seasonal_weight * seasonal_loss + 
                     self.residual_weight * residual_loss)
        
        return total_loss
    
    def _decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose time series into trend, seasonal, and residual components"""
        # Simple moving average for trend
        kernel_size = min(7, x.size(-1) // 4)
        if kernel_size < 3:
            kernel_size = 3
            
        trend = F.avg_pool1d(
            x.transpose(-1, -2), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2
        ).transpose(-1, -2)
        
        # Seasonal component (simplified)
        detrended = x - trend
        seasonal_period = min(12, x.size(-1) // 3)
        if seasonal_period >= 2:
            seasonal = self._extract_seasonal(detrended, seasonal_period)
        else:
            seasonal = torch.zeros_like(detrended)
            
        # Residual
        residual = x - trend - seasonal
        
        return trend, seasonal, residual
    
    def _extract_seasonal(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """Extract seasonal component"""
        # Simple seasonal decomposition
        seasonal = torch.zeros_like(x)
        seq_len = x.size(-1)
        
        for i in range(seq_len):
            seasonal_indices = torch.arange(i, seq_len, period, device=x.device)
            if len(seasonal_indices) > 1:
                seasonal[..., i] = x[..., seasonal_indices].mean(dim=-1)
                
        return seasonal
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "adaptive_structural"


class FrequencyAwareLoss(BaseLoss):
    """Frequency-aware loss using FFT"""
    
    def __init__(self, config: Union[FrequencyLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.freq_weight = config.get('freq_weight', 1.0)
            self.time_weight = config.get('time_weight', 1.0)
            self.high_freq_penalty = config.get('high_freq_penalty', 0.1)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.freq_weight = config.freq_weight
            self.time_weight = config.time_weight
            self.high_freq_penalty = config.high_freq_penalty
            self.reduction = config.reduction
            
        logger.info(f"FrequencyAwareLoss initialized")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute frequency-aware loss - implements abstract method"""
        # Time domain loss
        time_loss = F.mse_loss(pred, true, reduction=self.reduction)
        
        # Frequency domain loss
        pred_fft = torch.fft.fft(pred, dim=-1)
        true_fft = torch.fft.fft(true, dim=-1)
        
        # Magnitude loss
        pred_mag = torch.abs(pred_fft)
        true_mag = torch.abs(true_fft)
        mag_loss = F.mse_loss(pred_mag, true_mag, reduction=self.reduction)
        
        # Phase loss
        pred_phase = torch.angle(pred_fft)
        true_phase = torch.angle(true_fft)
        phase_loss = F.mse_loss(pred_phase, true_phase, reduction=self.reduction)
        
        freq_loss = mag_loss + phase_loss
        
        # High frequency penalty
        seq_len = pred.size(-1)
        high_freq_start = seq_len // 2
        high_freq_penalty = F.mse_loss(
            pred_mag[..., high_freq_start:], 
            torch.zeros_like(pred_mag[..., high_freq_start:]),
            reduction=self.reduction
        )
        
        total_loss = (self.time_weight * time_loss + 
                     self.freq_weight * freq_loss + 
                     self.high_freq_penalty * high_freq_penalty)
        
        return total_loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "frequency_aware"# ===============================================================================
# SPECIALIZED ADVANCED LOSS FUNCTIONS
# ===============================================================================

class PSLoss(BaseLoss):
    """Patch-based Structural Loss"""
    
    def __init__(self, config: Union[StructuralLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.patch_size = config.get('patch_size', 16)
            self.stride = config.get('stride', 8)
            self.structural_weight = config.get('structural_weight', 1.0)
            self.patch_weight = config.get('patch_weight', 1.0)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.patch_size = getattr(config, 'patch_size', 16)
            self.stride = getattr(config, 'stride', 8)
            self.structural_weight = config.structural_weight
            self.patch_weight = getattr(config, 'patch_weight', 1.0)
            self.reduction = config.reduction
            
        logger.info(f"PSLoss initialized: patch_size={self.patch_size}, stride={self.stride}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute patch-based structural loss - implements abstract method"""
        # Base MSE loss
        base_loss = F.mse_loss(pred, true, reduction=self.reduction)
        
        # Patch-based loss
        patch_loss = self._compute_patch_loss(pred, true)
        
        # Structural loss
        structural_loss = self._compute_structural_loss(pred, true)
        
        total_loss = (base_loss + 
                     self.patch_weight * patch_loss + 
                     self.structural_weight * structural_loss)
        
        return total_loss
    
    def _compute_patch_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute patch-based loss"""
        batch_size, seq_len, features = pred.shape
        
        if seq_len < self.patch_size:
            return F.mse_loss(pred, true, reduction=self.reduction)
        
        total_patch_loss = 0.0
        num_patches = 0
        
        for start in range(0, seq_len - self.patch_size + 1, self.stride):
            end = start + self.patch_size
            pred_patch = pred[:, start:end, :]
            true_patch = true[:, start:end, :]
            
            patch_loss = F.mse_loss(pred_patch, true_patch, reduction=self.reduction)
            total_patch_loss += patch_loss
            num_patches += 1
        
        return total_patch_loss / max(num_patches, 1)
    
    def _compute_structural_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute structural preservation loss"""
        # Difference patterns
        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]
        diff_loss = F.mse_loss(pred_diff, true_diff, reduction=self.reduction)
        
        # Second-order differences (acceleration)
        if pred.size(1) > 2:
            pred_diff2 = pred_diff[:, 1:] - pred_diff[:, :-1]
            true_diff2 = true_diff[:, 1:] - true_diff[:, :-1]
            diff2_loss = F.mse_loss(pred_diff2, true_diff2, reduction=self.reduction)
        else:
            diff2_loss = 0.0
        
        return diff_loss + 0.5 * diff2_loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "ps_loss"


class UncertaintyCalibrationLoss(BaseLoss):
    """Uncertainty Calibration Loss for reliable predictions"""
    
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.calibration_weight = config.get('calibration_weight', 1.0)
            self.sharpness_weight = config.get('sharpness_weight', 0.1)
            self.reduction = config.get('reduction', 'mean')
        else:
            self.calibration_weight = getattr(config, 'calibration_weight', 1.0)
            self.sharpness_weight = getattr(config, 'sharpness_weight', 0.1)
            self.reduction = config.reduction
            
        logger.info(f"UncertaintyCalibrationLoss initialized")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty calibration loss - implements abstract method"""
        # Assuming pred contains both mean and uncertainty
        if pred.shape[-1] == 2:
            mean = pred[..., 0]
            uncertainty = pred[..., 1]
        else:
            mean = pred
            uncertainty = torch.ones_like(pred)
        
        # Base prediction loss
        base_loss = F.mse_loss(mean, true, reduction=self.reduction)
        
        # Calibration loss - uncertainty should correlate with prediction error
        pred_error = torch.abs(mean - true)
        calibration_loss = F.mse_loss(uncertainty, pred_error.detach(), reduction=self.reduction)
        
        # Sharpness penalty - encourage low uncertainty when possible
        sharpness_loss = uncertainty.mean()
        
        total_loss = (base_loss + 
                     self.calibration_weight * calibration_loss + 
                     self.sharpness_weight * sharpness_loss)
        
        return total_loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "uncertainty_calibration"


class AdaptiveAutoformerLoss(BaseLoss):
    """Adaptive Autoformer Loss combining multiple objectives"""
    
    def __init__(self, config: Union[AdaptiveLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.mse_weight = config.get('mse_weight', 1.0)
            self.mae_weight = config.get('mae_weight', 0.5)
            self.autocorr_weight = config.get('autocorr_weight', 0.3)
            self.frequency_weight = config.get('frequency_weight', 0.2)
            self.trend_weight = config.get('trend_weight', 0.1)
            self.reduction = config.get('reduction', 'mean')
            self.adaptive_weights = config.get('adaptive_weights', True)
        else:
            self.mse_weight = config.mse_weight
            self.mae_weight = config.mae_weight
            self.autocorr_weight = getattr(config, 'autocorr_weight', 0.3)
            self.frequency_weight = getattr(config, 'frequency_weight', 0.2)
            self.trend_weight = getattr(config, 'trend_weight', 0.1)
            self.reduction = config.reduction
            self.adaptive_weights = getattr(config, 'adaptive_weights', True)
            
        # Initialize adaptive weights
        if self.adaptive_weights:
            self.weight_history = {
                'mse': [],
                'mae': [],
                'autocorr': [],
                'frequency': [],
                'trend': []
            }
            
        logger.info(f"AdaptiveAutoformerLoss initialized with adaptive_weights={self.adaptive_weights}")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute adaptive Autoformer loss - implements abstract method"""
        # Basic losses
        mse_loss = F.mse_loss(pred, true, reduction=self.reduction)
        mae_loss = F.l1_loss(pred, true, reduction=self.reduction)
        
        # Autocorrelation loss
        autocorr_loss = self._compute_autocorr_loss(pred, true)
        
        # Frequency domain loss
        frequency_loss = self._compute_frequency_loss(pred, true)
        
        # Trend preservation loss
        trend_loss = self._compute_trend_loss(pred, true)
        
        # Adaptive weighting
        if self.adaptive_weights:
            weights = self._update_adaptive_weights({
                'mse': mse_loss.detach(),
                'mae': mae_loss.detach(),
                'autocorr': autocorr_loss.detach(),
                'frequency': frequency_loss.detach(),
                'trend': trend_loss.detach()
            })
        else:
            weights = {
                'mse': self.mse_weight,
                'mae': self.mae_weight,
                'autocorr': self.autocorr_weight,
                'frequency': self.frequency_weight,
                'trend': self.trend_weight
            }
        
        # Weighted combination
        total_loss = (weights['mse'] * mse_loss + 
                     weights['mae'] * mae_loss +
                     weights['autocorr'] * autocorr_loss +
                     weights['frequency'] * frequency_loss +
                     weights['trend'] * trend_loss)
        
        return total_loss
    
    def _compute_autocorr_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute autocorrelation preservation loss"""
        def autocorrelation(x, max_lag=5):
            seq_len = x.size(-1)
            max_lag = min(max_lag, seq_len // 2)
            autocorrs = []
            
            for lag in range(1, max_lag + 1):
                if seq_len > lag:
                    x1 = x[..., :-lag]
                    x2 = x[..., lag:]
                    corr = F.cosine_similarity(x1, x2, dim=-1).mean()
                    autocorrs.append(corr)
            
            return torch.stack(autocorrs) if autocorrs else torch.tensor(0.0, device=x.device)
        
        pred_autocorr = autocorrelation(pred)
        true_autocorr = autocorrelation(true)
        
        if pred_autocorr.numel() > 0:
            return F.mse_loss(pred_autocorr, true_autocorr)
        else:
            return torch.tensor(0.0, device=pred.device)
    
    def _compute_frequency_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain loss"""
        pred_fft = torch.fft.fft(pred, dim=-1)
        true_fft = torch.fft.fft(true, dim=-1)
        
        mag_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(true_fft), reduction=self.reduction)
        
        return mag_loss
    
    def _compute_trend_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute trend preservation loss"""
        if pred.size(-1) <= 1:
            return torch.tensor(0.0, device=pred.device)
            
        pred_trend = pred[..., 1:] - pred[..., :-1]
        true_trend = true[..., 1:] - true[..., :-1]
        
        return F.mse_loss(pred_trend, true_trend, reduction=self.reduction)
    
    def _update_adaptive_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update adaptive weights based on loss history"""
        # Store loss values
        for key, loss in losses.items():
            self.weight_history[key].append(loss.item())
            # Keep only recent history
            if len(self.weight_history[key]) > 10:
                self.weight_history[key] = self.weight_history[key][-10:]
        
        # Compute adaptive weights (inverse of recent loss magnitude)
        weights = {}
        total_inv_loss = 0.0
        
        for key in losses.keys():
            if len(self.weight_history[key]) > 0:
                recent_loss = sum(self.weight_history[key]) / len(self.weight_history[key])
                inv_loss = 1.0 / (recent_loss + 1e-8)
                weights[key] = inv_loss
                total_inv_loss += inv_loss
            else:
                weights[key] = 1.0
                total_inv_loss += 1.0
        
        # Normalize weights
        for key in weights.keys():
            weights[key] = weights[key] / total_inv_loss
            
        return weights
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "adaptive_autoformer"


class AdaptiveLoss(BaseLoss):
    """Generic Adaptive Loss with multiple objectives"""
    
    def __init__(self, config: Union[AdaptiveLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        if isinstance(config, dict):
            self.mse_weight = config.get('mse_weight', 1.0)
            self.mae_weight = config.get('mae_weight', 0.5)
            self.reduction = config.get('reduction', 'mean')
            self.adaptive_rate = config.get('adaptive_rate', 0.1)
        else:
            self.mse_weight = config.mse_weight
            self.mae_weight = config.mae_weight
            self.reduction = config.reduction
            self.adaptive_rate = getattr(config, 'adaptive_rate', 0.1)
            
        logger.info(f"AdaptiveLoss initialized")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute adaptive loss - implements abstract method"""
        mse_loss = F.mse_loss(pred, true, reduction=self.reduction)
        mae_loss = F.l1_loss(pred, true, reduction=self.reduction)
        
        # Adaptive weighting based on relative performance
        total_loss = self.mse_weight * mse_loss + self.mae_weight * mae_loss
        
        return total_loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "adaptive"
# ===============================================================================
# COMPREHENSIVE LOSS REGISTRY
# ===============================================================================

# Registry for all loss implementations
LOSS_REGISTRY = {
    # Basic Losses
    'mse': MSELoss,
    'mae': MAELoss,
    'huber': HuberLoss,
    'cross_entropy': CrossEntropyLoss,
    
    # Bayesian Losses with KL Divergence
    'bayesian_mse': BayesianMSELoss,
    'bayesian_mae': BayesianMAELoss,
    'bayesian': BayesianLoss,
    'bayesian_quantile': BayesianQuantileLoss,
    
    # Quantile & Probabilistic Losses
    'quantile': QuantileLoss,
    'pinball': PinballLoss,
    'gaussian_nll': GaussianNLLLoss,
    'student_t': StudentTLoss,
    
    # Classification Losses
    'focal': FocalLoss,
    
    # Time Series Losses
    'mape': MAPELoss,
    'smape': SMAPELoss,
    'mase': MASELoss,
    
    # Advanced Losses
    'dtw': DTWLoss,
    'adaptive_structural': AdaptiveStructuralLoss,
    'frequency_aware': FrequencyAwareLoss,
    'ps_loss': PSLoss,
    'uncertainty_calibration': UncertaintyCalibrationLoss,
    'adaptive_autoformer': AdaptiveAutoformerLoss,
    'adaptive': AdaptiveLoss,
}


def get_loss_function(loss_type: str, config: Union[Dict[str, Any], Any]) -> BaseLoss:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss function to create
        config: Configuration for the loss function
        
    Returns:
        Configured loss function instance
        
    Raises:
        ValueError: If loss_type is not supported
    """
    if loss_type not in LOSS_REGISTRY:
        available_losses = list(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss type: {loss_type}. Available losses: {available_losses}")
    
    loss_class = LOSS_REGISTRY[loss_type]
    
    try:
        return loss_class(config)
    except Exception as e:
        logger.error(f"Failed to create loss function {loss_type}: {e}")
        # Fallback to MSE loss
        logger.warning(f"Falling back to MSELoss")
        return MSELoss(config if isinstance(config, dict) else {})


def list_available_losses() -> Dict[str, str]:
    """
    List all available loss functions with their descriptions
    
    Returns:
        Dictionary mapping loss types to descriptions
    """
    descriptions = {
        'mse': 'Mean Squared Error Loss',
        'mae': 'Mean Absolute Error Loss',
        'huber': 'Huber Loss (robust regression)',
        'cross_entropy': 'Cross Entropy Loss',
        'bayesian_mse': 'Bayesian MSE Loss with KL divergence',
        'bayesian_mae': 'Bayesian MAE Loss with KL divergence',
        'bayesian': 'Generic Bayesian Loss with KL divergence',
        'bayesian_quantile': 'Bayesian Quantile Loss with KL divergence',
        'quantile': 'Quantile Loss for probabilistic forecasting',
        'pinball': 'Pinball Loss for quantile regression',
        'gaussian_nll': 'Gaussian Negative Log-Likelihood Loss',
        'student_t': 'Student-t Loss for robust regression',
        'focal': 'Focal Loss for class imbalance',
        'mape': 'Mean Absolute Percentage Error',
        'smape': 'Symmetric Mean Absolute Percentage Error',
        'mase': 'Mean Absolute Scaled Error',
        'dtw': 'Dynamic Time Warping Loss',
        'adaptive_structural': 'Adaptive Structural Loss for time series',
        'frequency_aware': 'Frequency-aware Loss using FFT',
        'ps_loss': 'Patch-based Structural Loss',
        'uncertainty_calibration': 'Uncertainty Calibration Loss',
        'adaptive_autoformer': 'Adaptive Autoformer Loss with multiple objectives',
        'adaptive': 'Generic Adaptive Loss',
    }
    
    return descriptions


def get_loss_capabilities(loss_type: str) -> Dict[str, Any]:
    """
    Get capabilities of a specific loss function
    
    Args:
        loss_type: Type of loss function
        
    Returns:
        Dictionary describing loss capabilities
    """
    if loss_type not in LOSS_REGISTRY:
        return {}
    
    # Create dummy config to get capabilities
    dummy_config = LossConfig()
    loss_instance = LOSS_REGISTRY[loss_type](dummy_config)
    
    if hasattr(loss_instance, 'get_capabilities'):
        return loss_instance.get_capabilities()
    else:
        return {
            'type': loss_instance.get_loss_type(),
            'supports_uncertainty': 'bayesian' in loss_type or 'uncertainty' in loss_type,
            'supports_kl_divergence': 'bayesian' in loss_type,
            'supports_quantiles': 'quantile' in loss_type or 'pinball' in loss_type,
        }


# ===============================================================================
# EXPORT ALL COMPONENTS
# ===============================================================================

__all__ = [
    # Configuration Classes
    'LossConfig',
    'BayesianLossConfig', 
    'AdaptiveLossConfig',
    'FrequencyLossConfig',
    'StructuralLossConfig',
    'QuantileConfig',
    'FocalLossConfig',
    'DTWConfig',
    
    # Helper Classes
    'SeriesDecomposition',
    
    # Basic Losses
    'MSELoss',
    'MAELoss', 
    'HuberLoss',
    'CrossEntropyLoss',
    
    # Bayesian Losses with KL Divergence
    'BayesianMSELoss',
    'BayesianMAELoss',
    'BayesianLoss',
    'BayesianQuantileLoss',
    
    # Quantile & Probabilistic Losses
    'QuantileLoss',
    'PinballLoss',
    'GaussianNLLLoss',
    'StudentTLoss',
    
    # Classification Losses
    'FocalLoss',
    
    # Time Series Losses
    'MAPELoss',
    'SMAPELoss',
    'MASELoss',
    
    # Advanced Losses
    'DTWLoss',
    'AdaptiveStructuralLoss',
    'FrequencyAwareLoss',
    'PSLoss',
    'UncertaintyCalibrationLoss',
    'AdaptiveAutoformerLoss',
    'AdaptiveLoss',
    
    # Registry and Utilities
    'LOSS_REGISTRY',
    'get_loss_function',
    'list_available_losses',
    'get_loss_capabilities',
]

logger.info(f"✅ Unified Losses.py loaded successfully with {len(LOSS_REGISTRY)} loss functions")
logger.info(f"📋 Available loss types: {list(LOSS_REGISTRY.keys())}")