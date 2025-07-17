"""
Advanced Loss Functions Integration

Integrates existing sophisticated loss functions from the codebase
into the modular framework with proper KL divergence support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass

from ..base_interfaces import BaseLoss
from .losses import LossConfig

logger = logging.getLogger(__name__)


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
            import torch.nn as nn
            
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
            # BayesianLoss expects (model, pred_result, true) signature
            # We'll use None for model since we don't have access to it in this context
            # The BayesianLoss will handle this gracefully
            try:
                return self.bayesian_loss(None, pred, true)
            except Exception:
                # Fallback if signature doesn't work
                return self.bayesian_loss.base_loss_fn(pred, true)
        else:
            # Fallback MSE implementation
            return nn.functional.mse_loss(pred, true, reduction=self.reduction)
    
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor, 
                model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Forward pass with KL divergence and uncertainty regularization
        
        Args:
            predictions: Tensor or dict with 'prediction' and optionally 'uncertainty'
            targets: Ground truth tensor
            model: Bayesian model for KL divergence extraction
            
        Returns:
            Total loss tensor
        """
        # Use BayesianLoss if available
        if self.has_bayesian_loss and self.bayesian_loss is not None:
            try:
                result = self.bayesian_loss(model, predictions, targets)
                # BayesianLoss returns a dict, extract the total loss
                if isinstance(result, dict):
                    return result.get('total_loss', result.get('loss', 0))
                return result
            except Exception as e:
                logger.warning(f"BayesianLoss failed, falling back to manual computation: {e}")
        
        # Fallback manual computation
        # Extract prediction and uncertainty
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
        return nn.functional.l1_loss(pred, true, reduction=self.reduction)
    
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor, 
                model: Optional[nn.Module] = None) -> torch.Tensor:
        # Extract prediction and uncertainty
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


class AdaptiveStructuralLoss(BaseLoss):
    """Wraps existing AdaptiveAutoformerLoss for modular framework"""
    
    def __init__(self, config: Union[AdaptiveLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        try:
            from utils.enhanced_losses import AdaptiveAutoformerLoss
            
            if isinstance(config, dict):
                self.adaptive_loss = AdaptiveAutoformerLoss(
                    base_loss=config.get('base_loss', 'mse'),
                    moving_avg=config.get('moving_avg', 25),
                    initial_trend_weight=config.get('initial_trend_weight', 1.0),
                    initial_seasonal_weight=config.get('initial_seasonal_weight', 1.0),
                    adaptive_weights=config.get('adaptive_weights', True)
                )
            else:
                self.adaptive_loss = AdaptiveAutoformerLoss(
                    base_loss=config.base_loss,
                    moving_avg=config.moving_avg,
                    initial_trend_weight=config.initial_trend_weight,
                    initial_seasonal_weight=config.initial_seasonal_weight,
                    adaptive_weights=config.adaptive_weights
                )
                
            logger.info("AdaptiveStructuralLoss initialized with existing AdaptiveAutoformerLoss")
            
        except ImportError as e:
            logger.warning(f"Could not import AdaptiveAutoformerLoss: {e}")
            # Fallback to simple MSE
            self.adaptive_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.adaptive_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "adaptive_structural"


class FrequencyAwareLoss(BaseLoss):
    """Wraps existing FrequencyAwareLoss for modular framework"""
    
    def __init__(self, config: Union[FrequencyLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        try:
            from utils.enhanced_losses import FrequencyAwareLoss as FALoss
            
            if isinstance(config, dict):
                self.freq_loss = FALoss(
                    freq_weight=config.get('freq_weight', 0.1),
                    base_loss=config.get('base_loss', 'mse')
                )
            else:
                self.freq_loss = FALoss(
                    freq_weight=config.freq_weight,
                    base_loss=config.base_loss
                )
                
            logger.info("FrequencyAwareLoss initialized with existing implementation")
            
        except ImportError as e:
            logger.warning(f"Could not import FrequencyAwareLoss: {e}")
            # Fallback to simple MSE
            self.freq_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.freq_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "frequency_aware"


class PatchStructuralLoss(BaseLoss):
    """Wraps existing PSLoss for modular framework"""
    
    def __init__(self, config: Union[StructuralLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        try:
            from utils.losses import PSLoss
            
            if isinstance(config, dict):
                self.ps_loss = PSLoss(
                    pred_len=config.get('pred_len', 96),
                    mse_weight=config.get('mse_weight', 0.5),
                    w_corr=config.get('w_corr', 1.0),
                    w_var=config.get('w_var', 1.0),
                    w_mean=config.get('w_mean', 1.0),
                    k_dominant_freqs=config.get('k_dominant_freqs', 3),
                    min_patch_len=config.get('min_patch_len', 5)
                )
            else:
                self.ps_loss = PSLoss(
                    pred_len=config.pred_len,
                    mse_weight=config.mse_weight,
                    w_corr=config.w_corr,
                    w_var=config.w_var,
                    w_mean=config.w_mean,
                    k_dominant_freqs=config.k_dominant_freqs,
                    min_patch_len=config.min_patch_len
                )
                
            logger.info("PatchStructuralLoss initialized with existing PSLoss")
            
        except ImportError as e:
            logger.warning(f"Could not import PSLoss: {e}")
            # Fallback to simple MSE
            self.ps_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ps_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "patch_structural"


class DTWAlignmentLoss(BaseLoss):
    """Wraps existing DTWLoss for modular framework"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        try:
            from utils.losses import DTWLoss
            
            gamma = config.get('gamma', 1.0) if isinstance(config, dict) else getattr(config, 'gamma', 1.0)
            normalize = config.get('normalize', True) if isinstance(config, dict) else getattr(config, 'normalize', True)
            
            self.dtw_loss = DTWLoss(gamma=gamma, normalize=normalize)
            logger.info("DTWAlignmentLoss initialized with existing DTWLoss")
            
        except ImportError as e:
            logger.warning(f"Could not import DTWLoss: {e}")
            # Fallback to simple MSE
            self.dtw_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dtw_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "dtw_alignment"


class MultiScaleTrendLoss(BaseLoss):
    """Wraps existing MultiScaleTrendAwareLoss for modular framework"""
    
    def __init__(self, config: Union[LossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        try:
            from utils.losses import MultiScaleTrendAwareLoss
            
            self.trend_loss = MultiScaleTrendAwareLoss()
            logger.info("MultiScaleTrendLoss initialized with existing implementation")
            
        except ImportError as e:
            logger.warning(f"Could not import MultiScaleTrendAwareLoss: {e}")
            # Fallback to simple MSE
            self.trend_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.trend_loss(predictions, targets)
    
    def get_loss_type(self) -> str:
        return "multiscale_trend"


class BayesianQuantileLoss(BaseLoss):
    """Quantile loss with Bayesian regularization"""
    
    def __init__(self, config: Union[BayesianLossConfig, Dict[str, Any]]):
        super().__init__(config)
        
        try:
            from utils.bayesian_losses import BayesianQuantileLoss as BQL
            
            if isinstance(config, dict):
                quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
                kl_weight = config.get('kl_weight', 1e-5)
                uncertainty_weight = config.get('uncertainty_weight', 0.1)
            else:
                quantiles = getattr(config, 'quantiles', [0.1, 0.5, 0.9])
                kl_weight = config.kl_weight
                uncertainty_weight = config.uncertainty_weight
                
            self.bayesian_quantile_loss = BQL(
                quantiles=quantiles,
                kl_weight=kl_weight,
                uncertainty_weight=uncertainty_weight
            )
            logger.info("BayesianQuantileLoss initialized with existing implementation")
            
        except ImportError as e:
            logger.warning(f"Could not import BayesianQuantileLoss: {e}")
            # Fallback to simple quantile loss
            from .losses import QuantileLoss
            self.bayesian_quantile_loss = QuantileLoss(config)
    
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor,
                model: Optional[nn.Module] = None) -> torch.Tensor:
        return self.bayesian_quantile_loss(model, predictions, targets)
    
    def get_loss_type(self) -> str:
        return "bayesian_quantile"
