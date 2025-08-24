"""
Adaptive and Bayesian Loss Functions for Modular Autoformer

This module implements adaptive and Bayesian loss functions that enhance
training with trend/seasonal awareness and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
import math

try:
    from layers.Autoformer_EncDec import series_decomp
except ImportError:
    # Fallback simple decomposition if series_decomp not available
    class series_decomp(nn.Module):
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


class AdaptiveAutoformerLoss(nn.Module):
    """
    Adaptive loss function that dynamically weights trend and seasonal components.
    
    Features:
    - Learnable trend/seasonal weights
    - Multiple base loss options
    - Component-wise loss tracking
    - Adaptive weighting based on data characteristics
    
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, base_loss='mse', moving_avg=25, initial_trend_weight=1.0, 
                 initial_seasonal_weight=1.0, adaptive_weights=True):
        super(AdaptiveAutoformerLoss, self).__init__()
        
        self.base_loss = base_loss
        self.adaptive_weights = adaptive_weights
        self.output_dim_multiplier = 1
        
        # Decomposition for loss calculation
        self.decomp = series_decomp(kernel_size=moving_avg)
        
        # Learnable weight parameters
        if adaptive_weights:
            self.trend_weight = nn.Parameter(torch.tensor(initial_trend_weight))
            self.seasonal_weight = nn.Parameter(torch.tensor(initial_seasonal_weight))
        else:
            self.register_buffer('trend_weight', torch.tensor(initial_trend_weight))
            self.register_buffer('seasonal_weight', torch.tensor(initial_seasonal_weight))
        
        # Base loss function
        self.loss_fn = self._get_loss_function(base_loss)
        
    def _get_loss_function(self, loss_name):
        """Get the base loss function"""
        if loss_name == 'mse':
            return F.mse_loss
        elif loss_name == 'mae':
            return F.l1_loss
        elif loss_name == 'huber':
            return F.huber_loss
        elif loss_name == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            return F.mse_loss
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor, 
                return_components: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute adaptive loss with trend/seasonal decomposition.
        
        Args:
            pred: [B, L, D] predicted values
            true: [B, L, D] ground truth values
            return_components: whether to return component losses
            
        Returns:
            total_loss: scalar loss value
            components: dict with component losses (if return_components=True)
        """
        # Decompose both predictions and ground truth
        pred_seasonal, pred_trend = self.decomp(pred)
        true_seasonal, true_trend = self.decomp(true)
        
        # Compute component-wise losses
        trend_loss = self.loss_fn(pred_trend, true_trend, reduction='mean')
        seasonal_loss = self.loss_fn(pred_seasonal, true_seasonal, reduction='mean')
        
        # Apply adaptive weighting with softplus for positivity
        if self.adaptive_weights:
            trend_w = F.softplus(self.trend_weight)
            seasonal_w = F.softplus(self.seasonal_weight)
        else:
            trend_w = self.trend_weight
            seasonal_w = self.seasonal_weight
        
        # Total adaptive loss
        total_loss = trend_w * trend_loss + seasonal_w * seasonal_loss
        
        if return_components:
            components = {
                'total_loss': total_loss.item(),
                'trend_loss': trend_loss.item(),
                'seasonal_loss': seasonal_loss.item(),
                'trend_weight': trend_w.item(),
                'seasonal_weight': seasonal_w.item()
            }
            return total_loss, components
        
        return total_loss


class FrequencyAwareLoss(nn.Module):
    """
    Frequency-aware loss that emphasizes different frequency components.
    
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, freq_bands=None, band_weights=None, base_loss='mse'):
        super(FrequencyAwareLoss, self).__init__()
        
        # Default frequency bands (low, medium, high)
        self.freq_bands = freq_bands or [(0.0, 0.1), (0.1, 0.4), (0.4, 0.5)]
        self.band_weights = band_weights or [1.0, 1.0, 1.0]
        self.base_loss = base_loss
        self.output_dim_multiplier = 1
        
        # Learnable band weights
        self.learnable_weights = nn.Parameter(torch.tensor(self.band_weights, dtype=torch.float32))
        
        # Base loss function
        self.loss_fn = self._get_loss_function(base_loss)
    
    def _get_loss_function(self, loss_name):
        """Get the base loss function"""
        if loss_name == 'mse':
            return F.mse_loss
        elif loss_name == 'mae':
            return F.l1_loss
        else:
            return F.mse_loss
    
    def _frequency_decompose(self, x):
        """Decompose signal into frequency bands"""
        # FFT
        x_freq = torch.fft.rfft(x, dim=-2)  # FFT along time dimension
        
        freq_components = []
        seq_len = x.shape[-2]
        
        for low, high in self.freq_bands:
            # Frequency indices
            low_idx = int(low * seq_len)
            high_idx = int(high * seq_len)
            
            # Create frequency mask
            mask = torch.zeros_like(x_freq)
            mask[..., low_idx:high_idx, :] = 1.0
            
            # Apply mask and inverse FFT
            masked_freq = x_freq * mask
            component = torch.fft.irfft(masked_freq, n=seq_len, dim=-2)
            freq_components.append(component)
        
        return freq_components
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-aware loss.
        
        Args:
            pred: [B, L, D] predicted values
            true: [B, L, D] ground truth values
            
        Returns:
            Loss value
        """
        # Decompose into frequency bands
        pred_components = self._frequency_decompose(pred)
        true_components = self._frequency_decompose(true)
        
        # Compute weighted loss for each band
        total_loss = 0.0
        weights = F.softmax(self.learnable_weights, dim=0)
        
        for i, (pred_comp, true_comp) in enumerate(zip(pred_components, true_components)):
            band_loss = self.loss_fn(pred_comp, true_comp, reduction='mean')
            total_loss += weights[i] * band_loss
        
        return total_loss


class BayesianLoss(nn.Module):
    """
    Base class for Bayesian loss functions that incorporate uncertainty.
    
    This class can wrap any standard loss function and add Bayesian regularization
    terms for proper uncertainty quantification training.
    
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, base_loss_fn, kl_weight=1e-5, uncertainty_weight=0.1):
        super(BayesianLoss, self).__init__()
        
        self.base_loss_fn = base_loss_fn
        self.kl_weight = kl_weight
        self.uncertainty_weight = uncertainty_weight
        self.output_dim_multiplier = 1
        
    def _compute_uncertainty_regularization(self, uncertainty, pred, true):
        """Compute uncertainty regularization term."""
        # Penalize extreme uncertainties (too high or too low)
        uncertainty_reg = torch.mean(torch.log(uncertainty + 1e-8) ** 2)
        
        # Penalize uncertainty that doesn't match prediction error
        pred_error = torch.abs(pred - true)
        uncertainty_mismatch = torch.mean((uncertainty - pred_error) ** 2)
        
        return uncertainty_reg + uncertainty_mismatch
        
    def forward(self, model, pred_result, true, **kwargs):
        """
        Compute Bayesian loss including uncertainty regularization.
        
        Args:
            model: Bayesian model (for KL divergence extraction)
            pred_result: Either tensor (standard) or dict (with uncertainty)
            true: Ground truth tensor
            **kwargs: Additional arguments for base loss
            
        Returns:
            Dict with loss components
        """
        # Extract prediction and uncertainty if available
        if isinstance(pred_result, dict):
            pred = pred_result['prediction']
            uncertainty = pred_result.get('uncertainty', None)
        else:
            pred = pred_result
            uncertainty = None
        
        # Compute base loss
        if hasattr(self.base_loss_fn, '__call__'):
            base_loss = self.base_loss_fn(pred, true, **kwargs)
        else:
            base_loss = self.base_loss_fn(pred, true)
        
        # Extract base loss value if it's a tuple/dict
        if isinstance(base_loss, tuple):
            base_loss_value = base_loss[0]
            base_loss_components = base_loss[1] if len(base_loss) > 1 else {}
        elif isinstance(base_loss, dict):
            base_loss_value = base_loss.get('loss', base_loss.get('total_loss', 0))
            base_loss_components = base_loss
        else:
            base_loss_value = base_loss
            base_loss_components = {}
        
        # KL divergence regularization (for Bayesian layers)
        kl_loss = 0.0
        if hasattr(model, 'get_kl_loss'):
            kl_loss = model.get_kl_loss()
        elif hasattr(model, 'kl_divergence'):
            kl_loss = model.kl_divergence()
        
        # Uncertainty regularization
        uncertainty_loss = 0.0
        if uncertainty is not None:
            uncertainty_loss = self._compute_uncertainty_regularization(uncertainty, pred, true)
        
        # Total loss
        total_loss = base_loss_value + self.kl_weight * kl_loss + self.uncertainty_weight * uncertainty_loss
        
        # Prepare result
        result = {
            'total_loss': total_loss,
            'base_loss': base_loss_value,
            'kl_loss': kl_loss,
            'uncertainty_loss': uncertainty_loss,
            'kl_weight': self.kl_weight,
            'uncertainty_weight': self.uncertainty_weight
        }
        
        # Add base loss components
        result.update(base_loss_components)
        
        return result


class BayesianQuantileLoss(BayesianLoss):
    """
    Bayesian quantile loss combining quantile regression with uncertainty quantification.
    
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9], kl_weight=1e-5, uncertainty_weight=0.1):
        # Create quantile loss as base
        base_loss = QuantileLoss(quantiles=quantiles)
        super(BayesianQuantileLoss, self).__init__(base_loss, kl_weight, uncertainty_weight)
        self.quantiles = quantiles


class QuantileLoss(nn.Module):
    """
    Multi-quantile loss for probabilistic forecasting.
    
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.register_buffer('quantile_tensor', torch.tensor(quantiles, dtype=torch.float32))
        self.output_dim_multiplier = 1
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-quantile loss accepting flattened or explicit quantile dimension.

        Accepted prediction shapes:
            1. [B, T, C * Q]  (legacy / flattened)
            2. [B, T, C, Q]    (explicit quantile dim)

        Args:
            predictions: Forecast tensor in one of the accepted shapes.
            targets: Ground truth values [B, T, C].

        Returns:
            Scalar quantile loss averaged over quantiles.
        """
        num_quantiles = len(self.quantiles)

        if predictions.ndim == 4:
            # Shape already [B, T, C, Q]
            batch_size, seq_len, num_features, q_dim = predictions.shape
            if q_dim != num_quantiles:
                raise ValueError(
                    f"Quantile dimension mismatch: got Q={q_dim}, expected {num_quantiles} "
                    f"for quantiles {self.quantiles}"
                )
            pred_quantiles = predictions
        elif predictions.ndim == 3:
            batch_size, seq_len, combined_dim = predictions.shape
            if combined_dim % num_quantiles != 0:
                raise ValueError(
                    "Predictions last dimension not divisible by number of quantiles: "
                    f"{combined_dim} % {num_quantiles} != 0"
                )
            num_features = combined_dim // num_quantiles
            pred_quantiles = predictions.view(batch_size, seq_len, num_features, num_quantiles)
        else:  # pragma: no cover - guard for improper usage
            raise ValueError(
                "QuantileLoss expects predictions of shape [B,T,C*Q] or [B,T,C,Q]; "
                f"received tensor with shape {tuple(predictions.shape)}"
            )

        # Expand targets to [B, T, C, Q]
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, -1, num_quantiles)

        quantile_loss = 0.0
        for i, tau in enumerate(self.quantiles):
            pred_q = pred_quantiles[..., i]
            target_q = targets_expanded[..., i]
            residual = target_q - pred_q
            loss_q = torch.where(
                residual >= 0,
                tau * residual,
                (tau - 1) * residual,
            )
            quantile_loss += loss_q.mean()

        return quantile_loss / num_quantiles


class UncertaintyCalibrationLoss(nn.Module):
    """
    Loss for calibrating uncertainty estimates with actual prediction errors.
    
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, calibration_weight=1.0):
        super(UncertaintyCalibrationLoss, self).__init__()
        self.calibration_weight = calibration_weight
        self.output_dim_multiplier = 1
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty calibration loss.
        
        Args:
            predictions: Model predictions [B, T, C]
            targets: Ground truth [B, T, C]
            uncertainties: Predicted uncertainties [B, T, C]
            
        Returns:
            Calibration loss value
        """
        # Compute actual prediction errors
        pred_errors = torch.abs(predictions - targets)
        
        # Calibration loss: uncertainty should match prediction error
        calibration_loss = F.mse_loss(uncertainties, pred_errors)
        
        # Sharpness penalty: encourage tight uncertainty bounds
        sharpness_penalty = torch.mean(uncertainties)
        
        total_loss = calibration_loss + 0.1 * sharpness_penalty
        
        return self.calibration_weight * total_loss


class KLTuner:
    """
    Utility class for tuning KL loss in Bayesian models.
    Not a loss function itself, but a helper for adaptive KL weighting.
    """
    
    def __init__(self, target_kl_percentage=0.1, min_weight=1e-6, max_weight=1e-1):
        self.target_kl_percentage = target_kl_percentage
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # History tracking
        self.kl_history = []
        self.data_loss_history = []
        self.kl_weight_history = []
        self.kl_percentage_history = []
    
    def compute_kl_contribution(self, data_loss, kl_loss, kl_weight):
        """Compute current KL contribution percentage"""
        total_loss = data_loss + kl_weight * kl_loss
        if total_loss > 0:
            return (kl_weight * kl_loss) / total_loss
        return 0.0
    
    def adaptive_kl_weight(self, data_loss, kl_loss, current_weight):
        """Adaptively adjust KL weight to target percentage"""
        if kl_loss <= 0:
            return current_weight
            
        # Calculate what weight would give us target percentage
        target_kl_contribution = self.target_kl_percentage * data_loss / (1 - self.target_kl_percentage)
        optimal_weight = target_kl_contribution / kl_loss
        
        # Clamp to reasonable range and smooth the adjustment
        optimal_weight = torch.clamp(torch.tensor(optimal_weight), self.min_weight, self.max_weight)
        
        # Smooth adjustment (moving average)
        adjustment_rate = 0.1  # How fast to adjust
        new_weight = (1 - adjustment_rate) * current_weight + adjustment_rate * optimal_weight.item()
        
        return new_weight
    
    def annealing_schedule(self, epoch, total_epochs, schedule_type='linear'):
        """Get KL weight based on annealing schedule"""
        progress = epoch / total_epochs
        
        if schedule_type == 'linear':
            weight = self.min_weight + (self.max_weight - self.min_weight) * progress
        elif schedule_type == 'cosine':
            weight = self.min_weight + (self.max_weight - self.min_weight) * (1 - math.cos(math.pi * progress)) / 2
        elif schedule_type == 'exponential':
            weight = self.min_weight * (self.max_weight / self.min_weight) ** progress
        else:
            weight = self.max_weight  # constant
            
        return weight


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
            from layers.modular.losses.adaptive_bayesian_losses import AdaptiveAutoformerLoss
            
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
        except ImportError:
            raise ImportError("AdaptiveAutoformerLoss not found in adaptive_bayesian_losses")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.adaptive_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "adaptive_structural"

# Continuing from previous partial PatchStructuralLoss
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.ps_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "patch_structural"

class DTWAlignmentLoss(BaseLoss):
    """Wraps existing DTWLoss for modular framework"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            from utils.losses import DTWLoss
            self.dtw_loss = DTWLoss()
        except ImportError:
            raise ImportError("DTWLoss not found in utils.losses")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.dtw_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "dtw_alignment"

class MultiScaleTrendLoss(BaseLoss):
    """Wraps existing MultiScaleTrendAwareLoss for modular framework"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            from utils.enhanced_losses import MultiScaleTrendAwareLoss
            self.ms_loss = MultiScaleTrendAwareLoss()
        except ImportError:
            raise ImportError("MultiScaleTrendAwareLoss not found in utils.enhanced_losses")
    
    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.ms_loss(pred, true)
    
    def get_loss_type(self) -> str:
        return "multi_scale_trend"

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
