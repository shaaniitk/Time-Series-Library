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
        """
        Compute quantile loss.
        
        Args:
            predictions: [B, T, C*Q] where Q is number of quantiles
            targets: [B, T, C] target values
            
        Returns:
            Loss value
        """
        batch_size, seq_len, combined_dim = predictions.shape
        num_quantiles = len(self.quantiles)
        num_features = combined_dim // num_quantiles
        
        # Reshape predictions: [B, T, C, Q]
        pred_quantiles = predictions.view(batch_size, seq_len, num_features, num_quantiles)
        
        # Expand targets: [B, T, C, 1] -> [B, T, C, Q]
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, -1, num_quantiles)
        
        # Compute quantile loss
        quantile_loss = 0.0
        for i, tau in enumerate(self.quantiles):
            pred_q = pred_quantiles[:, :, :, i]  # [B, T, C]
            target_q = targets_expanded[:, :, :, i]  # [B, T, C]
            
            residual = target_q - pred_q
            loss_q = torch.where(residual >= 0, 
                               tau * residual, 
                               (tau - 1) * residual)
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
