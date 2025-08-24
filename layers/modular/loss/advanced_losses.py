"""
Advanced Loss Functions for Modular Autoformer

This module implements advanced loss functions copied from utils/ folders
and adapted for the modular component system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math

from .standard_losses import StandardLossWrapper


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE) loss.
    
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self):
        super(MAPELoss, self).__init__()
        self.output_dim_multiplier = 1

    def forward(self, forecast: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute MAPE loss.
        
        Args:
            forecast: Forecast values. Shape: [batch, time, features]
            target: Target values. Shape: [batch, time, features]
            mask: Optional 0/1 mask. Shape: [batch, time, features]
            
        Returns:
            Loss value
        """
        if mask is None:
            mask = torch.ones_like(target)
            
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class SMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) loss.
    
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self):
        super(SMAPELoss, self).__init__()
        self.output_dim_multiplier = 1

    def forward(self, forecast: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute sMAPE loss.
        
        Args:
            forecast: Forecast values. Shape: [batch, time, features]
            target: Target values. Shape: [batch, time, features]
            mask: Optional 0/1 mask. Shape: [batch, time, features]
            
        Returns:
            Loss value
        """
        if mask is None:
            mask = torch.ones_like(target)
            
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                            torch.abs(forecast.data) + torch.abs(target.data)) * mask)


class MASELoss(nn.Module):
    """
    Mean Absolute Scaled Error (MASE) loss.
    
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, freq: int = 1):
        super(MASELoss, self).__init__()
        self.freq = freq
        self.output_dim_multiplier = 1

    def forward(self, forecast: torch.Tensor, target: torch.Tensor,
                insample: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute MASE loss.
        
        Args:
            forecast: Forecast values. Shape: [batch, time_o, features]
            target: Target values. Shape: [batch, time_o, features]
            insample: Insample values. Shape: [batch, time_i, features]
            mask: Optional 0/1 mask. Shape: [batch, time_o, features]
            
        Returns:
            Loss value
        """
        if mask is None:
            mask = torch.ones_like(target)
            
        if insample is None:
            # Fallback to MAE if no insample data
            return F.l1_loss(forecast, target)
            
        # Compute naive forecast error (seasonal naive)
        masep = torch.mean(torch.abs(insample[:, self.freq:] - insample[:, :-self.freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)


class PSLoss(nn.Module):
    """
    Patch-wise Structural Loss (PS Loss) for time series forecasting.
    
    Combines a point-wise loss (e.g., MSE) with structural losses
    calculated over patches of the time series.
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self,
                 pred_len: int,
                 point_wise_loss_fn: nn.Module = None,
                 mse_weight: float = 0.5,
                 w_corr: float = 1.0,
                 w_var: float = 1.0,
                 w_mean: float = 1.0,
                 k_dominant_freqs: int = 3,
                 min_patch_len: int = 5,
                 use_learnable_weights: bool = False,
                 eps: float = 1e-8):
        super(PSLoss, self).__init__()
        self.pred_len = pred_len
        self.point_wise_loss_fn = point_wise_loss_fn or nn.MSELoss()
        self.mse_weight = mse_weight
        self.k_dominant_freqs = k_dominant_freqs
        self.min_patch_len = max(2, min_patch_len)
        self.eps = eps
        self.output_dim_multiplier = 1

        if use_learnable_weights:
            self.raw_w_corr = nn.Parameter(torch.tensor(float(w_corr)))
            self.raw_w_var = nn.Parameter(torch.tensor(float(w_var)))
            self.raw_w_mean = nn.Parameter(torch.tensor(float(w_mean)))
        else:
            self.register_buffer('fixed_w_corr', torch.tensor(float(w_corr)))
            self.register_buffer('fixed_w_var', torch.tensor(float(w_var)))
            self.register_buffer('fixed_w_mean', torch.tensor(float(w_mean)))
        self.use_learnable_weights = use_learnable_weights

    def _fourier_adaptive_patching(self, series: torch.Tensor) -> List[Tuple[int, int]]:
        """Fourier-based Adaptive Patching (FAP) for a single time series."""
        seq_len = series.shape[0]
        patch_indices = []

        if seq_len < self.min_patch_len * 2:
            if seq_len >= self.min_patch_len:
                patch_indices.append((0, seq_len))
            return patch_indices

        # FFT
        xf = torch.fft.rfft(series)
        amplitudes = torch.abs(xf)

        if amplitudes.shape[0] <= 1:
            if seq_len >= self.min_patch_len:
                patch_indices.append((0, seq_len))
            return patch_indices

        actual_freq_amplitudes = amplitudes[1:]
        num_actual_freqs = actual_freq_amplitudes.shape[0]

        if num_actual_freqs == 0:
             if seq_len >= self.min_patch_len:
                patch_indices.append((0, seq_len))
             return patch_indices

        num_to_select = min(self.k_dominant_freqs, num_actual_freqs)
        if num_to_select == 0:
             if seq_len >= self.min_patch_len:
                patch_indices.append((0, seq_len))
             return patch_indices

        _, top_freq_indices_relative = torch.topk(actual_freq_amplitudes, k=num_to_select)
        top_freq_indices_absolute = top_freq_indices_relative + 1

        dominant_periods_float = seq_len / top_freq_indices_absolute.float()
        processed_periods_for_series = set()

        for period_val_float in dominant_periods_float:
            period_len = int(torch.floor(period_val_float))

            if period_len < self.min_patch_len or period_len in processed_periods_for_series:
                continue
            processed_periods_for_series.add(period_len)

            num_segments = seq_len // period_len
            if num_segments == 0:
                continue

            for seg_idx in range(num_segments):
                start = seg_idx * period_len
                end = (seg_idx + 1) * period_len
                if end - start >= self.min_patch_len:
                    patch_indices.append((start, end))
            
            remaining_start = num_segments * period_len
            if seq_len - remaining_start >= self.min_patch_len:
                patch_indices.append((remaining_start, seq_len))
        
        if not patch_indices and seq_len >= self.min_patch_len:
            patch_indices.append((0, seq_len))
            
        return list(set(patch_indices))

    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient."""
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        x_var = torch.sum((x - x_mean) ** 2)
        y_var = torch.sum((y - y_mean) ** 2)
        
        denominator = torch.sqrt(x_var * y_var) + self.eps
        correlation = numerator / denominator
        
        return correlation

    def forward(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute PS Loss.
        
        Args:
            forecast: Forecast values. Shape: [batch, time, features]
            target: Target values. Shape: [batch, time, features]
            
        Returns:
            Loss value
        """
        batch_size, seq_len, num_features = forecast.shape
        
        # Point-wise loss
        pointwise_loss = self.point_wise_loss_fn(forecast, target)
        
        # Structural losses over patches
        total_structural_loss = 0.0
        total_patches = 0
        
        # Get weights
        if self.use_learnable_weights:
            w_corr = torch.sigmoid(self.raw_w_corr)
            w_var = torch.sigmoid(self.raw_w_var) 
            w_mean = torch.sigmoid(self.raw_w_mean)
        else:
            w_corr = self.fixed_w_corr
            w_var = self.fixed_w_var
            w_mean = self.fixed_w_mean
        
        for b in range(batch_size):
            for f in range(num_features):
                # Get patches for this time series
                target_series = target[b, :, f]
                forecast_series = forecast[b, :, f]
                
                patches = self._fourier_adaptive_patching(target_series)
                
                for start_idx, end_idx in patches:
                    if end_idx - start_idx < 2:
                        continue
                        
                    target_patch = target_series[start_idx:end_idx]
                    forecast_patch = forecast_series[start_idx:end_idx]
                    
                    # Correlation loss
                    corr_loss = 1.0 - torch.abs(self._pearson_correlation(target_patch, forecast_patch))
                    
                    # Variance loss
                    target_var = torch.var(target_patch)
                    forecast_var = torch.var(forecast_patch)
                    var_loss = torch.abs(target_var - forecast_var) / (target_var + self.eps)
                    
                    # Mean loss
                    target_mean = torch.mean(target_patch)
                    forecast_mean = torch.mean(forecast_patch)
                    mean_loss = torch.abs(target_mean - forecast_mean) / (torch.abs(target_mean) + self.eps)
                    
                    # Combine structural losses
                    patch_structural_loss = w_corr * corr_loss + w_var * var_loss + w_mean * mean_loss
                    total_structural_loss += patch_structural_loss
                    total_patches += 1
        
        # Average structural loss
        if total_patches > 0:
            avg_structural_loss = total_structural_loss / total_patches
        else:
            avg_structural_loss = 0.0
        
        # Combine losses
        total_loss = self.mse_weight * pointwise_loss + (1.0 - self.mse_weight) * avg_structural_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling imbalanced prediction scenarios.
    
    This component declares that it does not change the output dimension.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.output_dim_multiplier = 1
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Model predictions. Shape: [batch, time, features] 
            targets: Target values. Shape: [batch, time, features]
            
        Returns:
            Loss value
        """
        # Compute base loss (MSE for regression)
        ce_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Compute focusing term
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BayesianMSELoss(BaseLoss):
    def __init__(self, config: BayesianLossConfig):
        super().__init__(config)
        self.kl_weight = config.kl_weight
        self.base_loss = nn.MSELoss(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        mean = predictions['mean']
        logvar = predictions['logvar']
        mse = self.base_loss(mean, targets)
        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return mse + self.kl_weight * kl_div

class BayesianMAELoss(BaseLoss):
    def __init__(self, config: BayesianLossConfig):
        super().__init__(config)
        self.kl_weight = config.kl_weight
        self.base_loss = nn.L1Loss(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        mean = predictions['mean']
        logvar = predictions['logvar']
        mae = self.base_loss(mean, targets)
        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return mae + self.kl_weight * kl_div

class AdaptiveStructuralLoss(BaseLoss):
    def __init__(self, config: AdaptiveLossConfig):
        super().__init__(config)
        try:
            from layers.Autoformer_EncDec import AdaptiveAutoformerLoss
        except ImportError:
            AdaptiveAutoformerLoss = nn.MSELoss
        self.loss_fn = AdaptiveAutoformerLoss(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)

class FrequencyAwareLoss(BaseLoss):
    def __init__(self, config: FrequencyLossConfig):
        super().__init__(config)
        try:
            from layers.FrequencyAwareLoss import FrequencyAwareLoss as FAL
        except ImportError:
            FAL = nn.MSELoss
        self.loss_fn = FAL(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)

class PatchStructuralLoss(BaseLoss):
    def __init__(self, config: StructuralLossConfig):
        super().__init__(config)
        try:
            from layers.PSLoss import PSLoss as PSL
        except ImportError:
            PSL = nn.MSELoss
        self.loss_fn = PSL(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)

class DTWAlignmentLoss(BaseLoss):
    def __init__(self, config: StructuralLossConfig):
        super().__init__(config)
        try:
            from layers.DTWLoss import DTWLoss as DTWL
        except ImportError:
            DTWL = nn.MSELoss
        self.loss_fn = DTWL(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)

class MultiScaleTrendLoss(BaseLoss):
    def __init__(self, config: StructuralLossConfig):
        super().__init__(config)
        try:
            from layers.MultiScaleTrendAwareLoss import MultiScaleTrendAwareLoss as MSTAL
        except ImportError:
            MSTAL = nn.MSELoss
        self.loss_fn = MSTAL(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)

class BayesianQuantileLoss(BaseLoss):
    def __init__(self, config: BayesianLossConfig):
        super().__init__(config)
        try:
            from layers.BayesianQuantileLoss import BayesianQuantileLoss as BQL
        except ImportError:
            BQL = nn.MSELoss
        self.loss_fn = BQL(reduction=config.reduction)
        self.output_dim_multiplier = 1

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)
