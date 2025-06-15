# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
from utils.logger import logger
import torch.nn.functional as F
from typing import List, Tuple, Optional


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


def compute_loss(pred, target):
    logger.debug("Computing loss")
    # ...existing code...

class PSLoss(nn.Module):
    """
    Patch-wise Structural Loss (PS Loss) for time series forecasting.
    Combines a point-wise loss (e.g., MSE) with structural losses
    calculated over patches of the time series.
    """
    def __init__(self,
                 pred_len: int,
                 point_wise_loss_fn: nn.Module = nn.MSELoss(),
                 mse_weight: float = 0.5,
                 w_corr: float = 1.0,
                 w_var: float = 1.0,
                 w_mean: float = 1.0,
                 k_dominant_freqs: int = 3,
                 min_patch_len: int = 5,
                 use_learnable_weights: bool = False,
                 eps: float = 1e-8
                ):
        super().__init__()
        self.pred_len = pred_len
        self.point_wise_loss_fn = point_wise_loss_fn
        self.mse_weight = mse_weight
        self.k_dominant_freqs = k_dominant_freqs
        self.min_patch_len = max(2, min_patch_len)
        self.eps = eps

        if use_learnable_weights:
            self.raw_w_corr = nn.Parameter(t.tensor(float(w_corr)))
            self.raw_w_var = nn.Parameter(t.tensor(float(w_var)))
            self.raw_w_mean = nn.Parameter(t.tensor(float(w_mean)))
        else:
            self.register_buffer('fixed_w_corr', t.tensor(float(w_corr)))
            self.register_buffer('fixed_w_var', t.tensor(float(w_var)))
            self.register_buffer('fixed_w_mean', t.tensor(float(w_mean)))
        self.use_learnable_weights = use_learnable_weights

    def _fourier_adaptive_patching(self, series: t.Tensor) -> List[Tuple[int, int]]:
        """Fourier-based Adaptive Patching (FAP) for a single time series."""
        seq_len = series.shape[0]
        patch_indices = []

        if seq_len < self.min_patch_len * 2:
            if seq_len >= self.min_patch_len:
                patch_indices.append((0, seq_len))
            return patch_indices

        # FFT
        xf = t.fft.rfft(series)
        amplitudes = t.abs(xf)

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

        _, top_freq_indices_relative = t.topk(actual_freq_amplitudes, k=num_to_select)
        top_freq_indices_absolute = top_freq_indices_relative + 1

        dominant_periods_float = seq_len / top_freq_indices_absolute.float()
        processed_periods_for_series = set()

        for period_val_float in dominant_periods_float:
            period_len = int(t.floor(period_val_float))

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

    def _pearson_correlation(self, x: t.Tensor, y: t.Tensor):
        if x.numel() < 2 or y.numel() < 2:
            return t.tensor(0.0, device=x.device, dtype=x.dtype)
        
        vx = x - t.mean(x)
        vy = y - t.mean(y)
        
        denom = (t.sqrt(t.sum(vx ** 2) + self.eps) * t.sqrt(t.sum(vy ** 2) + self.eps))
        if denom < self.eps:
            return t.tensor(0.0, device=x.device, dtype=x.dtype)
            
        corr = t.sum(vx * vy) / denom
        return t.clamp(corr, -1.0, 1.0)

    def _calculate_structural_losses_for_patches(self, y_pred_patches: List[t.Tensor], y_true_patches: List[t.Tensor]):
        l_corr_vals, l_var_vals, l_mean_vals = [], [], []

        if not y_pred_patches:
            device = t.device('cpu')
            return (t.tensor(0.0, device=device, dtype=t.float32),
                   t.tensor(0.0, device=device, dtype=t.float32),
                   t.tensor(0.0, device=device, dtype=t.float32))

        for pred_p, true_p in zip(y_pred_patches, y_true_patches):
            if pred_p.ndim > 1: pred_p = pred_p.squeeze()
            if true_p.ndim > 1: true_p = true_p.squeeze()
            
            if pred_p.numel() < self.min_patch_len or true_p.numel() < self.min_patch_len:
                continue

            corr = self._pearson_correlation(pred_p, true_p)
            l_corr_vals.append((1.0 - corr)**2)

            var_pred = t.var(pred_p, unbiased=False)
            var_true = t.var(true_p, unbiased=False)
            l_var_vals.append((var_pred - var_true)**2)

            mean_pred = t.mean(pred_p)
            mean_true = t.mean(true_p)
            l_mean_vals.append((mean_pred - mean_true)**2)
        
        device = y_pred_patches[0].device if y_pred_patches else t.device('cpu')
        
        l_c = t.mean(t.stack(l_corr_vals)) if l_corr_vals else t.tensor(0.0, device=device, dtype=t.float32)
        l_v = t.mean(t.stack(l_var_vals)) if l_var_vals else t.tensor(0.0, device=device, dtype=t.float32)
        l_m = t.mean(t.stack(l_mean_vals)) if l_mean_vals else t.tensor(0.0, device=device, dtype=t.float32)
        
        return l_c, l_v, l_m

    def forward(self, y_pred_batch: t.Tensor, y_true_batch: t.Tensor) -> t.Tensor:
        loss_pointwise = self.point_wise_loss_fn(y_pred_batch, y_true_batch)

        all_pred_patches_for_batch: List[t.Tensor] = []
        all_true_patches_for_batch: List[t.Tensor] = []
        
        batch_size, current_pred_len, n_targets = y_true_batch.shape

        for i in range(batch_size):
            for j in range(n_targets):
                y_true_series = y_true_batch[i, :, j]
                y_pred_series = y_pred_batch[i, :, j]

                series_patch_indices = self._fourier_adaptive_patching(y_true_series)
                
                for start, end in series_patch_indices:
                    all_pred_patches_for_batch.append(y_pred_series[start:end])
                    all_true_patches_for_batch.append(y_true_series[start:end])
        
        if not all_pred_patches_for_batch:
            return loss_pointwise

        l_c, l_v, l_m = self._calculate_structural_losses_for_patches(all_pred_patches_for_batch, all_true_patches_for_batch)

        if self.use_learnable_weights:
            w_corr = t.sigmoid(self.raw_w_corr)
            w_var = t.sigmoid(self.raw_w_var)
            w_mean = t.sigmoid(self.raw_w_mean)
        else:
            w_corr = self.fixed_w_corr
            w_var = self.fixed_w_var
            w_mean = self.fixed_w_mean
            
        loss_structural = w_corr * l_c + w_var * l_v + w_mean * l_m
        total_loss = self.mse_weight * loss_pointwise + (1.0 - self.mse_weight) * loss_structural
        return total_loss


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mu: t.Tensor, sigma: t.Tensor, target: t.Tensor) -> t.Tensor:
        sigma = sigma.clamp(min=self.eps)
        variance = sigma**2
        nll = 0.5 * (t.log(2 * np.pi * variance) + ((target - mu)**2 / variance))
        return nll.mean()


class PinballLoss(nn.Module):
    """Pinball loss function for quantile regression."""
    def __init__(self, quantiles: List[float], reduction: str = 'mean'):
        super().__init__()
        if not all(0 < q_val < 1 for q_val in quantiles):
            raise ValueError("Quantiles must be between 0 and 1.")
        self.quantiles = sorted(quantiles)
        self.num_quantiles = len(quantiles)
        self.reduction = reduction

    def forward(self, y_pred_quantiles: t.Tensor, y_true: t.Tensor) -> t.Tensor:
        batch_size, pred_len, n_targets_times_quantiles = y_pred_quantiles.shape
        n_targets = y_true.shape[2]

        if n_targets_times_quantiles != n_targets * self.num_quantiles:
            raise ValueError(f"Prediction dimension mismatch. Expected {n_targets * self.num_quantiles}, got {n_targets_times_quantiles}.")

        y_pred_quantiles_reshaped = y_pred_quantiles.reshape(batch_size, pred_len, n_targets, self.num_quantiles)
        y_true_expanded = y_true.unsqueeze(-1).expand_as(y_pred_quantiles_reshaped)
        
        errors = y_true_expanded - y_pred_quantiles_reshaped
        quantile_tensor = t.tensor(self.quantiles, device=y_pred_quantiles.device, dtype=y_pred_quantiles.dtype)
        loss_per_quantile = t.max((quantile_tensor - 1) * errors, quantile_tensor * errors)
        
        return loss_per_quantile.mean() if self.reduction == 'mean' else loss_per_quantile.sum()


class HuberLoss(nn.Module):
    """Huber loss function - robust to outliers."""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        abs_error = t.abs(pred - target)
        quadratic = t.minimum(abs_error, t.tensor(self.delta))
        linear = abs_error - quadratic
        return t.mean(0.5 * quadratic**2 + self.delta * linear)


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        mse = F.mse_loss(pred, target, reduction='none')
        p_t = t.exp(-mse)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return t.mean(focal_weight * mse)


class DTWLoss(nn.Module):
    """Dynamic Time Warping based loss function."""
    def __init__(self, gamma: float = 1.0, normalize: bool = True):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def _soft_dtw(self, x: t.Tensor, y: t.Tensor) -> t.Tensor:
        """Simplified soft-DTW implementation."""
        batch_size, seq_len_x, features = x.shape
        _, seq_len_y, _ = y.shape
        
        # Distance matrix
        dist_matrix = t.cdist(x, y, p=2)  # Euclidean distance
        
        # Soft-DTW computation (simplified)
        gamma = self.gamma
        dtw_matrix = t.zeros((seq_len_x + 1, seq_len_y + 1), device=x.device)
        dtw_matrix[0, 1:] = float('inf')
        dtw_matrix[1:, 0] = float('inf')
        
        for i in range(1, seq_len_x + 1):
            for j in range(1, seq_len_y + 1):
                cost = dist_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + gamma * t.logsumexp(
                    t.stack([
                        dtw_matrix[i-1, j] / gamma,
                        dtw_matrix[i, j-1] / gamma,
                        dtw_matrix[i-1, j-1] / gamma
                    ]), dim=0
                )
        
        return dtw_matrix[seq_len_x, seq_len_y]

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        batch_losses = []
        for i in range(pred.shape[0]):
            dtw_loss = self._soft_dtw(pred[i].unsqueeze(0), target[i].unsqueeze(0))
            if self.normalize:
                dtw_loss = dtw_loss / (pred.shape[1] + target.shape[1])
            batch_losses.append(dtw_loss)
        return t.mean(t.stack(batch_losses))


class SeasonalLoss(nn.Module):
    """Loss function that emphasizes seasonal patterns."""
    def __init__(self, season_length: int, seasonal_weight: float = 1.0):
        super().__init__()
        self.season_length = season_length
        self.seasonal_weight = seasonal_weight

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        # Standard MSE loss
        mse_loss = F.mse_loss(pred, target)
        
        # Seasonal component loss
        if pred.shape[1] >= self.season_length:
            # Compare seasonal patterns
            pred_seasonal = pred[:, :-self.season_length] - pred[:, self.season_length:]
            target_seasonal = target[:, :-self.season_length] - target[:, self.season_length:]
            seasonal_loss = F.mse_loss(pred_seasonal, target_seasonal)
            
            return mse_loss + self.seasonal_weight * seasonal_loss
        else:
            return mse_loss


class TrendAwareLoss(nn.Module):
    """Loss function with separate penalties for trend and noise."""
    def __init__(self, trend_weight: float = 1.0, noise_weight: float = 0.5):
        super().__init__()
        self.trend_weight = trend_weight
        self.noise_weight = noise_weight

    def _extract_trend(self, x: t.Tensor, window_size: int = 3) -> t.Tensor:
        """Extract trend using moving average."""
        if x.shape[1] < window_size:
            return x
        
        # Simple moving average for trend
        padding = window_size // 2
        x_padded = F.pad(x, (0, 0, padding, padding), mode='replicate')
        trend = F.avg_pool1d(x_padded.transpose(1, 2), kernel_size=window_size, stride=1).transpose(1, 2)
        return trend

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        # Extract trends
        pred_trend = self._extract_trend(pred)
        target_trend = self._extract_trend(target)
        
        # Extract noise (residuals)
        pred_noise = pred - pred_trend
        target_noise = target - target_trend
        
        # Separate losses
        trend_loss = F.mse_loss(pred_trend, target_trend)
        noise_loss = F.mse_loss(pred_noise, target_noise)
        
        return self.trend_weight * trend_loss + self.noise_weight * noise_loss


class QuantileLoss(nn.Module):
    """Simple quantile loss for a single quantile."""
    def __init__(self, quantile: float):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        error = target - pred
        loss = t.max((self.quantile - 1) * error, self.quantile * error)
        return t.mean(loss)


# Helper function to get loss by name
def get_loss_function(loss_name: str, **kwargs):
    """Get loss function by name with optional parameters."""
    loss_functions = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': HuberLoss,
        'focal': FocalLoss,
        'mape': mape_loss,
        'smape': smape_loss,
        'mase': mase_loss,
        'ps_loss': PSLoss,
        'gaussian_nll': GaussianNLLLoss,
        'pinball': PinballLoss,
        'dtw': DTWLoss,
        'seasonal': SeasonalLoss,
        'trend_aware': TrendAwareLoss,
        'quantile': QuantileLoss
    }
    
    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name.lower()](**kwargs)
