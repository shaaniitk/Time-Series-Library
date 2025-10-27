"""
Directional and Trend-Focused Loss Functions for Financial Time Series Forecasting.

This module provides loss functions that prioritize directional accuracy (sign prediction)
and trend alignment over exact value prediction, specifically designed for trading applications
where direction matters more than magnitude.

Classes:
    DirectionalTrendLoss: Combines directional accuracy, trend correlation, and magnitude terms
    HybridMDNDirectionalLoss: Wrapper combining MixtureNLLLoss with DirectionalTrendLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union


class DirectionalTrendLoss(nn.Module):
    """
    Loss function that prioritizes directional accuracy and trend alignment.
    
    Designed for financial forecasting where predicting the correct direction (increase/decrease)
    is more important than exact magnitude. Combines three components:
    
    1. Directional Accuracy: Penalizes sign mismatches heavily
    2. Trend Correlation: Rewards predictions that follow trend momentum
    3. Magnitude Accuracy: Light penalty to prevent extreme predictions
    
    Args:
        direction_weight: Weight for directional accuracy term (default: 5.0)
        trend_weight: Weight for trend correlation term (default: 2.0)
        magnitude_weight: Weight for magnitude MSE term (default: 0.1)
        use_mdn_mean: If True, extract mean from MDN parameters (default: True)
        correlation_type: 'pearson' or 'spearman' for trend correlation (default: 'pearson')
        per_target_weights: Optional weights for each target feature [num_targets]
        eps: Small constant for numerical stability (default: 1e-8)
        smooth_tanh_scale: Scale factor for tanh smoothing (default: 1.0)
    
    Example:
        >>> loss_fn = DirectionalTrendLoss(direction_weight=5.0, trend_weight=2.0, magnitude_weight=0.1)
        >>> pred = torch.randn(32, 96, 4)  # [batch, seq_len, num_targets]
        >>> target = torch.randn(32, 96, 4)
        >>> loss = loss_fn(pred, target)
    
    Notes:
        - For MDN outputs, pass (means, log_stds, log_weights) as pred
        - Directional loss uses smooth tanh approximation for differentiability
        - Trend correlation is computed on temporal differences
    """
    
    def __init__(
        self,
        direction_weight: float = 5.0,
        trend_weight: float = 2.0,
        magnitude_weight: float = 0.1,
        use_mdn_mean: bool = True,
        correlation_type: str = 'pearson',
        per_target_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
        smooth_tanh_scale: float = 1.0
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.trend_weight = trend_weight
        self.magnitude_weight = magnitude_weight
        self.use_mdn_mean = use_mdn_mean
        self.correlation_type = correlation_type.lower()
        self.eps = eps
        self.smooth_tanh_scale = smooth_tanh_scale
        
        if per_target_weights is not None:
            self.register_buffer('per_target_weights', per_target_weights)
        else:
            self.per_target_weights = None
        
        if self.correlation_type not in ['pearson', 'spearman']:
            raise ValueError(f"correlation_type must be 'pearson' or 'spearman', got {correlation_type}")
    
    def forward(
        self,
        pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the directional trend loss.
        
        Args:
            pred: Predictions - either:
                - Tensor [batch, seq_len] or [batch, seq_len, num_targets]
                - Tuple (means, log_stds, log_weights) for MDN outputs
            target: Ground truth [batch, seq_len] or [batch, seq_len, num_targets]
        
        Returns:
            Scalar loss tensor
        """
        # Extract point predictions from MDN if needed
        if isinstance(pred, tuple):
            if self.use_mdn_mean:
                pred_values = self._extract_mdn_mean(pred)
            else:
                raise ValueError("MDN outputs require use_mdn_mean=True")
        else:
            pred_values = pred
        
        # Ensure both have same number of dimensions
        if pred_values.dim() != target.dim():
            # Try to match dimensions by squeezing/unsqueezing last dim
            if pred_values.dim() == 2 and target.dim() == 3 and target.size(-1) == 1:
                target = target.squeeze(-1)
            elif pred_values.dim() == 3 and target.dim() == 2:
                target = target.unsqueeze(-1)
        
        # Ensure target has same shape as predictions
        if target.shape != pred_values.shape:
            raise ValueError(f"Shape mismatch: pred {pred_values.shape} vs target {target.shape}")
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=pred_values.device, dtype=pred_values.dtype)
        
        # Compute directional accuracy loss
        if self.direction_weight > 0:
            dir_loss = self._compute_directional_accuracy(pred_values, target)
            total_loss = total_loss + self.direction_weight * dir_loss
        
        # Compute trend correlation loss
        if self.trend_weight > 0:
            trend_loss = self._compute_trend_correlation(pred_values, target)
            total_loss = total_loss + self.trend_weight * trend_loss
        
        # Compute magnitude loss
        if self.magnitude_weight > 0:
            mag_loss = self._compute_magnitude_loss(pred_values, target)
            total_loss = total_loss + self.magnitude_weight * mag_loss
        
        return total_loss
    
    def _extract_mdn_mean(
        self,
        mdn_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract mixture mean from MDN parameters.
        
        Args:
            mdn_params: Tuple (means, log_stds, log_weights)
        
        Returns:
            Mixture mean [batch, seq_len] or [batch, seq_len, num_targets]
        """
        means, log_stds, log_weights = mdn_params
        
        # Compute mixture weights
        weights = F.softmax(log_weights, dim=-1)  # [batch, seq_len, num_components]
        
        # Handle multivariate case
        if means.dim() == 4:  # [batch, seq_len, num_targets, num_components]
            # Expand weights to match means dimensions
            weights_expanded = weights.unsqueeze(2)  # [batch, seq_len, 1, num_components]
            mixture_mean = torch.sum(weights_expanded * means, dim=-1)  # [batch, seq_len, num_targets]
        else:  # [batch, seq_len, num_components]
            mixture_mean = torch.sum(weights * means, dim=-1)  # [batch, seq_len]
        
        return mixture_mean
    
    def _compute_directional_accuracy(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute directional accuracy loss using smooth sign matching.
        
        Uses tanh approximation of sign function for smooth gradients:
        - Loss = -tanh(pred_diff) * tanh(target_diff)
        - Same sign: loss → -1 (negative, will be minimized)
        - Opposite sign: loss → +1 (positive, penalty)
        
        Args:
            pred: Predictions [batch, seq_len] or [batch, seq_len, num_targets]
            target: Ground truth [batch, seq_len] or [batch, seq_len, num_targets]
        
        Returns:
            Scalar directional loss
        """
        # Compute temporal differences (returns)
        pred_diff = pred[:, 1:] - pred[:, :-1]  # [batch, seq_len-1, ...]
        target_diff = target[:, 1:] - target[:, :-1]  # [batch, seq_len-1, ...]
        
        # Smooth sign matching using tanh
        pred_sign = torch.tanh(pred_diff * self.smooth_tanh_scale)
        target_sign = torch.tanh(target_diff * self.smooth_tanh_scale)
        
        # Sign matching: negative when same sign, positive when opposite
        sign_match = pred_sign * target_sign
        
        # Convert to loss: minimize negative sign matches, penalize positive mismatches
        # We want to MAXIMIZE sign_match (same sign), so MINIMIZE -sign_match
        directional_loss = -sign_match
        
        # Apply per-target weights if provided
        if self.per_target_weights is not None and directional_loss.dim() == 3:
            # Expand weights to [1, 1, num_targets]
            weights = self.per_target_weights.view(1, 1, -1)
            directional_loss = directional_loss * weights
        
        return directional_loss.mean()
    
    def _compute_trend_correlation(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trend correlation loss using Pearson or Spearman correlation.
        
        Measures how well predicted differences correlate with actual differences
        across the sequence. Higher correlation = better trend alignment.
        
        Args:
            pred: Predictions [batch, seq_len] or [batch, seq_len, num_targets]
            target: Ground truth [batch, seq_len] or [batch, seq_len, num_targets]
        
        Returns:
            Scalar trend correlation loss (1 - correlation)
        """
        # Compute temporal differences
        pred_diff = pred[:, 1:] - pred[:, :-1]  # [batch, seq_len-1, ...]
        target_diff = target[:, 1:] - target[:, :-1]  # [batch, seq_len-1, ...]
        
        # Flatten batch and target dimensions for correlation
        if pred_diff.dim() == 3:
            # [batch, seq_len-1, num_targets] -> [batch * num_targets, seq_len-1]
            batch_size, seq_len, num_targets = pred_diff.shape
            pred_diff_flat = pred_diff.transpose(1, 2).reshape(-1, seq_len)
            target_diff_flat = target_diff.transpose(1, 2).reshape(-1, seq_len)
        else:
            # [batch, seq_len-1] -> already in correct shape
            pred_diff_flat = pred_diff
            target_diff_flat = target_diff
        
        # Compute correlation based on type
        if self.correlation_type == 'pearson':
            correlation = self._pearson_correlation(pred_diff_flat, target_diff_flat)
        else:  # spearman
            correlation = self._spearman_correlation(pred_diff_flat, target_diff_flat)
        
        # Convert to loss: maximize correlation, minimize (1 - correlation)
        trend_loss = 1.0 - correlation.mean()
        
        return trend_loss
    
    def _pearson_correlation(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            x: Tensor [batch, seq_len]
            y: Tensor [batch, seq_len]
        
        Returns:
            Correlation coefficients [batch]
        """
        # Center the data
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)
        
        # Compute correlation
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = torch.sqrt(
            (x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1) + self.eps
        )
        
        correlation = numerator / (denominator + self.eps)
        
        # Clamp to [-1, 1] for numerical stability
        return torch.clamp(correlation, -1.0, 1.0)
    
    def _spearman_correlation(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Spearman rank correlation coefficient.
        
        Args:
            x: Tensor [batch, seq_len]
            y: Tensor [batch, seq_len]
        
        Returns:
            Correlation coefficients [batch]
        """
        # Convert to ranks
        x_ranks = self._compute_ranks(x)
        y_ranks = self._compute_ranks(y)
        
        # Compute Pearson correlation on ranks
        return self._pearson_correlation(x_ranks, y_ranks)
    
    def _compute_ranks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ranks for Spearman correlation.
        
        Args:
            x: Tensor [batch, seq_len]
        
        Returns:
            Ranks [batch, seq_len]
        """
        # Sort and get indices
        _, indices = torch.sort(x, dim=1)
        
        # Create ranks
        ranks = torch.zeros_like(x)
        batch_size, seq_len = x.shape
        
        # Assign ranks
        for i in range(batch_size):
            ranks[i, indices[i]] = torch.arange(seq_len, dtype=x.dtype, device=x.device)
        
        return ranks
    
    def _compute_magnitude_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute magnitude loss using MSE.
        
        Args:
            pred: Predictions [batch, seq_len] or [batch, seq_len, num_targets]
            target: Ground truth [batch, seq_len] or [batch, seq_len, num_targets]
        
        Returns:
            Scalar MSE loss
        """
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Apply per-target weights if provided
        if self.per_target_weights is not None and mse.dim() == 3:
            weights = self.per_target_weights.view(1, 1, -1)
            mse = mse * weights
        
        return mse.mean()


class HybridMDNDirectionalLoss(nn.Module):
    """
    Hybrid loss combining MixtureNLLLoss with DirectionalTrendLoss.
    
    Provides both uncertainty quantification (via MDN) and directional focus.
    Recommended for financial forecasting where you need:
    - Probabilistic predictions for risk management
    - Strong directional signals for trading decisions
    
    Args:
        nll_weight: Weight for MixtureNLLLoss component (default: 0.3)
        direction_weight: Weight for directional accuracy (default: 3.0)
        trend_weight: Weight for trend correlation (default: 1.5)
        magnitude_weight: Weight for magnitude term (default: 0.1)
        mdn_kwargs: Additional kwargs for MixtureNLLLoss
        directional_kwargs: Additional kwargs for DirectionalTrendLoss
    
    Example:
        >>> from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss
        >>> loss_fn = HybridMDNDirectionalLoss(
        ...     nll_weight=0.3,
        ...     direction_weight=3.0,
        ...     trend_weight=1.5,
        ...     magnitude_weight=0.1
        ... )
        >>> mdn_params = (means, log_stds, log_weights)
        >>> target = torch.randn(32, 96, 4)
        >>> loss = loss_fn(mdn_params, target)
    """
    
    def __init__(
        self,
        nll_weight: float = 0.3,
        direction_weight: float = 3.0,
        trend_weight: float = 1.5,
        magnitude_weight: float = 0.1,
        mdn_kwargs: Optional[dict] = None,
        directional_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.nll_weight = nll_weight
        
        # Import MixtureNLLLoss
        try:
            from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss
        except ImportError:
            raise ImportError(
                "MixtureNLLLoss not found. Ensure layers.modular.decoder.mixture_density_decoder is available."
            )
        
        # Initialize component losses
        mdn_kwargs = mdn_kwargs or {}
        directional_kwargs = directional_kwargs or {}
        
        self.mdn_loss = MixtureNLLLoss(**mdn_kwargs)
        self.directional_loss = DirectionalTrendLoss(
            direction_weight=direction_weight,
            trend_weight=trend_weight,
            magnitude_weight=magnitude_weight,
            use_mdn_mean=True,
            **directional_kwargs
        )
    
    def forward(
        self,
        mdn_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hybrid loss.
        
        Args:
            mdn_params: Tuple (means, log_stds, log_weights)
            target: Ground truth [batch, seq_len] or [batch, seq_len, num_targets]
        
        Returns:
            Scalar hybrid loss
        """
        # Compute MDN NLL loss
        nll_loss = self.mdn_loss(mdn_params, target)
        
        # Compute directional trend loss
        dir_trend_loss = self.directional_loss(mdn_params, target)
        
        # Combine with weights
        total_loss = self.nll_weight * nll_loss + dir_trend_loss
        
        return total_loss


# Utility function for computing directional accuracy metric
def compute_directional_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0
) -> float:
    """
    Compute directional accuracy as percentage of correct sign predictions.
    
    Args:
        pred: Predictions [batch, seq_len] or [batch, seq_len, num_targets]
        target: Ground truth [batch, seq_len] or [batch, seq_len, num_targets]
        threshold: Threshold for considering a change as zero (default: 0.0)
    
    Returns:
        Directional accuracy as percentage (0-100)
    """
    # Compute differences
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    
    # Compute signs (excluding near-zero changes)
    pred_sign = torch.sign(pred_diff)
    target_sign = torch.sign(target_diff)
    
    # Apply threshold if needed
    if threshold > 0:
        near_zero = torch.abs(target_diff) < threshold
        pred_sign = pred_sign.masked_fill(near_zero, 0)
        target_sign = target_sign.masked_fill(near_zero, 0)
    
    # Compute accuracy
    correct = (pred_sign == target_sign).float()
    accuracy = correct.mean().item() * 100.0
    
    return accuracy
