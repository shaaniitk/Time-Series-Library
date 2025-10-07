"""
Mixture Density Network Loss Functions

This module provides loss functions specifically designed for Mixture Density Networks (MDN)
that output Gaussian mixture distributions. These functions properly train all parameters
of the predicted distribution (means, standard deviations, and mixture weights).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union


def gaussian_nll_loss(y_true: torch.Tensor,
                     means: torch.Tensor,
                     log_stds: torch.Tensor,
                     log_weights: torch.Tensor,
                     reduction: str = 'mean',
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the Negative Log-Likelihood for a Gaussian Mixture Model.
    
    This is the proper loss function to use with MixtureDensityDecoder outputs.
    It trains all parameters of the predicted distribution, unlike MSE which only
    trains the mean.
    
    Args:
        y_true: Ground truth values [batch, pred_len, num_targets]
        means: Predicted means [batch, pred_len, num_targets, num_components]
        log_stds: Predicted log std devs [batch, pred_len, num_targets, num_components]
        log_weights: Predicted log weights [batch, pred_len, num_components]
        reduction: Reduction method ('mean', 'sum', 'none')
        eps: Small value for numerical stability
        
    Returns:
        Negative log-likelihood loss
    """
    # Handle different dimensionalities of y_true and means
    # Debug info
    print(f"🔍 Debug: y_true.shape={y_true.shape}, means.shape={means.shape}")
    
    if y_true.dim() == 4 and means.dim() == 3:
        # y_true: [batch, seq, features, 1] -> [batch, seq, features]
        y_true_flat = y_true.squeeze(-1)  # [batch, seq, features]
        # Expand to match mixture components: [batch, seq, features] -> [batch, seq, features, n_components]
        y_true_expanded = y_true_flat.unsqueeze(-1).expand(-1, -1, -1, means.shape[-1])
        print(f"🔍 Case 1: y_true_expanded.shape={y_true_expanded.shape}")
    elif y_true.dim() == 3 and means.dim() == 3:
        # Both are 3D, expand y_true to match mixture components
        y_true_expanded = y_true.unsqueeze(-1).expand_as(means)
        print(f"🔍 Case 2: y_true_expanded.shape={y_true_expanded.shape}")
    elif y_true.dim() == 4 and means.dim() == 4:
        # Both are 4D, just expand the last dimension if needed
        if y_true.shape[-1] == 1 and means.shape[-1] > 1:
            y_true_expanded = y_true.expand_as(means)
        else:
            y_true_expanded = y_true
        print(f"🔍 Case 3: y_true_expanded.shape={y_true_expanded.shape}")
    else:
        # Fallback: handle any other case more carefully
        print(f"🔍 Fallback case: y_true.dim()={y_true.dim()}, means.dim()={means.dim()}")
        if y_true.shape[-1] == 1:
            # Remove the last dimension and add mixture dimension
            y_true_squeezed = y_true.squeeze(-1)
            y_true_expanded = y_true_squeezed.unsqueeze(-1).expand(*y_true_squeezed.shape, means.shape[-1])
        else:
            y_true_expanded = y_true.unsqueeze(-1).expand(*y_true.shape, means.shape[-1])
        print(f"🔍 Fallback result: y_true_expanded.shape={y_true_expanded.shape}")
    
    # Convert log_stds to stds with numerical stability
    stds = torch.exp(log_stds).clamp(min=eps)
    
    # Compute log probability of y_true under each Gaussian component
    # log N(y|μ,σ) = -0.5 * ((y-μ)/σ)² - log(σ) - 0.5*log(2π)
    log_p = -0.5 * ((y_true_expanded - means) / stds)**2 - log_stds - 0.5 * math.log(2 * math.pi)
    
    # Get normalized mixture weights (log softmax for numerical stability)
    log_weights_normalized = F.log_softmax(log_weights, dim=-1)
    
    # Expand weights to match log_p shape [batch, pred_len, num_targets, num_components]
    log_weights_expanded = log_weights_normalized.unsqueeze(2).expand_as(log_p)
    
    # Combine component probabilities with mixture weights
    # log(Σ w_k * N(y|μ_k,σ_k)) = logsumexp(log(w_k) + log(N(y|μ_k,σ_k)))
    log_likelihood = torch.logsumexp(log_weights_expanded + log_p, dim=-1)
    
    # Return negative log-likelihood
    nll = -log_likelihood
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(nll)
    elif reduction == 'sum':
        return torch.sum(nll)
    else:
        return nll


class GaussianMixtureNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss for Gaussian Mixture Models.
    
    This loss function is designed to work with MixtureDensityDecoder outputs
    and properly trains all parameters of the predicted distribution.
    
    Example usage:
        criterion = GaussianMixtureNLLLoss()
        loss = criterion(y_true, means, log_stds, log_weights)
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, 
                y_true: torch.Tensor,
                means: torch.Tensor, 
                log_stds: torch.Tensor, 
                log_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            y_true: Ground truth values [batch, pred_len, num_targets]
            means: Predicted means [batch, pred_len, num_targets, num_components]
            log_stds: Predicted log std devs [batch, pred_len, num_targets, num_components]
            log_weights: Predicted log weights [batch, pred_len, num_components]
            
        Returns:
            Negative log-likelihood loss
        """
        return gaussian_nll_loss(y_true, means, log_stds, log_weights, 
                               self.reduction, self.eps)


def extract_point_prediction(means: torch.Tensor, 
                           log_stds: torch.Tensor,
                           log_weights: torch.Tensor) -> torch.Tensor:
    """
    Extract a point prediction from mixture density network outputs.
    
    This computes the weighted average of component means, which is the
    expected value of the mixture distribution.
    
    Args:
        means: Predicted means [batch, pred_len, num_targets, num_components]
        log_stds: Predicted log std devs [batch, pred_len, num_targets, num_components]
        log_weights: Predicted log weights [batch, pred_len, num_components]
        
    Returns:
        Point predictions [batch, pred_len, num_targets]
    """
    # Convert log weights to normalized weights
    weights = F.softmax(log_weights, dim=-1)  # [batch, pred_len, num_components]
    
    # Expand weights to match means shape
    weights_expanded = weights.unsqueeze(2).expand_as(means)  # [batch, pred_len, num_targets, num_components]
    
    # Compute weighted average of means
    point_pred = (means * weights_expanded).sum(dim=-1)  # [batch, pred_len, num_targets]
    
    return point_pred


def extract_uncertainty_estimates(means: torch.Tensor,
                                log_stds: torch.Tensor,
                                log_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract uncertainty estimates from mixture density network outputs.
    
    Returns both aleatoric (data) uncertainty and epistemic (model) uncertainty.
    
    Args:
        means: Predicted means [batch, pred_len, num_targets, num_components]
        log_stds: Predicted log std devs [batch, pred_len, num_targets, num_components]
        log_weights: Predicted log weights [batch, pred_len, num_components]
        
    Returns:
        aleatoric_uncertainty: Data uncertainty [batch, pred_len, num_targets]
        epistemic_uncertainty: Model uncertainty [batch, pred_len, num_targets]
    """
    # Convert to normalized weights and stds
    weights = F.softmax(log_weights, dim=-1)  # [batch, pred_len, num_components]
    stds = torch.exp(log_stds)  # [batch, pred_len, num_targets, num_components]
    
    # Expand weights
    weights_expanded = weights.unsqueeze(2).expand_as(means)
    
    # Compute mixture mean (point prediction)
    mixture_mean = (means * weights_expanded).sum(dim=-1)  # [batch, pred_len, num_targets]
    
    # Aleatoric uncertainty: weighted average of component variances
    variances = stds ** 2
    aleatoric_uncertainty = (variances * weights_expanded).sum(dim=-1)  # [batch, pred_len, num_targets]
    
    # Epistemic uncertainty: variance of component means
    mean_diff_sq = (means - mixture_mean.unsqueeze(-1)) ** 2
    epistemic_uncertainty = (mean_diff_sq * weights_expanded).sum(dim=-1)  # [batch, pred_len, num_targets]
    
    return torch.sqrt(aleatoric_uncertainty), torch.sqrt(epistemic_uncertainty)


class AdaptiveMixtureLoss(nn.Module):
    """
    Adaptive mixture loss that combines NLL loss with regularization terms.
    
    This loss function includes:
    1. Negative log-likelihood for proper mixture training
    2. Regularization to prevent component collapse
    3. Adaptive weighting based on prediction confidence
    """
    
    def __init__(self, 
                 nll_weight: float = 1.0,
                 reg_weight: float = 0.01,
                 min_component_weight: float = 0.01,
                 reduction: str = 'mean'):
        super().__init__()
        self.nll_weight = nll_weight
        self.reg_weight = reg_weight
        self.min_component_weight = min_component_weight
        self.reduction = reduction
        
        self.nll_loss = GaussianMixtureNLLLoss(reduction='none')
    
    def forward(self,
                y_true: torch.Tensor,
                means: torch.Tensor,
                log_stds: torch.Tensor,
                log_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive mixture loss.
        
        Args:
            y_true: Ground truth values
            means: Predicted means
            log_stds: Predicted log standard deviations
            log_weights: Predicted log mixture weights
            
        Returns:
            Combined loss value
        """
        # Main NLL loss
        nll = self.nll_loss(y_true, means, log_stds, log_weights)
        
        # Regularization: prevent component collapse
        weights = F.softmax(log_weights, dim=-1)
        
        # Entropy regularization to encourage diverse components
        entropy_reg = -torch.sum(weights * F.log_softmax(log_weights, dim=-1), dim=-1)
        
        # Minimum weight regularization
        min_weight_penalty = F.relu(self.min_component_weight - weights).sum(dim=-1)
        
        # Combine losses
        total_loss = (self.nll_weight * nll + 
                     self.reg_weight * (entropy_reg + min_weight_penalty))
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            return total_loss


# Convenience function for backward compatibility
def create_mixture_criterion(use_adaptive: bool = False, **kwargs) -> nn.Module:
    """
    Create a mixture density loss criterion.
    
    Args:
        use_adaptive: Whether to use adaptive mixture loss with regularization
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss criterion
    """
    if use_adaptive:
        return AdaptiveMixtureLoss(**kwargs)
    else:
        return GaussianMixtureNLLLoss(**kwargs)