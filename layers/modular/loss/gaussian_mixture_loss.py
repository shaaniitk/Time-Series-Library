"""
Gaussian Mixture Model Loss Functions

This module implements proper loss functions for Mixture Density Networks (MDN)
that output Gaussian mixture distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .registry import LossRegistry


class GaussianMixtureNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss for Gaussian Mixture Models.
    
    This loss function is designed to work with MixtureDensityDecoder outputs
    and properly trains all parameters of the predicted distribution (means,
    standard deviations, and mixture weights).
    
    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
            'none': no reduction will be applied
        eps (float): Small value to prevent numerical instability
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
        Compute the Negative Log-Likelihood for a Gaussian Mixture Model.
        
        Args:
            y_true: Ground truth values [batch, pred_len, num_targets]
            means: Predicted means [batch, pred_len, num_targets, num_components]
            log_stds: Predicted log std devs [batch, pred_len, num_targets, num_components]
            log_weights: Predicted log weights [batch, pred_len, num_components]
            
        Returns:
            Negative log-likelihood loss
        """
        # Handle different dimensionalities of y_true and means
        if y_true.dim() == 4 and means.dim() == 3:
            # y_true: [batch, seq, features, 1] -> [batch, seq, features]
            y_true_flat = y_true.squeeze(-1)  # [batch, seq, features]
            # Expand to match mixture components: [batch, seq, features] -> [batch, seq, features, n_components]
            y_true_expanded = y_true_flat.unsqueeze(-1).expand(-1, -1, -1, means.shape[-1])
        elif y_true.dim() == 3 and means.dim() == 3:
            # Both are 3D, expand y_true to match mixture components
            y_true_expanded = y_true.unsqueeze(-1).expand_as(means)
        elif y_true.dim() == 4 and means.dim() == 4:
            # Both are 4D, just expand the last dimension if needed
            if y_true.shape[-1] == 1 and means.shape[-1] > 1:
                y_true_expanded = y_true.expand_as(means)
            else:
                y_true_expanded = y_true
        else:
            # Fallback: try to expand as before
            y_true_expanded = y_true.unsqueeze(-1).expand_as(means)
        
        # Convert log_stds to stds with numerical stability
        stds = torch.exp(log_stds).clamp(min=self.eps)
        
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
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:
            return nll


class GaussianMixtureKLLoss(nn.Module):
    """
    KL Divergence loss between predicted and target Gaussian mixtures.
    Useful when you have target distributions rather than point targets.
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self,
                pred_means: torch.Tensor,
                pred_log_stds: torch.Tensor,
                pred_log_weights: torch.Tensor,
                target_means: torch.Tensor,
                target_log_stds: torch.Tensor,
                target_log_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between two Gaussian mixtures.
        This is an approximation using Monte Carlo sampling.
        """
        # This is a simplified implementation - full KL divergence between
        # Gaussian mixtures doesn't have a closed form
        # For now, we'll use a component-wise approximation
        
        # Convert to normalized weights
        pred_weights = F.softmax(pred_log_weights, dim=-1)
        target_weights = F.softmax(target_log_weights, dim=-1)
        
        # Component-wise KL divergence approximation
        pred_stds = torch.exp(pred_log_stds).clamp(min=self.eps)
        target_stds = torch.exp(target_log_stds).clamp(min=self.eps)
        
        # KL between corresponding Gaussian components
        kl_components = (
            torch.log(target_stds / pred_stds) +
            (pred_stds**2 + (pred_means - target_means)**2) / (2 * target_stds**2) -
            0.5
        )
        
        # Weight by mixture coefficients
        kl_weighted = pred_weights.unsqueeze(2) * kl_components
        
        # KL divergence between mixture weights
        kl_weights = F.kl_div(pred_log_weights, target_weights, reduction='none')
        
        total_kl = kl_weighted.sum(dim=-1) + kl_weights.unsqueeze(2)
        
        if self.reduction == 'mean':
            return torch.mean(total_kl)
        elif self.reduction == 'sum':
            return torch.sum(total_kl)
        else:
            return total_kl


def gaussian_nll_loss(y_true: torch.Tensor,
                     means: torch.Tensor,
                     log_stds: torch.Tensor,
                     log_weights: torch.Tensor,
                     reduction: str = 'mean',
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Functional interface for Gaussian Mixture NLL loss.
    
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
    loss_fn = GaussianMixtureNLLLoss(reduction=reduction, eps=eps)
    return loss_fn(y_true, means, log_stds, log_weights)


# Register the loss functions
LossRegistry.register("gaussian_mixture_nll", GaussianMixtureNLLLoss)
LossRegistry.register("gaussian_mixture_kl", GaussianMixtureKLLoss)