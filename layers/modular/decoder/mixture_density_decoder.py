import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MixtureDensityDecoder(nn.Module):
    """Mixture Density Network decoder that predicts a mixture of K Gaussian distributions.
    
    This decoder can capture multi-modal uncertainty by predicting multiple Gaussian
    components with their respective means, standard deviations, and mixture weights.
    """
    
    def __init__(self, d_model, pred_len, num_components=3, num_targets=1):
        """
        Args:
            d_model: Input feature dimension
            pred_len: Prediction horizon (number of time steps to predict)
            num_components: Number of Gaussian components in the mixture (K)
            num_targets: Number of target features to predict
        """
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.num_components = num_components
        self.num_targets = num_targets
        
        # Shared feature processing from aggregated temporal context
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Learnable attention pooling over temporal dimension to avoid mean pooling
        self.pool_attention = nn.Linear(d_model, 1)
        
        # Separate heads for each mixture component parameter
        # Means/stds have per-target parameters; weights are per time step, shared across targets
        out_dim_params = pred_len * num_targets * num_components
        out_dim_weights = pred_len * num_components
        self.mean_head = nn.Linear(d_model, out_dim_params)
        self.std_head = nn.Linear(d_model, out_dim_params)      # outputs log-stds (unconstrained)
        self.weight_head = nn.Linear(d_model, out_dim_weights)  # outputs log-weights (unnormalized)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            tuple: (means, log_stds, log_weights)
                - means: [batch_size, pred_len, num_targets, num_components] if num_targets > 1
                         [batch_size, pred_len, num_components] if num_targets == 1
                - log_stds: Same shape as means
                - log_weights: [batch_size, pred_len, num_components] (shared across targets)
        """
        # Aggregate temporal dimension via learnable attention pooling (avoids information loss)
        # x: [B, seq_len, d_model]
        attn_logits = self.pool_attention(x)                 # [B, seq_len, 1]
        attn_weights = torch.softmax(attn_logits, dim=1)     # [B, seq_len, 1]
        context = torch.sum(attn_weights * x, dim=1)         # [B, d_model]
        hidden = self.mlp(context)  # [B, d_model]
        
        B = x.size(0)
        
        # Generate mixture parameters
        means_flat = self.mean_head(hidden)    # [B, pred_len * num_targets * num_components]
        log_stds_flat = self.std_head(hidden)  # [B, pred_len * num_targets * num_components]
        log_weights_flat = self.weight_head(hidden)  # [B, pred_len * num_components]
        
        if self.num_targets == 1:
            # Univariate case - maintain backward compatibility
            means = means_flat.view(B, self.pred_len, self.num_components)
            log_stds = log_stds_flat.view(B, self.pred_len, self.num_components)
            log_weights = log_weights_flat.view(B, self.pred_len, self.num_components)
        else:
            # Multivariate case
            means = means_flat.view(B, self.pred_len, self.num_targets, self.num_components)
            log_stds = log_stds_flat.view(B, self.pred_len, self.num_targets, self.num_components)
            # Weights are shared across targets per time step
            log_weights = log_weights_flat.view(B, self.pred_len, self.num_components)
        
        return means, log_stds, log_weights
    
    def sample(self, mixture_params, num_samples=1):
        """
        Sample from the mixture distribution.
        
        Args:
            mixture_params: Tuple (means, log_stds, log_weights)
            num_samples: Number of samples to draw
            
        Returns:
            samples: Tensor of shape (num_samples, batch_size, pred_len)
        """
        means, log_stds, log_weights = mixture_params
        B, T, K = means.shape
        
        # Convert to usable parameters
        stds = torch.exp(log_stds).clamp_min(1e-6)
        weights = F.softmax(log_weights, dim=-1)
        
        # Sample mixture indices for each (B, T) position
        weights_flat = weights.view(B * T, K)
        comp_idx = torch.multinomial(weights_flat, num_samples, replacement=True)  # [B*T, S]
        
        # Gather selected parameters
        means_flat = means.view(B * T, K)
        stds_flat = stds.view(B * T, K)
        
        sel_means = torch.gather(means_flat.unsqueeze(-1).expand(-1, -1, num_samples), 1, comp_idx.unsqueeze(1)).squeeze(1)  # [B*T, S]
        sel_stds = torch.gather(stds_flat.unsqueeze(-1).expand(-1, -1, num_samples), 1, comp_idx.unsqueeze(1)).squeeze(1)    # [B*T, S]
        
        # Reparameterization trick for sampling
        noise = torch.randn_like(sel_means)
        samples_flat = sel_means + sel_stds * noise  # [B*T, S]
        
        samples = samples_flat.view(B, T, num_samples).permute(2, 0, 1).contiguous()  # [S, B, T]
        return samples
    
    def prediction_summary(self, mixture_params):
        """
        Get summary statistics of the mixture distribution.
        
        Args:
            mixture_params: Tuple (means, log_stds, log_weights)
            
        Returns:
            pred_mean: [batch_size, pred_len]
            pred_std: [batch_size, pred_len]
        """
        means, log_stds, log_weights = mixture_params
        stds = torch.exp(log_stds).clamp_min(1e-6)
        weights = F.softmax(log_weights, dim=-1)
        
        # Mixture mean: E[X] = sum(w_k * mu_k)
        mixture_mean = torch.sum(weights * means, dim=-1)
        
        # Mixture variance: Var[X] = sum(w_k * (sigma_k^2 + mu_k^2)) - E[X]^2
        mixture_variance = torch.sum(weights * (stds**2 + means**2), dim=-1) - mixture_mean**2
        mixture_std = torch.sqrt(mixture_variance.clamp_min(1e-8))
        
        return mixture_mean, mixture_std

class MixtureNLLLoss(nn.Module):
    """Negative Log-Likelihood Loss for Mixture Density Networks with Multivariate Support."""
    
    def __init__(self, eps=1e-8, multivariate_mode='independent'):
        """
        Args:
            eps: Small constant for numerical stability
            multivariate_mode: How to handle multiple target features
                - 'independent': Treat each target feature independently (current behavior)
                - 'joint': Model joint distribution across all target features
                - 'first_only': Use only first target feature (fallback)
        """
        super().__init__()
        self.eps = eps
        self.multivariate_mode = multivariate_mode
        
    def forward(self, mixture_params, targets=None, *args):
        """
        Compute the negative log-likelihood of targets under the mixture distribution.
        
        Args:
            mixture_params: Tuple (means, log_stds, log_weights) or legacy (means, stds, weights, targets)
            targets: Tensor of shape (batch_size, pred_len)
            
        Returns:
            loss: Scalar tensor
        """
        # Backward compatibility: allow (means, stds, weights, targets)
        if targets is None and isinstance(mixture_params, tuple) and len(mixture_params) == 4:
            means, stds, weights, targets = mixture_params
            log_stds = torch.log(stds.clamp_min(self.eps))
            log_weights = torch.log(weights.clamp_min(self.eps))
        else:
            means, log_stds, log_weights = mixture_params
            if targets is None and len(args) > 0:
                targets = args[0]
            assert targets is not None, "Targets must be provided."
        
        # Convert parameters with hardening
        # Clamp log_stds to prevent overflow/underflow
        # TIGHTENED CLAMPING: [-5.0, 5.0] to prevent extreme penalty on OOB data
        log_stds = torch.clamp(log_stds, min=-5.0, max=5.0)
        stds = torch.exp(log_stds).clamp_min(self.eps)
        
        # Check for NaNs
        if torch.isnan(mixture_params[0]).any() or torch.isnan(log_stds).any():
             print("Warning: NaNs detected in MixtureNLLLoss inputs")
        
        log_weights = F.log_softmax(log_weights, dim=-1)
        
        # Handle targets with different shapes - MULTIVARIATE SUPPORT
        if targets.dim() > 2:
            targets = targets.view(targets.size(0), targets.size(1), -1)
            if targets.size(-1) == 1:
                targets = targets.squeeze(-1)  # [batch, pred_len]
                return self._compute_univariate_nll(means, log_stds, log_weights, targets)
            else:
                # Multiple target features - handle based on mode
                return self._compute_multivariate_nll(means, log_stds, log_weights, targets)
        else:
            # Single target feature
            return self._compute_univariate_nll(means, log_stds, log_weights, targets)
        
    def _compute_univariate_nll(self, means, log_stds, log_weights, targets):
        """Compute NLL for single target feature (original implementation)."""
        # Convert parameters with hardening
        # TIGHTENED CLAMPING: [-5.0, 5.0]
        log_stds = torch.clamp(log_stds, min=-5.0, max=5.0)
        stds = torch.exp(log_stds).clamp_min(self.eps)
        log_weights = F.log_softmax(log_weights, dim=-1)
        
        # Expand targets to match mixture dimensions
        targets_expanded = targets.unsqueeze(-1).expand_as(means)
        
        # log N(x|mu,sigma) = -0.5*((x-mu)^2/sigma^2 + 2*log(sigma) + log(2*pi))
        log_probs = (
            -0.5 * ((targets_expanded - means) ** 2) / (stds ** 2 + self.eps)
            - torch.log(stds + self.eps)
            - 0.5 * np.log(2 * np.pi)
        )
        
        # log-sum over mixture components
        log_weighted = log_weights + log_probs
        max_log = torch.max(log_weighted, dim=-1, keepdim=True)[0]
        log_sum = max_log + torch.log(torch.sum(torch.exp(log_weighted - max_log), dim=-1, keepdim=True) + self.eps)
        log_sum = log_sum.squeeze(-1)  # [B, T]
        
        # Negative log-likelihood
        return -(log_sum.mean())
    
    def _compute_multivariate_nll(self, means, log_stds, log_weights, targets):
        """Compute NLL for multiple target features."""
        batch_size, pred_len, num_targets = targets.shape
        
        if self.multivariate_mode == 'independent':
            return self._compute_independent_nll(means, log_stds, log_weights, targets)
        elif self.multivariate_mode == 'joint':
            return self._compute_joint_nll(means, log_stds, log_weights, targets)
        else:  # 'first_only'
            return self._compute_univariate_nll(means, log_stds, log_weights, targets[:, :, 0])
    
    def _compute_independent_nll(self, means, log_stds, log_weights, targets):
        """Compute NLL treating each target feature independently."""
        batch_size, pred_len, num_targets = targets.shape
        
        # Convert parameters with hardening
        # TIGHTENED CLAMPING: [-5.0, 5.0]
        log_stds = torch.clamp(log_stds, min=-5.0, max=5.0)
        stds = torch.exp(log_stds).clamp_min(self.eps)
        log_weights = F.log_softmax(log_weights, dim=-1)
        
        total_nll = 0.0
        
        # Process each target feature independently
        for target_idx in range(num_targets):
            target_feature = targets[:, :, target_idx]  # [batch, pred_len]
            
            # Get means, stds, and weights for this target feature
            # FIXED: Check dimensions of each tensor independently
            if means.dim() == 4:  # [B, T, num_targets, K]
                target_means = means[:, :, target_idx, :]  # [B, T, K]
            else:  # [B, T, K] - shared across targets
                target_means = means
                
            if stds.dim() == 4:  # [B, T, num_targets, K]
                target_stds = stds[:, :, target_idx, :]    # [B, T, K]
            else:  # [B, T, K] - shared across targets
                target_stds = stds
                
            if log_weights.dim() == 4:  # [B, T, num_targets, K]
                target_log_weights = log_weights[:, :, target_idx, :]  # [B, T, K]
            else:  # [B, T, K] - shared weights across targets
                target_log_weights = log_weights
            
            # Expand target to match mixture dimensions
            target_expanded = target_feature.unsqueeze(-1).expand_as(target_means)  # [B, T, K]
            
            # Compute log probabilities for this target feature
            log_probs = (
                -0.5 * ((target_expanded - target_means) ** 2) / (target_stds ** 2 + self.eps)
                - torch.log(target_stds + self.eps)
                - 0.5 * np.log(2 * np.pi)
            )
            
            # log-sum over mixture components
            log_weighted = target_log_weights + log_probs
            max_log = torch.max(log_weighted, dim=-1, keepdim=True)[0]
            log_sum = max_log + torch.log(torch.sum(torch.exp(log_weighted - max_log), dim=-1, keepdim=True) + self.eps)
            log_sum = log_sum.squeeze(-1)  # [B, T]
            
            # Accumulate negative log-likelihood
            total_nll += -(log_sum.mean())
        
        # Average across target features
        return total_nll / num_targets
    
    def _compute_joint_nll(self, means, log_stds, log_weights, targets):
        """Compute NLL for joint multivariate distribution (simplified diagonal covariance)."""
        batch_size, pred_len, num_targets = targets.shape
        
        # Convert parameters with hardening
        # TIGHTENED CLAMPING: [-5.0, 5.0]
        log_stds = torch.clamp(log_stds, min=-5.0, max=5.0)
        stds = torch.exp(log_stds).clamp_min(self.eps)
        log_weights = F.log_softmax(log_weights, dim=-1)
        
        # Handle different mean/std shapes
        if means.dim() == 4:  # [B, T, num_targets, K] - multivariate
            # Transpose to [B, T, K, num_targets] for easier computation
            means_joint = means.transpose(2, 3)  # [B, T, K, num_targets]
            stds_joint = stds.transpose(2, 3)    # [B, T, K, num_targets]
        else:  # [B, T, K] - shared across targets
            # Expand to multivariate dimensions
            means_joint = means.unsqueeze(-1).expand(-1, -1, -1, num_targets)  # [B, T, K, num_targets]
            stds_joint = stds.unsqueeze(-1).expand(-1, -1, -1, num_targets)    # [B, T, K, num_targets]
        
        # Expand targets to match mixture dimensions
        targets_expanded = targets.unsqueeze(2).expand(-1, -1, means_joint.size(2), -1)  # [B, T, K, num_targets]
        
        # Compute log probabilities for joint distribution (assuming independence for simplicity)
        # Full multivariate would use: log|2πΣ| + (x-μ)ᵀΣ⁻¹(x-μ)
        log_probs_per_target = (
            -0.5 * ((targets_expanded - means_joint) ** 2) / (stds_joint ** 2 + self.eps)
            - torch.log(stds_joint + self.eps)
            - 0.5 * np.log(2 * np.pi)
        )
        
        # Sum log probabilities across target features (assuming independence)
        log_probs = log_probs_per_target.sum(dim=-1)  # [B, T, K]
        
        # log-sum over mixture components
        log_weighted = log_weights + log_probs
        max_log = torch.max(log_weighted, dim=-1, keepdim=True)[0]
        log_sum = max_log + torch.log(torch.sum(torch.exp(log_weighted - max_log), dim=-1, keepdim=True) + self.eps)
        log_sum = log_sum.squeeze(-1)  # [B, T]
        
        # Negative log-likelihood
        return -(log_sum.mean())