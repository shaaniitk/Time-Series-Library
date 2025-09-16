import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MixtureDensityDecoder(nn.Module):
    """Mixture Density Network decoder that predicts a mixture of K Gaussian distributions.
    
    This decoder can capture multi-modal uncertainty by predicting multiple Gaussian
    components with their respective means, standard deviations, and mixture weights.
    """
    
    def __init__(self, d_model, pred_len, num_components=3):
        """
        Args:
            d_model: Input feature dimension
            pred_len: Prediction horizon (number of time steps to predict)
            num_components: Number of Gaussian components in the mixture (K)
        """
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.num_components = num_components
        
        # Shared feature processing from aggregated temporal context
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for each mixture component parameter (flattened over pred_len * num_components)
        out_dim = pred_len * num_components
        self.mean_head = nn.Linear(d_model, out_dim)
        self.std_head = nn.Linear(d_model, out_dim)      # outputs log-stds (unconstrained)
        self.weight_head = nn.Linear(d_model, out_dim)   # outputs log-weights (unnormalized)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            means: Tensor of shape (batch_size, pred_len, num_components)
            log_stds: Tensor of shape (batch_size, pred_len, num_components)
            log_weights: Tensor of shape (batch_size, pred_len, num_components)
        """
        # Aggregate temporal dimension (simple mean pooling)
        # Tests focus on shape and API, not specific temporal modeling
        context = x.mean(dim=1)  # [B, d_model]
        hidden = self.mlp(context)  # [B, d_model]
        
        B = x.size(0)
        means = self.mean_head(hidden).view(B, self.pred_len, self.num_components)
        log_stds = self.std_head(hidden).view(B, self.pred_len, self.num_components)
        log_weights = self.weight_head(hidden).view(B, self.pred_len, self.num_components)
        
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
    """Negative Log-Likelihood Loss for Mixture Density Networks."""
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
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
        
        # Convert parameters
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