import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MixtureDensityDecoder(nn.Module):
    """Mixture Density Network decoder that predicts a mixture of K Gaussian distributions.
    
    This decoder can capture multi-modal uncertainty by predicting multiple Gaussian
    components with their respective means, standard deviations, and mixture weights.
    """
    
    def __init__(self, d_model, num_mixtures=5):
        """
        Args:
            d_model: Input feature dimension
            num_mixtures: Number of Gaussian components in the mixture (K)
        """
        super().__init__()
        self.num_mixtures = num_mixtures
        
        # Shared feature processing
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for each mixture component parameter
        self.mean_heads = nn.Linear(d_model // 2, num_mixtures)
        self.std_dev_heads = nn.Linear(d_model // 2, num_mixtures)
        self.mixture_weight_heads = nn.Linear(d_model // 2, num_mixtures)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_targets, d_model)
            
        Returns:
            means: Tensor of shape (batch_size, num_targets, num_mixtures)
            std_devs: Tensor of shape (batch_size, num_targets, num_mixtures)
            mixture_weights: Tensor of shape (batch_size, num_targets, num_mixtures)
        """
        processed = self.mlp(x)
        
        # Predict means (no activation needed)
        means = self.mean_heads(processed)
        
        # Predict standard deviations (ensure positive with softplus)
        std_devs = F.softplus(self.std_dev_heads(processed)) + 1e-6
        
        # Predict mixture weights (ensure they sum to 1 with softmax)
        mixture_weights = F.softmax(self.mixture_weight_heads(processed), dim=-1)
        
        return means, std_devs, mixture_weights
    
    def sample(self, means, std_devs, mixture_weights, num_samples=1):
        """
        Sample from the mixture distribution.
        
        Args:
            means: Tensor of shape (batch_size, num_targets, num_mixtures)
            std_devs: Tensor of shape (batch_size, num_targets, num_mixtures)
            mixture_weights: Tensor of shape (batch_size, num_targets, num_mixtures)
            num_samples: Number of samples to draw
            
        Returns:
            samples: Tensor of shape (batch_size, num_targets, num_samples)
        """
        batch_size, num_targets, num_mixtures = means.shape
        
        # Sample which mixture component to use for each sample
        mixture_indices = torch.multinomial(
            mixture_weights.view(-1, num_mixtures), 
            num_samples, 
            replacement=True
        ).view(batch_size, num_targets, num_samples)
        
        # Gather the corresponding means and std_devs
        selected_means = torch.gather(
            means.unsqueeze(-1).expand(-1, -1, -1, num_samples),
            dim=2,
            index=mixture_indices.unsqueeze(2)
        ).squeeze(2)
        
        selected_std_devs = torch.gather(
            std_devs.unsqueeze(-1).expand(-1, -1, -1, num_samples),
            dim=2,
            index=mixture_indices.unsqueeze(2)
        ).squeeze(2)
        
        # Sample from the selected Gaussian components
        noise = torch.randn_like(selected_means)
        samples = selected_means + selected_std_devs * noise
        
        return samples
    
    def get_prediction_summary(self, means, std_devs, mixture_weights):
        """
        Get summary statistics of the mixture distribution.
        
        Args:
            means: Tensor of shape (batch_size, num_targets, num_mixtures)
            std_devs: Tensor of shape (batch_size, num_targets, num_mixtures)
            mixture_weights: Tensor of shape (batch_size, num_targets, num_mixtures)
            
        Returns:
            dict with 'mean', 'variance', 'std_dev' of the mixture
        """
        # Mixture mean: E[X] = sum(w_k * mu_k)
        mixture_mean = torch.sum(mixture_weights * means, dim=-1)
        
        # Mixture variance: Var[X] = sum(w_k * (sigma_k^2 + mu_k^2)) - E[X]^2
        mixture_variance = torch.sum(
            mixture_weights * (std_devs**2 + means**2), dim=-1
        ) - mixture_mean**2
        
        mixture_std_dev = torch.sqrt(mixture_variance + 1e-8)
        
        return {
            'mean': mixture_mean,
            'variance': mixture_variance,
            'std_dev': mixture_std_dev
        }

class MixtureNLLLoss(nn.Module):
    """Negative Log-Likelihood Loss for Mixture Density Networks."""
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, means, std_devs, mixture_weights, targets):
        """
        Compute the negative log-likelihood of targets under the mixture distribution.
        
        Args:
            means: Tensor of shape (batch_size, num_targets, num_mixtures)
            std_devs: Tensor of shape (batch_size, num_targets, num_mixtures)
            mixture_weights: Tensor of shape (batch_size, num_targets, num_mixtures)
            targets: Tensor of shape (batch_size, num_targets)
            
        Returns:
            loss: Scalar tensor
        """
        # Expand targets to match mixture dimensions
        targets_expanded = targets.unsqueeze(-1).expand_as(means)
        
        # Compute log probability for each mixture component
        # log N(x|mu_k, sigma_k) = -0.5 * log(2*pi*sigma_k^2) - 0.5 * (x-mu_k)^2/sigma_k^2
        log_probs = (
            -0.5 * torch.log(2 * np.pi * std_devs**2 + self.eps)
            - 0.5 * (targets_expanded - means)**2 / (std_devs**2 + self.eps)
        )
        
        # Compute log mixture probability: log(sum(w_k * N(x|mu_k, sigma_k)))
        # Use log-sum-exp trick for numerical stability
        log_mixture_weights = torch.log(mixture_weights + self.eps)
        log_weighted_probs = log_mixture_weights + log_probs
        
        # Log-sum-exp
        max_log_prob = torch.max(log_weighted_probs, dim=-1, keepdim=True)[0]
        log_mixture_prob = max_log_prob + torch.log(
            torch.sum(torch.exp(log_weighted_probs - max_log_prob), dim=-1, keepdim=True) + self.eps
        )
        log_mixture_prob = log_mixture_prob.squeeze(-1)
        
        # Return negative log-likelihood
        return -log_mixture_prob.mean()