#!/usr/bin/env python3
"""
Mixture Density Network (MDN) Decoder for Probabilistic Time Series Forecasting

Produces calibrated predictive distributions via Gaussian mixture components with
stable parameterization and numerically robust loss computation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNDecoder(nn.Module):
    """
    Mixture Density Network decoder for sequence prediction tasks.
    
    Outputs a Gaussian mixture distribution per target and timestep, enabling
    probabilistic forecasting with calibrated uncertainty estimates.
    
    Args:
        d_input: Dimensionality of input hidden states (from encoder/decoder).
        n_targets: Number of target variables to predict.
        n_components: Number of Gaussian mixture components (K).
        sigma_min: Minimum standard deviation floor to prevent collapse.
        use_softplus: If True, use softplus(·) for σ; else exp(logσ).
    """

    def __init__(
        self,
        d_input: int,
        n_targets: int,
        n_components: int = 5,
        sigma_min: float = 1e-3,
        use_softplus: bool = True,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.n_targets = n_targets
        self.n_components = n_components
        self.sigma_min = sigma_min
        self.use_softplus = use_softplus

        # Mixture weights (logits) projection
        self.pi_head = nn.Linear(d_input, n_targets * n_components)
        
        # Component means projection
        self.mu_head = nn.Linear(d_input, n_targets * n_components)
        
        # Component std (log or pre-softplus) projection
        self.sigma_head = nn.Linear(d_input, n_targets * n_components)

        # Initialize heads with small weights for stability
        nn.init.xavier_uniform_(self.pi_head.weight, gain=0.1)
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.1)
        nn.init.xavier_uniform_(self.sigma_head.weight, gain=0.1)
        nn.init.zeros_(self.pi_head.bias)
        nn.init.zeros_(self.mu_head.bias)
        # Bias sigma head to produce σ ≈ 1.0 initially
        if use_softplus:
            nn.init.constant_(self.sigma_head.bias, 1.0)
        else:
            nn.init.constant_(self.sigma_head.bias, 0.0)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mixture parameters from decoder hidden states.
        
        Args:
            hidden_states: [batch, seq_len, d_input] decoder outputs.
        
        Returns:
            pi: [batch, seq_len, n_targets, n_components] mixture weights (normalized).
            mu: [batch, seq_len, n_targets, n_components] component means.
            sigma: [batch, seq_len, n_targets, n_components] component std (≥ sigma_min).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to mixture parameters
        pi_logits = self.pi_head(hidden_states)  # [B, S, T*K]
        mu_raw = self.mu_head(hidden_states)      # [B, S, T*K]
        sigma_raw = self.sigma_head(hidden_states)  # [B, S, T*K]

        # Reshape to [B, S, T, K]
        pi_logits = pi_logits.view(batch_size, seq_len, self.n_targets, self.n_components)
        mu = mu_raw.view(batch_size, seq_len, self.n_targets, self.n_components)
        sigma_param = sigma_raw.view(batch_size, seq_len, self.n_targets, self.n_components)

        # Normalize mixture weights via softmax over components
        pi = F.softmax(pi_logits, dim=-1)

        # Compute positive std with floor
        if self.use_softplus:
            sigma = F.softplus(sigma_param) + self.sigma_min
        else:
            sigma = torch.exp(sigma_param).clamp(min=self.sigma_min)

        return pi, mu, sigma

    def sample(
        self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Draw samples from the Gaussian mixture distribution.
        
        Args:
            pi: [batch, seq_len, n_targets, n_components] mixture weights.
            mu: [batch, seq_len, n_targets, n_components] component means.
            sigma: [batch, seq_len, n_targets, n_components] component std.
            n_samples: Number of samples to draw per (batch, seq, target).
        
        Returns:
            samples: [n_samples, batch, seq_len, n_targets] drawn from the mixture.
        """
        batch_size, seq_len, n_targets, n_components = pi.shape
        device = pi.device

        # Sample component indices via categorical
        # Reshape pi to [B*S*T, K] for sampling
        pi_flat = pi.view(-1, n_components)
        component_indices = torch.multinomial(
            pi_flat, num_samples=n_samples, replacement=True
        )  # [B*S*T, n_samples]
        component_indices = component_indices.view(batch_size, seq_len, n_targets, n_samples)

        # Gather selected means and stds
        # Expand for indexing: [B, S, T, n_samples, K] and select via gather
        mu_expanded = mu.unsqueeze(3).expand(-1, -1, -1, n_samples, -1)  # [B,S,T,n_samples,K]
        sigma_expanded = sigma.unsqueeze(3).expand(-1, -1, -1, n_samples, -1)

        component_indices_expanded = component_indices.unsqueeze(-1)  # [B,S,T,n_samples,1]
        selected_mu = torch.gather(mu_expanded, dim=-1, index=component_indices_expanded).squeeze(-1)
        selected_sigma = torch.gather(sigma_expanded, dim=-1, index=component_indices_expanded).squeeze(-1)

        # Draw Gaussian samples
        noise = torch.randn(batch_size, seq_len, n_targets, n_samples, device=device)
        samples = selected_mu + selected_sigma * noise  # [B, S, T, n_samples]

        # Transpose to [n_samples, B, S, T]
        return samples.permute(3, 0, 1, 2)

    def mean_prediction(
        self, pi: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expected value E[Y] = Σ π_k μ_k for point predictions.
        
        Args:
            pi: [batch, seq_len, n_targets, n_components] mixture weights.
            mu: [batch, seq_len, n_targets, n_components] component means.
        
        Returns:
            mean: [batch, seq_len, n_targets] expected predictions.
        """
        return (pi * mu).sum(dim=-1)


def mdn_nll_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    targets: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Compute the negative log-likelihood for a Gaussian mixture model with log-sum-exp stability.
    
    Args:
        pi: [batch, seq_len, n_targets, n_components] mixture weights.
        mu: [batch, seq_len, n_targets, n_components] component means.
        sigma: [batch, seq_len, n_targets, n_components] component std.
        targets: [batch, seq_len, n_targets] ground truth values.
        reduce: 'mean', 'sum', or 'none' for aggregation.
    
    Returns:
        loss: Scalar (if reduce='mean'/'sum') or [batch, seq_len, n_targets] (if 'none').
    """
    batch_size, seq_len, n_targets, n_components = pi.shape
    device = pi.device

    # Expand targets to [B, S, T, 1] for broadcasting
    targets_expanded = targets.unsqueeze(-1)  # [B, S, T, 1]

    # Compute log Gaussian probabilities per component
    # log N(y | μ_k, σ_k) = -0.5 * log(2π) - log(σ_k) - 0.5 * ((y - μ_k) / σ_k)^2
    log_2pi = torch.tensor(1.8378770664093453, device=device)  # log(2π)
    z_scores = (targets_expanded - mu) / sigma  # [B, S, T, K]
    log_gauss = -0.5 * log_2pi - torch.log(sigma) - 0.5 * z_scores ** 2  # [B, S, T, K]

    # Weighted log probabilities: log(π_k) + log N(y | μ_k, σ_k)
    log_weighted = torch.log(pi.clamp(min=1e-8)) + log_gauss  # [B, S, T, K]

    # Log-sum-exp over components for numerical stability
    log_prob = torch.logsumexp(log_weighted, dim=-1)  # [B, S, T]

    # Negative log-likelihood
    nll = -log_prob

    if reduce == "mean":
        return nll.mean()
    elif reduce == "sum":
        return nll.sum()
    else:
        return nll
