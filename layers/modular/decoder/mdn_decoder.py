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
    Mixture Density Network decoder for sequence prediction tasks with dimension validation.
    
    Outputs a Gaussian mixture distribution per target and timestep, enabling
    probabilistic forecasting with calibrated uncertainty estimates.
    
    Features:
    - Automatic dimension adaptation for input features
    - Robust dimension validation and error handling
    - Numerically stable parameterization
    
    Args:
        d_input: Expected dimensionality of input hidden states (from encoder/decoder).
        n_targets: Number of target variables to predict.
        n_components: Number of Gaussian mixture components (K).
        sigma_min: Minimum standard deviation floor to prevent collapse.
        use_softplus: If True, use softplus(·) for σ; else exp(logσ).
        adaptive_input: If True, adapt to different input dimensions automatically.
    """

    def __init__(
        self,
        d_input: int,
        n_targets: int,
        n_components: int = 5,
        sigma_min: float = 1e-3,
        use_softplus: bool = True,
        adaptive_input: bool = True,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.n_targets = n_targets
        self.n_components = n_components
        self.sigma_min = sigma_min
        self.use_softplus = use_softplus
        self.adaptive_input = adaptive_input

        # Input dimension adaptation layer (if needed)
        if adaptive_input:
            self.input_adapter = None  # Will be created dynamically
        else:
            self.input_adapter = None

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

    def _validate_and_adapt_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Validate input dimensions and adapt if necessary.
        
        Args:
            hidden_states: [batch, seq_len, d_actual] input tensor
            
        Returns:
            adapted_states: [batch, seq_len, d_input] adapted tensor
            
        Raises:
            ValueError: If dimensions cannot be adapted
        """
        batch_size, seq_len, d_actual = hidden_states.shape
        
        if d_actual == self.d_input:
            # Perfect match, no adaptation needed
            return hidden_states
        
        if not self.adaptive_input:
            raise ValueError(
                f"MDNDecoder expected input dimension {self.d_input}, "
                f"but got {d_actual}. Set adaptive_input=True to enable automatic adaptation."
            )
        
        # Create or update input adapter if needed
        if self.input_adapter is None or self.input_adapter.in_features != d_actual:
            self.input_adapter = nn.Linear(d_actual, self.d_input).to(hidden_states.device)
            # Initialize with small weights for stability
            nn.init.xavier_uniform_(self.input_adapter.weight, gain=0.1)
            nn.init.zeros_(self.input_adapter.bias)
            
            print(f"MDNDecoder: Created input adapter {d_actual} -> {self.d_input}")
        
        # Adapt the input dimensions
        adapted_states = self.input_adapter(hidden_states)
        return adapted_states

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mixture parameters from decoder hidden states with dimension validation.
        
        Args:
            hidden_states: [batch, seq_len, d_actual] decoder outputs (any dimension).
        
        Returns:
            pi: [batch, seq_len, n_targets, n_components] mixture weights (normalized).
            mu: [batch, seq_len, n_targets, n_components] component means.
            sigma: [batch, seq_len, n_targets, n_components] component std (≥ sigma_min).
            
        Raises:
            ValueError: If input dimensions are invalid and cannot be adapted.
        """
        try:
            # Validate and adapt input dimensions
            adapted_states = self._validate_and_adapt_input(hidden_states)
            batch_size, seq_len, _ = adapted_states.shape

            # Project to mixture parameters
            pi_logits = self.pi_head(adapted_states)  # [B, S, T*K]
            mu_raw = self.mu_head(adapted_states)      # [B, S, T*K]
            sigma_raw = self.sigma_head(adapted_states)  # [B, S, T*K]

            # Reshape to [B, S, T, K]
            pi_logits = pi_logits.view(batch_size, seq_len, self.n_targets, self.n_components)
            mu = mu_raw.view(batch_size, seq_len, self.n_targets, self.n_components)
            sigma_param = sigma_raw.view(batch_size, seq_len, self.n_targets, self.n_components)

            # Normalize mixture weights via softmax over components
            pi = F.softmax(pi_logits, dim=-1)

            # Compute positive std with floor (numerically stable)
            if self.use_softplus:
                sigma = F.softplus(sigma_param) + self.sigma_min
            else:
                # Clamp before exp to prevent overflow
                sigma_param_clamped = torch.clamp(sigma_param, min=-10, max=10)
                sigma = torch.exp(sigma_param_clamped).clamp(min=self.sigma_min)

            # Validate output shapes
            expected_shape = (batch_size, seq_len, self.n_targets, self.n_components)
            assert pi.shape == expected_shape, f"pi shape {pi.shape} != expected {expected_shape}"
            assert mu.shape == expected_shape, f"mu shape {mu.shape} != expected {expected_shape}"
            assert sigma.shape == expected_shape, f"sigma shape {sigma.shape} != expected {expected_shape}"

            return pi, mu, sigma
            
        except Exception as e:
            print(f"MDNDecoder forward error: {e}")
            print(f"Input shape: {hidden_states.shape}")
            print(f"Expected d_input: {self.d_input}")
            print(f"n_targets: {self.n_targets}, n_components: {self.n_components}")
            raise

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
        Compute the expected value E[Y] = Σ π_k μ_k for point predictions with validation.
        
        Args:
            pi: [batch, seq_len, n_targets, n_components] mixture weights.
            mu: [batch, seq_len, n_targets, n_components] component means.
        
        Returns:
            mean: [batch, seq_len, n_targets] expected predictions.
            
        Raises:
            ValueError: If input shapes are incompatible.
        """
        try:
            # Validate input shapes
            if pi.shape != mu.shape:
                raise ValueError(f"pi shape {pi.shape} != mu shape {mu.shape}")
            
            batch_size, seq_len, n_targets, n_components = pi.shape
            
            if n_targets != self.n_targets:
                raise ValueError(f"Expected {self.n_targets} targets, got {n_targets}")
            if n_components != self.n_components:
                raise ValueError(f"Expected {self.n_components} components, got {n_components}")
            
            # Compute weighted mean
            mean = (pi * mu).sum(dim=-1)  # [B, S, T]
            
            # Validate output shape
            expected_shape = (batch_size, seq_len, self.n_targets)
            assert mean.shape == expected_shape, f"Output shape {mean.shape} != expected {expected_shape}"
            
            return mean
            
        except Exception as e:
            print(f"MDNDecoder mean_prediction error: {e}")
            print(f"pi shape: {pi.shape}, mu shape: {mu.shape}")
            print(f"Expected targets: {self.n_targets}, components: {self.n_components}")
            raise


def mdn_nll_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    targets: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Compute the negative log-likelihood for a Gaussian mixture model with robust error handling.
    
    Args:
        pi: [batch, seq_len, n_targets, n_components] mixture weights.
        mu: [batch, seq_len, n_targets, n_components] component means.
        sigma: [batch, seq_len, n_targets, n_components] component std.
        targets: [batch, seq_len, n_targets] ground truth values.
        reduce: 'mean', 'sum', or 'none' for aggregation.
    
    Returns:
        loss: Scalar (if reduce='mean'/'sum') or [batch, seq_len, n_targets] (if 'none').
        
    Raises:
        ValueError: If tensor shapes are incompatible.
    """
    try:
        # Validate input shapes
        if len(pi.shape) != 4 or len(mu.shape) != 4 or len(sigma.shape) != 4:
            raise ValueError(f"Expected 4D tensors, got pi: {pi.shape}, mu: {mu.shape}, sigma: {sigma.shape}")
        
        if pi.shape != mu.shape or pi.shape != sigma.shape:
            raise ValueError(f"Shape mismatch: pi {pi.shape}, mu {mu.shape}, sigma {sigma.shape}")
        
        batch_size, seq_len, n_targets, n_components = pi.shape
        device = pi.device
        
        # Validate targets shape
        expected_target_shape = (batch_size, seq_len, n_targets)
        if targets.shape != expected_target_shape:
            raise ValueError(f"Target shape {targets.shape} != expected {expected_target_shape}")

        # Expand targets to [B, S, T, 1] for broadcasting
        targets_expanded = targets.unsqueeze(-1)  # [B, S, T, 1]

        # Compute log Gaussian probabilities per component with numerical stability
        log_2pi = torch.tensor(1.8378770664093453, device=device)  # log(2π)
        
        # Clamp sigma to prevent division by zero and log(0)
        sigma_clamped = torch.clamp(sigma, min=1e-6)
        
        z_scores = (targets_expanded - mu) / sigma_clamped  # [B, S, T, K]
        
        # Clamp z_scores to prevent overflow in z_scores^2
        z_scores_clamped = torch.clamp(z_scores, min=-10, max=10)
        
        log_gauss = -0.5 * log_2pi - torch.log(sigma_clamped) - 0.5 * z_scores_clamped ** 2  # [B, S, T, K]

        # Weighted log probabilities: log(π_k) + log N(y | μ_k, σ_k)
        pi_clamped = torch.clamp(pi, min=1e-8, max=1.0)
        log_weighted = torch.log(pi_clamped) + log_gauss  # [B, S, T, K]

        # Log-sum-exp over components for numerical stability
        log_prob = torch.logsumexp(log_weighted, dim=-1)  # [B, S, T]

        # Negative log-likelihood with NaN/Inf checking
        nll = -log_prob
        
        # Check for NaN or Inf values
        if torch.isnan(nll).any() or torch.isinf(nll).any():
            print("Warning: NaN or Inf detected in MDN loss computation")
            print(f"pi range: [{pi.min():.6f}, {pi.max():.6f}]")
            print(f"mu range: [{mu.min():.6f}, {mu.max():.6f}]")
            print(f"sigma range: [{sigma.min():.6f}, {sigma.max():.6f}]")
            print(f"targets range: [{targets.min():.6f}, {targets.max():.6f}]")
            
            # Return a safe fallback loss
            return torch.tensor(1.0, device=device, requires_grad=True)

        if reduce == "mean":
            return nll.mean()
        elif reduce == "sum":
            return nll.sum()
        else:
            return nll
            
    except Exception as e:
        print(f"MDN loss computation error: {e}")
        print(f"pi shape: {pi.shape if 'pi' in locals() else 'undefined'}")
        print(f"mu shape: {mu.shape if 'mu' in locals() else 'undefined'}")
        print(f"sigma shape: {sigma.shape if 'sigma' in locals() else 'undefined'}")
        print(f"targets shape: {targets.shape if 'targets' in locals() else 'undefined'}")
        raise
