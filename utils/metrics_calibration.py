#!/usr/bin/env python3
"""
Calibration Metrics for Probabilistic Forecasting

Utilities for evaluating predictive distribution quality via coverage, sharpness,
and proper scoring rules (CRPS).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_mixture_quantiles(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    quantiles: List[float],
    n_samples: int = 1000,
) -> torch.Tensor:
    """
    Estimate quantiles of a Gaussian mixture distribution via Monte Carlo sampling.
    
    Args:
        pi: [batch, seq_len, n_targets, n_components] mixture weights.
        mu: [batch, seq_len, n_targets, n_components] component means.
        sigma: [batch, seq_len, n_targets, n_components] component std.
        quantiles: List of quantile levels in [0, 1].
        n_samples: Number of MC samples for quantile estimation.
    
    Returns:
        quantile_values: [batch, seq_len, n_targets, len(quantiles)] estimated quantiles.
    """
    batch_size, seq_len, n_targets, n_components = pi.shape
    device = pi.device

    # Sample from the mixture
    # Flatten to [B*S*T, K] for categorical sampling
    pi_flat = pi.view(-1, n_components)
    component_indices = torch.multinomial(
        pi_flat, num_samples=n_samples, replacement=True
    )  # [B*S*T, n_samples]
    component_indices = component_indices.view(batch_size, seq_len, n_targets, n_samples)

    # Gather corresponding means and stds
    mu_expanded = mu.unsqueeze(3).expand(-1, -1, -1, n_samples, -1)
    sigma_expanded = sigma.unsqueeze(3).expand(-1, -1, -1, n_samples, -1)
    component_idx = component_indices.unsqueeze(-1)

    selected_mu = torch.gather(mu_expanded, dim=-1, index=component_idx).squeeze(-1)
    selected_sigma = torch.gather(sigma_expanded, dim=-1, index=component_idx).squeeze(-1)

    # Draw Gaussian samples
    noise = torch.randn(batch_size, seq_len, n_targets, n_samples, device=device)
    samples = selected_mu + selected_sigma * noise  # [B, S, T, n_samples]

    # Compute quantiles
    quantile_tensors = torch.tensor(quantiles, device=device, dtype=samples.dtype)
    quantile_values = torch.quantile(samples, quantile_tensors, dim=-1)  # [Q, B, S, T]

    return quantile_values.permute(1, 2, 3, 0)  # [B, S, T, Q]


def compute_coverage(
    predictions: np.ndarray,
    targets: np.ndarray,
    pi: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    coverage_levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute empirical coverage rates for predictive intervals.
    
    For Gaussian mixtures, intervals are derived from quantiles. For point predictions,
    this metric is not applicable and returns empty dict.
    
    Args:
        predictions: [n_samples, pred_len, n_targets] point predictions (unused if mixture given).
        targets: [n_samples, pred_len, n_targets] ground truth values.
        pi: [n_samples, pred_len, n_targets, K] mixture weights (optional).
        mu: [n_samples, pred_len, n_targets, K] component means (optional).
        sigma: [n_samples, pred_len, n_targets, K] component std (optional).
        coverage_levels: List of nominal coverage levels (e.g., [0.5, 0.9]).
    
    Returns:
        coverage_dict: {f'coverage_{level}': empirical_rate, ...}
    """
    if pi is None or mu is None or sigma is None:
        # Point predictions only; coverage not applicable
        return {}

    if coverage_levels is None:
        coverage_levels = [0.5, 0.9]

    coverage_dict: Dict[str, float] = {}

    for level in coverage_levels:
        # Compute lower and upper quantiles for symmetric interval
        lower_q = (1.0 - level) / 2.0
        upper_q = 1.0 - lower_q
        quantiles_torch = compute_mixture_quantiles(
            pi, mu, sigma, quantiles=[lower_q, upper_q], n_samples=1000
        )  # [N, S, T, 2]

        lower_bounds = quantiles_torch[..., 0].detach().cpu().numpy()
        upper_bounds = quantiles_torch[..., 1].detach().cpu().numpy()

        # Check if targets fall within [lower, upper]
        in_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
        empirical_coverage = in_interval.mean()

        coverage_dict[f"coverage_{int(level * 100)}"] = float(empirical_coverage)

    return coverage_dict


def compute_crps_gaussian_mixture(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    targets: torch.Tensor,
    n_samples: int = 100,
) -> float:
    """
    Estimate CRPS for a Gaussian mixture via Monte Carlo sampling.
    
    CRPS = E[|Y - y*|] - 0.5 * E[|Y - Y'|] where Y, Y' are independent draws.
    
    Args:
        pi: [batch, seq_len, n_targets, n_components] mixture weights.
        mu: [batch, seq_len, n_targets, n_components] component means.
        sigma: [batch, seq_len, n_targets, n_components] component std.
        targets: [batch, seq_len, n_targets] ground truth values.
        n_samples: Number of MC samples for CRPS estimation.
    
    Returns:
        crps: Scalar CRPS estimate averaged over all samples and targets.
    """
    batch_size, seq_len, n_targets, n_components = pi.shape
    device = pi.device

    # Draw samples Y ~ mixture
    pi_flat = pi.view(-1, n_components)
    component_indices = torch.multinomial(
        pi_flat, num_samples=n_samples, replacement=True
    )  # [B*S*T, n_samples]
    component_indices = component_indices.view(batch_size, seq_len, n_targets, n_samples)

    mu_expanded = mu.unsqueeze(3).expand(-1, -1, -1, n_samples, -1)
    sigma_expanded = sigma.unsqueeze(3).expand(-1, -1, -1, n_samples, -1)
    component_idx = component_indices.unsqueeze(-1)

    selected_mu = torch.gather(mu_expanded, dim=-1, index=component_idx).squeeze(-1)
    selected_sigma = torch.gather(sigma_expanded, dim=-1, index=component_idx).squeeze(-1)

    noise = torch.randn(batch_size, seq_len, n_targets, n_samples, device=device)
    samples_Y = selected_mu + selected_sigma * noise  # [B, S, T, n_samples]

    # Term 1: E[|Y - y*|]
    targets_expanded = targets.unsqueeze(-1)  # [B, S, T, 1]
    abs_error = torch.abs(samples_Y - targets_expanded).mean(dim=-1)  # [B, S, T]

    # Term 2: 0.5 * E[|Y - Y'|] where Y' is independent draw
    # Draw another set of samples
    component_indices_prime = torch.multinomial(
        pi_flat, num_samples=n_samples, replacement=True
    ).view(batch_size, seq_len, n_targets, n_samples)
    component_idx_prime = component_indices_prime.unsqueeze(-1)

    selected_mu_prime = torch.gather(mu_expanded, dim=-1, index=component_idx_prime).squeeze(-1)
    selected_sigma_prime = torch.gather(sigma_expanded, dim=-1, index=component_idx_prime).squeeze(-1)

    noise_prime = torch.randn(batch_size, seq_len, n_targets, n_samples, device=device)
    samples_Y_prime = selected_mu_prime + selected_sigma_prime * noise_prime

    # Pairwise differences (broadcast over samples)
    # E[|Y - Y'|] ≈ mean over all pairs; for efficiency, sample n_samples pairs
    abs_diff = torch.abs(samples_Y - samples_Y_prime).mean(dim=-1)  # [B, S, T]

    crps_per_sample = abs_error - 0.5 * abs_diff
    return float(crps_per_sample.mean().item())


def log_calibration_metrics(
    logger,
    predictions: np.ndarray,
    targets: np.ndarray,
    pi: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    coverage_levels: Optional[List[float]] = None,
    compute_crps: bool = False,
    crps_samples: int = 100,
) -> Dict[str, float]:
    """
    Compute and log calibration metrics to a logger (file-based).
    
    Args:
        logger: Logger instance for file output (e.g., MEMORY_LOGGER).
        predictions: [n_samples, pred_len, n_targets] point predictions.
        targets: [n_samples, pred_len, n_targets] ground truth.
        pi, mu, sigma: Mixture parameters (optional).
        coverage_levels: Nominal coverage levels to evaluate.
        compute_crps: Whether to compute CRPS (expensive).
        crps_samples: Number of MC samples for CRPS.
    
    Returns:
        metrics_dict: Calibration metrics for potential downstream use.
    """
    metrics_dict: Dict[str, float] = {}

    # Coverage
    if pi is not None and mu is not None and sigma is not None:
        coverage = compute_coverage(
            predictions, targets, pi, mu, sigma, coverage_levels=coverage_levels
        )
        metrics_dict.update(coverage)
        for key, val in coverage.items():
            logger.info(f"Calibration | {key}={val:.4f}")

    # CRPS
    if compute_crps and pi is not None and mu is not None and sigma is not None:
        crps_value = compute_crps_gaussian_mixture(
            pi, mu, sigma, torch.from_numpy(targets).to(pi.device), n_samples=crps_samples
        )
        metrics_dict["crps"] = crps_value
        logger.info(f"Calibration | crps={crps_value:.4f}")

    return metrics_dict
