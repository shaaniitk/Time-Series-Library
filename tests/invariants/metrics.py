"""Reusable metric helpers for invariant tests."""
from __future__ import annotations

import torch

Tensor = torch.Tensor


def l2_relative_error(a: Tensor, b: Tensor) -> float:
    eps = 1e-12
    return float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + eps))


def high_frequency_energy(x: Tensor, cutoff_ratio: float = 0.33) -> float:
    B, L, D = x.shape
    fft = torch.fft.rfft(x, dim=1)
    mag = fft.abs()
    k_cut = int((L // 2 + 1) * cutoff_ratio)
    high = mag[:, k_cut:, :].pow(2).sum()
    total = mag.pow(2).sum() + 1e-12
    return float(high / total)


def dominant_frequency_bin(x: Tensor) -> int:
    fft = torch.fft.rfft(x, dim=1)
    mag = fft.abs().sum(dim=(0, 2))
    return int(torch.argmax(mag).item())


def seasonal_zero_mean_ok(seasonal: Tensor, x: Tensor, ratio: float) -> bool:
    mean_abs = seasonal.mean().abs()
    std = x.std() + 1e-12
    return bool(mean_abs <= ratio * std)


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    num = torch.sum(a * b)
    denom = torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-12
    return float(num / denom)


def near_orthogonal(a: Tensor, b: Tensor, tol: float) -> bool:
    return abs(cosine_similarity(a, b)) < tol


def effective_rank(x: Tensor, energy_ratio: float = 0.9) -> int:
    """Compute an approximate effective rank based on cumulative singular value energy.

    Args:
        x: Tensor [..., features] flattened to 2D for SVD (batch/time collapsed).
        energy_ratio: Fraction of cumulative singular value energy to retain.

    Returns:
        Smallest k such that cumulative energy >= energy_ratio * total.
    """
    if x.dim() > 2:
        x2 = x.reshape(-1, x.shape[-1])
    else:
        x2 = x
    # center
    x2 = x2 - x2.mean(0, keepdim=True)
    try:
        # Use economical SVD
        U, S, Vh = torch.linalg.svd(x2, full_matrices=False)
    except RuntimeError:
        # Fallback to CPU if GPU SVD fails
        S = torch.linalg.svdvals(x2.cpu())
    total = torch.sum(S)
    if total <= 0:
        return 0
    cumsum = torch.cumsum(S, dim=0)
    thresh = energy_ratio * total
    k = int((cumsum >= thresh).nonzero(as_tuple=False)[0].item() + 1)
    return k


def gaussian_crps(mean: Tensor, std: Tensor, y: Tensor) -> Tensor:
    """Closed-form CRPS for Gaussian forecasts.

    Reference: Hersbach (2000). Implemented elementwise.
    Args:
        mean: Mean tensor
        std: Standard deviation tensor (non-negative)
        y: Observations (broadcastable to mean)
    Returns:
        Tensor of CRPS values (same shape as mean)
    """
    std = std.clamp_min(1e-12)
    z = (y - mean) / std
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    cdf_z = normal.cdf(z)
    pdf_z = torch.exp(normal.log_prob(z))
    # CRPS = std * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
    inv_sqrt_pi = 1.0 / torch.sqrt(torch.tensor(torch.pi))
    crps = std * (z * (2 * cdf_z - 1) + 2 * pdf_z - inv_sqrt_pi)
    return crps


def pit_uniform_ks(pit: Tensor) -> float:
    """Kolmogorov-Smirnov distance of PIT values from Uniform(0,1).

    Args:
        pit: Tensor of PIT values in (0,1)
    Returns:
        KS distance float
    """
    pit = pit.flatten().sort().values
    n = pit.numel()
    ecdf = torch.arange(1, n + 1, device=pit.device, dtype=pit.dtype) / n
    d_plus = torch.max(torch.abs(pit - ecdf))
    d_minus = torch.max(torch.abs(pit - (ecdf - 1 / n)))
    return float(torch.max(d_plus, d_minus))


__all__ = [
    "l2_relative_error",
    "high_frequency_energy",
    "dominant_frequency_bin",
    "seasonal_zero_mean_ok",
    "cosine_similarity",
    "near_orthogonal",
    "effective_rank",
    "gaussian_crps",
    "pit_uniform_ks",
]
