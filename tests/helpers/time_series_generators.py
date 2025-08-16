"""Synthetic time series generators for invariant & algorithmic tests.

All functions return a torch.Tensor of shape (batch, length, dim) unless noted.
Each generator aims to produce analytically tractable features so component
invariants can be asserted quantitatively.

Design goals:
  * Deterministic under supplied seed.
  * Vectorized (avoid python loops) for speed.
  * Minimal external deps (torch + optional numpy fallback).
  * Metadata friendly (optionally return dict when return_metadata=True).

Example:
    x, meta = sinusoid_mix(4, 256, 3, freqs=[3,7,13], return_metadata=True)
"""
from __future__ import annotations

from typing import Sequence, Tuple, Dict, Any, Optional
import math
import torch

Tensor = torch.Tensor


def _setup_seed(seed: Optional[int]):
    if seed is not None:
        torch.manual_seed(seed)


def sinusoid_mix(
    batch: int,
    length: int,
    dim: int,
    freqs: Optional[Sequence[int]] = None,
    amplitudes: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    noise_std: float = 0.0,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Multi-frequency sinusoidal mixture.

    Each feature dimension gets the same mixture (simple) for now; can extend
    to per-dim parameterization later.
    """
    _setup_seed(seed)
    t = torch.linspace(0, 2 * math.pi, length)
    if freqs is None:
        freqs = [3, 7]
    if amplitudes is None:
        amplitudes = [1.0 for _ in freqs]
    if phases is None:
        phases = [0.0 for _ in freqs]
    mix = torch.zeros(length)
    for f, a, p in zip(freqs, amplitudes, phases):
        mix = mix + a * torch.sin(f * t + p)
    if noise_std > 0:
        mix = mix + noise_std * torch.randn_like(mix)
    x = mix.view(1, length, 1).repeat(batch, 1, dim)
    meta = {"freqs": list(freqs), "amplitudes": list(amplitudes), "phases": list(phases)}
    return (x, meta) if return_metadata else x


def polynomial_trend(
    batch: int,
    length: int,
    dim: int,
    degree: int = 3,
    coeff_range: Tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 0.0,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Generate polynomial trend signals of given degree."""
    _setup_seed(seed)
    t = torch.linspace(-1, 1, length)
    coeffs = torch.empty(degree + 1).uniform_(*coeff_range)
    base = torch.zeros(length)
    for i, c in enumerate(coeffs):
        base = base + c * (t ** i)
    if noise_std > 0:
        base = base + noise_std * torch.randn_like(base)
    x = base.view(1, length, 1).repeat(batch, 1, dim)
    meta = {"coeffs": [float(c) for c in coeffs]}
    return (x, meta) if return_metadata else x


def seasonal_with_trend(
    batch: int,
    length: int,
    dim: int,
    freqs=(5, 11),
    trend_degree: int = 2,
    noise_std: float = 0.0,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Combine polynomial trend + sinusoidal seasonal components."""
    season, smeta = sinusoid_mix(batch, length, dim, freqs=freqs, seed=seed, return_metadata=True)
    trend, tmeta = polynomial_trend(batch, length, dim, degree=trend_degree, seed=seed, return_metadata=True)
    x = season + trend
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    meta = {"seasonal": smeta, "trend": tmeta}
    return (x, meta) if return_metadata else x


def step_changes(
    batch: int,
    length: int,
    dim: int,
    num_steps: int = 4,
    magnitude: float = 1.0,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Piecewise constant step function with random step magnitudes."""
    _setup_seed(seed)
    idx = torch.sort(torch.randint(1, length - 1, (num_steps,), dtype=torch.long))[0]
    levels = torch.randn(num_steps + 1) * magnitude
    base = torch.zeros(length)
    prev = 0
    for i, cut in enumerate(list(idx) + [length]):
        base[prev:cut] = levels[i]
        prev = cut
    x = base.view(1, length, 1).repeat(batch, 1, dim)
    meta = {"cut_points": [int(i) for i in idx], "levels": [float(l) for l in levels]}
    return (x, meta) if return_metadata else x


def impulse_train(
    batch: int,
    length: int,
    dim: int,
    sparsity: float = 0.95,
    amplitude: float = 5.0,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Sparse impulses for attention focusing tests."""
    _setup_seed(seed)
    num_impulses = max(1, int(length * (1 - sparsity)))
    positions = torch.randperm(length)[:num_impulses]
    signal = torch.zeros(length)
    signal[positions] = amplitude
    x = signal.view(1, length, 1).repeat(batch, 1, dim)
    meta = {"positions": [int(p) for p in positions]}
    return (x, meta) if return_metadata else x


def autoregressive_lagged(
    batch: int,
    length: int,
    dim: int,
    lags: Sequence[int] = (3, 7),
    weights: Optional[Sequence[float]] = None,
    noise_std: float = 0.01,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Generate series with known lag dependencies (AR-like)."""
    _setup_seed(seed)
    if weights is None:
        weights = [1.0 / len(lags) for _ in lags]
    base = torch.randn(batch, length + max(lags), dim) * 0.05
    for t in range(max(lags), length + max(lags)):
        val = torch.zeros(batch, dim)
        for lag, w in zip(lags, weights):
            val = val + w * base[:, t - lag, :]
        base[:, t, :] = val + noise_std * torch.randn(batch, dim)
    series = base[:, max(lags):, :]
    meta = {"lags": list(lags), "weights": [float(w) for w in weights]}
    return (series, meta) if return_metadata else series


def correlated_multivariate(
    batch: int,
    length: int,
    dim: int,
    rho: float = 0.8,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Generate multivariate correlated Gaussian time series (static corr)."""
    _setup_seed(seed)
    cov = torch.full((dim, dim), rho)
    cov.fill_diagonal_(1.0)
    L = torch.linalg.cholesky(cov)
    z = torch.randn(batch, length, dim)
    x = z @ L.T
    meta = {"rho": float(rho)}
    return (x, meta) if return_metadata else x


def piecewise_frequency_shift(
    batch: int,
    length: int,
    dim: int,
    segments: int = 3,
    base_freqs: Sequence[int] = (3, 9, 15),
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Series whose dominant frequency changes across segments."""
    _setup_seed(seed)
    assert segments == len(base_freqs), "segments must match length of base_freqs"
    seg_len = length // segments
    parts = []
    meta_freqs = []
    for f in base_freqs:
        t = torch.linspace(0, 2 * math.pi, seg_len)
        part = torch.sin(f * t)
        parts.append(part)
        meta_freqs.append(int(f))
    series = torch.cat(parts)
    if series.shape[0] < length:  # pad remainder
        series = torch.nn.functional.pad(series, (0, length - series.shape[0]))
    x = series.view(1, length, 1).repeat(batch, 1, dim)
    meta = {"segment_freqs": meta_freqs, "segment_length": int(seg_len)}
    return (x, meta) if return_metadata else x


def noise_only(
    batch: int,
    length: int,
    dim: int,
    variance: float = 1.0,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any]] | Tensor:
    """Pure Gaussian noise baseline without structure.

    Useful for asserting components don't hallucinate deterministic patterns.
    """
    _setup_seed(seed)
    x = torch.randn(batch, length, dim) * math.sqrt(variance)
    meta = {"variance": float(variance)}
    return (x, meta) if return_metadata else x


def sinusoid_with_dropout_mask(
    batch: int,
    length: int,
    dim: int,
    freq: int = 5,
    mask_density: float = 0.1,
    seed: Optional[int] = 1234,
    return_metadata: bool = False,
) -> Tuple[Tensor, Dict[str, Any], Tensor] | Tuple[Tensor, Tensor]:
    """Sinusoid plus a boolean dropout mask (1 keeps value, 0 masked).

    Returns (series, mask) or (series, meta, mask) if return_metadata.
    Mask independent Bernoulli(mask_density).
    """
    _setup_seed(seed)
    t = torch.linspace(0, 2 * math.pi, length)
    base = torch.sin(freq * t)
    x = base.view(1, length, 1).repeat(batch, 1, dim)
    mask = (torch.rand(batch, length, 1) < (1 - mask_density)).float().repeat(1, 1, dim)
    masked = x * mask
    meta = {"freq": int(freq), "mask_density": float(mask_density)}
    return (masked, meta, mask) if return_metadata else (masked, mask)

__all__ = [
    "sinusoid_mix",
    "polynomial_trend",
    "seasonal_with_trend",
    "step_changes",
    "impulse_train",
    "autoregressive_lagged",
    "correlated_multivariate",
    "piecewise_frequency_shift",
    "noise_only",
    "sinusoid_with_dropout_mask",
]
