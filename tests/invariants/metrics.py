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


__all__ = [
    "l2_relative_error",
    "high_frequency_energy",
    "dominant_frequency_bin",
    "seasonal_zero_mean_ok",
]
