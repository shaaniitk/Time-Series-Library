"""Synthetic time-series data generation helpers for TestsModule.

Functions here should be deterministic given a seed and avoid large tensors
for smoke/core layers.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch

DEFAULT_SEED = 1337

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_series(length: int = 64, channels: int = 1, noise: float = 0.05, trend: bool = True, season_period: int | None = 12) -> torch.Tensor:
    """Create a synthetic series tensor of shape (length, channels)."""
    set_seed()
    t = np.arange(length)
    base = np.zeros((length, channels), dtype=np.float32)
    if trend:
        base += (t[:, None] / max(length - 1, 1)) * 0.5
    if season_period and season_period < length:
        season = 0.3 * np.sin(2 * np.pi * t / season_period)
        base += season[:, None]
    noise_arr = noise * np.random.randn(length, channels).astype(np.float32)
    return torch.from_numpy(base + noise_arr)
