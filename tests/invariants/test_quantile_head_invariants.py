"""Quantile forecasting head invariants.

Covers:
1. Monotonicity: q_i <= q_{i+1} for all adjacent quantiles (no violations).
2. Coverage: empirical coverage of intervals vs nominal within tolerance (synthetic self-sampled data baseline).
"""
from __future__ import annotations

import torch
import pytest

from .thresholds import get_threshold


def _make_quantile_head():  # returns head or skips
    try:
        from utils.modular_components.implementations.outputs import QuantileForecastingHead, OutputConfig  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("Quantile head unavailable")
    cfg = OutputConfig(d_model=32, output_dim=2, horizon=6, dropout=0.0)
    quantiles = [0.1, 0.5, 0.9]
    head = QuantileForecastingHead(cfg, quantiles)
    head.eval()
    return head, quantiles


def _hidden(batch: int = 24, seq: int = 40, d_model: int = 32):
    return torch.randn(batch, seq, d_model)


@pytest.mark.invariant
def test_quantile_monotonicity():
    head, qs = _make_quantile_head()
    hidden = _hidden()
    with torch.no_grad():
        out = head(hidden)
    q_preds = out["quantiles"]  # [B,H,D,Q]
    diffs = q_preds.diff(dim=-1)
    violations = (diffs < -1e-8).float().sum().item()
    frac = violations / diffs.numel()
    assert frac <= get_threshold("quantile_monotonic_violation_max_frac"), frac


@pytest.mark.invariant
def test_quantile_interval_coverage():
    head, qs = _make_quantile_head()
    hidden = _hidden(batch=64)
    with torch.no_grad():
        out = head(hidden)
    q_preds = out["quantiles"]  # [B,H,D,Q]
    # Construct synthetic observations with controlled nominal coverage.
    # We build y so that with probability nominal it is uniform inside [q10, q90]
    # and otherwise placed just outside the interval.
    lower = q_preds[..., 0]
    upper = q_preds[..., -1]
    nominal = 0.9 - 0.1  # 0.8
    bern = torch.bernoulli(torch.full_like(lower, nominal))
    inside_samples = lower + (upper - lower) * torch.rand_like(lower)
    # Outside samples: extend interval width by 10% beyond upper (or below lower)
    width = (upper - lower)
    outside_shift = 0.1 * width
    outside_samples = torch.where(torch.rand_like(lower) < 0.5, lower - outside_shift, upper + outside_shift)
    y = bern * inside_samples + (1 - bern) * outside_samples
    max_err = get_threshold("quantile_coverage_abs_err_max")
    # Evaluate coverage for (0.1,0.9) interval only (baseline)
    inside = ((y >= lower) & (y <= upper)).float().mean().item()
    assert abs(inside - nominal) <= max_err, (inside, nominal)

__all__ = []
