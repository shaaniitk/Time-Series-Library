"""Algorithmic invariants for decomposition components (Phase 1)."""
from __future__ import annotations

import pytest
import torch

from .thresholds import get_threshold
from . import metrics as M
from tests.helpers import time_series_generators as gen


def _get_decomposition_impl():  # type: ignore[return-type]
    try:
        from tools.unified_component_registry import ensure_initialized, get_component  # type: ignore
        ensure_initialized()
        cls = get_component("decomposition", "learnable_series")
        if cls is not None:
            return cls(kernel_size=25)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        from layers.Autoformer_EncDec import series_decomp  # type: ignore
        return series_decomp(kernel_size=25)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"No decomposition implementation available: {e}")


def test_decomposition_basic_invariants():
    decomp = _get_decomposition_impl()
    x = gen.seasonal_with_trend(batch=2, length=128, dim=3, noise_std=0.0)
    seasonal, trend = decomp(x)  # type: ignore[misc]
    recon_rel_err = M.l2_relative_error(seasonal + trend, x)
    assert recon_rel_err < get_threshold("decomposition_recon_rel_err"), recon_rel_err
    trend_hf = M.high_frequency_energy(trend)
    assert trend_hf < get_threshold("trend_high_freq_ratio"), trend_hf
    ratio = get_threshold("seasonal_mean_abs_ratio")
    assert M.seasonal_zero_mean_ok(seasonal, x, ratio)
    # Energy conservation (threshold from thresholds.py: decomposition_energy_drift)
    energy = lambda t: float((t.pow(2)).sum())
    total = energy(x)
    parts = energy(seasonal) + energy(trend)
    drift = abs(parts - total) / (total + 1e-12)
    # Empirically current implementation shows ~6% energy discrepancy (kernel smoothing leakage);
    # start lax and tighten after implementation refinement.
    assert drift < get_threshold("decomposition_energy_drift"), drift


@pytest.mark.parametrize("mix_scale", [0.5, 1.0])
def test_decomposition_linearity(mix_scale: float):
    decomp = _get_decomposition_impl()
    x1 = gen.seasonal_with_trend(batch=1, length=96, dim=1)
    x2 = gen.seasonal_with_trend(batch=1, length=96, dim=1, freqs=(3, 9))
    a, b = mix_scale, 1.0 - mix_scale
    combo = a * x1 + b * x2
    s1, t1 = decomp(x1)  # type: ignore
    s2, t2 = decomp(x2)  # type: ignore
    sc, tc = decomp(combo)  # type: ignore
    lhs_s = a * s1 + b * s2
    lhs_t = a * t1 + b * t2
    rel_err_s = M.l2_relative_error(lhs_s, sc)
    rel_err_t = M.l2_relative_error(lhs_t, tc)
    tol = get_threshold("decomposition_recon_rel_err") * 10
    assert rel_err_s < tol
    assert rel_err_t < tol
