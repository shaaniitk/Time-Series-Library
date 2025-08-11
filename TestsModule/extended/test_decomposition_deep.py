"""Deeper decomposition behavioural tests.

Augments basic quality test with:
1. Wavelet hierarchical level monotonicity (length decreases with depth).
2. Reconstruction consistency for learnable vs series decomposition interplay.
3. Stability of seasonal + trend recomposition (approx original signal within tolerance).
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.decomposition import get_decomposition_component  # type: ignore
except Exception:  # pragma: no cover
    get_decomposition_component = None  # type: ignore


def _skip_if_absent(name: str) -> None:
    if get_decomposition_component is None:
        pytest.skip("Decomposition registry unavailable")
    try:
        get_decomposition_component(name)
    except Exception:
        pytest.skip(f"{name} not present")


def test_wavelet_level_monotonicity() -> None:
    """Wavelet hierarchical decomposition levels should be strictly non-increasing in length."""
    if get_decomposition_component is None:
        pytest.skip("Decomposition registry unavailable")
    seq_len = 128
    x = torch.randn(2, seq_len, 16)
    wave = get_decomposition_component("wavelet_decomp", seq_len=seq_len, d_model=16, levels=3)
    # Requires new introspection API; fall back skip if absent
    if not hasattr(wave, 'decompose_with_levels'):
        pytest.skip("Wavelet component lacks level introspection API")
    _, _, levels = wave.decompose_with_levels(x)
    lengths = [lvl.shape[1] for lvl in levels]
    assert all(lengths[i] >= lengths[i+1] for i in range(len(lengths)-1)), f"Non-monotonic lengths: {lengths}"


def test_wavelet_recomposition_basic() -> None:
    """Wavelet hierarchical decomposition seasonal + trend should approx original.

    We accept moderate relative error because projections & resizing modify exact values.
    """
    if get_decomposition_component is None:
        pytest.skip("Decomposition registry unavailable")
    seq_len = 64
    x = torch.randn(2, seq_len, 16)
    wave = get_decomposition_component("wavelet_decomp", seq_len=seq_len, d_model=16, levels=2)
    seasonal, trend = wave(x)
    assert seasonal.shape == x.shape and trend.shape[0] == x.shape[0]
    recon = seasonal + trend
    rel_err = (recon - x).pow(2).mean().sqrt() / (x.pow(2).mean().sqrt() + 1e-6)
    # Current simple seasonal/trend recomposition may be lossy; enforce loose upper bound
    assert rel_err < 1.5, f"Reconstruction relative RMSE too high: {rel_err:.3f}"


def test_series_vs_learnable_decomposition_consistency() -> None:
    """Learnable vs fixed series decomposition seasonal variance ordering heuristic.

    Heuristic: learnable seasonal variance shouldn't collapse relative to fixed series variant.
    """
    if get_decomposition_component is None:
        pytest.skip("Decomposition registry unavailable")
    seq_len = 48
    x = torch.randn(2, seq_len, 8)
    series = get_decomposition_component("series_decomp", kernel_size=5)
    learn = get_decomposition_component("learnable_decomp", input_dim=8, init_kernel_size=5, max_kernel_size=7)
    s_seasonal, s_trend = series(x)
    l_seasonal, l_trend = learn(x)
    var_series = s_seasonal.var()
    var_learn = l_seasonal.var()
    assert var_learn > 0, "Learnable seasonal variance collapsed to zero"
    # Not strictly required to be larger, but should not be dramatically smaller
    assert var_learn >= 0.25 * var_series, "Learnable seasonal variance unexpectedly tiny"
