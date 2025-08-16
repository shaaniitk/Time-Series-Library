"""Snapshot test for decomposition metrics (Phase 1 extension)."""
from __future__ import annotations

import pytest

from . import metrics as M
from .snapshot import save_or_compare
from tests.helpers import time_series_generators as gen


def _get_decomposition_impl():  # reuse logic
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


def test_decomposition_metric_snapshot():
    decomp = _get_decomposition_impl()
    x = gen.seasonal_with_trend(batch=1, length=128, dim=1)
    seasonal, trend = decomp(x)  # type: ignore
    metrics = {
        "reconstruction_rel_err": M.l2_relative_error(seasonal + trend, x),
        "trend_hf_energy": M.high_frequency_energy(trend),
    }
    snap = save_or_compare("decomposition", "default_kernel25", metrics, rel_tol=0.25)
    if snap.diffs:
        diffs_str = ", ".join(f"{k}:{v:.3f}" for k, v in snap.diffs.items())
        pytest.fail(f"Decomposition metrics drift exceeded tolerance: {diffs_str}")
