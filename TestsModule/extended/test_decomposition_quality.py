"""Extended decomposition quality / statistical property tests."""
from __future__ import annotations

import pytest
import torch

from TestsModule.utils import make_decomposition, random_series

pytestmark = [pytest.mark.extended]


@pytest.mark.parametrize("name", ["series_decomp", "stable_decomp", "learnable_decomp", "wavelet_decomp"])
def test_trend_mean_closer_to_input_mean(name: str) -> None:
    series = random_series(2, 96, 8)
    comp = make_decomposition(name, d_model=8, seq_len=96)
    seasonal, trend = comp(series)  # type: ignore[misc]
    assert seasonal.shape == series.shape and trend.shape[0] == series.shape[0]
    if name in {"series_decomp", "stable_decomp"}:
        mean_diff_input = (series.mean(dim=1) - trend.mean(dim=1)).abs().mean()
        mean_diff_seasonal = (series.mean(dim=1) - seasonal.mean(dim=1)).abs().mean()
        assert mean_diff_input < mean_diff_seasonal + 1e-6
    assert torch.isfinite(seasonal).all() and torch.isfinite(trend).all()
