"""Sampling + output head component coverage (migrated from legacy unittest)."""
from __future__ import annotations

import pytest
import torch

from TestsModule.utils import make_sampling, make_output_head, random_series

pytestmark = [pytest.mark.extended]


class _MockModel(torch.nn.Module):
    def __init__(self, c_out: int, seq_len: int, batch_size: int):
        super().__init__()
        self.c_out = c_out
        self.seq_len = seq_len
        self.batch = batch_size

    def forward(self, *args):  # type: ignore[override]
        return torch.randn(self.batch, self.seq_len, self.c_out)


@pytest.mark.parametrize("name", ["deterministic", "bayesian", "dropout"])
def test_sampling_components(name: str) -> None:
    model = _MockModel(c_out=5, seq_len=48, batch_size=2)
    params = {"n_samples": 2} if name != "deterministic" else {}
    sampler = make_sampling(name, **params)
    result = sampler(model.forward, None, None, None, None)  # type: ignore[misc]
    assert "prediction" in result
    assert result["prediction"].shape == (2, 48, 5)


@pytest.mark.parametrize("name", ["standard", "quantile"])
def test_output_head_components(name: str) -> None:
    d_model = 16
    c_out = 7
    x = random_series(2, 48, d_model)
    params = {"num_quantiles": 3} if name == "quantile" else {}
    head = make_output_head(name, d_model=d_model, c_out=c_out, **params)
    out = head(x)  # type: ignore[misc]
    expected_dim = c_out * 3 if name == "quantile" else c_out
    assert out.shape == (2, 48, expected_dim)
