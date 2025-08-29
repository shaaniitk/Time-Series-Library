"""Gradient finite-difference sanity for decomposition (tiny shape)."""
from __future__ import annotations

import pytest
import torch

from .gradient_check import finite_difference_check
from tests.helpers import time_series_generators as gen


@pytest.mark.gradcheck
def test_decomposition_gradient_fd():
    try:
        from tools.unified_component_registry import ensure_initialized, get_component  # type: ignore
        ensure_initialized()
        cls = get_component("decomposition", "learnable_series")
        if cls is None:
            pytest.skip("No registered decomposition component")
        decomp = cls(kernel_size=7)  # type: ignore[arg-type]
    except Exception:
        try:
            from layers.Autoformer_EncDec import series_decomp  # type: ignore
            decomp = series_decomp(kernel_size=7)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"No decomposition implementation available: {e}")

    x = gen.seasonal_with_trend(batch=1, length=16, dim=1)
    x = x.detach().requires_grad_(True)

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            s, t = self.mod(inp)  # type: ignore
            return s + t

    wrap = Wrapper(decomp)
    rel_diff, atol = finite_difference_check(wrap, x)
    assert rel_diff < atol, (rel_diff, atol)

__all__ = []
