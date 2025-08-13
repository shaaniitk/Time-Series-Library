from __future__ import annotations

"""Decomposition component tests.

Covers listing and basic reconstruction/shape checks for key decomposition variants.
"""

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.decomposition.registry import DecompositionRegistry, get_decomposition_component  # type: ignore
except Exception:  # pragma: no cover
    DecompositionRegistry = None  # type: ignore
    get_decomposition_component = None  # type: ignore


def _has(name: str) -> bool:
    if DecompositionRegistry is None:
        return False
    try:
        return name in DecompositionRegistry.list_components()
    except Exception:
        return False


def test_decomposition_registry_non_empty() -> None:
    if DecompositionRegistry is None:
        pytest.skip("DecompositionRegistry unavailable")
    names = DecompositionRegistry.list_components()
    assert isinstance(names, list) and names
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name,kwargs", [
    ("series_decomp", {"kernel_size": 3}),
    ("stable_decomp", {"kernel_size": 3}),
    ("learnable_decomp", {"input_dim": 4, "init_kernel_size": 5, "max_kernel_size": 8}),
    ("wavelet_decomp", {"seq_len": 40, "d_model": 4, "levels": 2}),
])
def test_decomposition_basic_shapes(name: str, kwargs: dict) -> None:
    if DecompositionRegistry is None or get_decomposition_component is None:
        pytest.skip("DecompositionRegistry unavailable")
    if not _has(name):
        pytest.skip(f"{name} not registered")

    comp = get_decomposition_component(name, **kwargs)
    x = torch.randn(2, 40, 4)
    seasonal, trend = comp(x)  # type: ignore[misc]
    assert seasonal.shape[0] == x.shape[0]
    assert seasonal.shape[-1] == x.shape[-1]
    assert torch.isfinite(seasonal).all() and torch.isfinite(trend).all()

    # For classic moving-average style, seasonal+trend should approximate x closely
    if name in {"series_decomp", "stable_decomp"}:
        recon = seasonal + trend if trend.shape == x.shape else seasonal
        assert torch.allclose(recon, x, atol=1e-4)
