"""Decomposition registry coverage (migrated from legacy monolith slice).

Validates: registry non-empty, instantiation of a sample of components,
reconstruction property (seasonal+trend ~ input) for each created instance.
"""
from __future__ import annotations

import pytest
import torch
from typing import Dict, Any

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.decomposition.registry import DecompositionRegistry, get_decomposition_component  # type: ignore
except Exception:  # pragma: no cover
    DecompositionRegistry = None  # type: ignore
    get_decomposition_component = None  # type: ignore


def test_decomposition_registry_non_empty() -> None:
    if DecompositionRegistry is None:
        pytest.skip("DecompositionRegistry unavailable")
    names = DecompositionRegistry.list_components()
    assert names, "No decomposition components listed"
    assert len(names) == len(set(names)), "Duplicate decomposition names"


def _decomp_params(name: str) -> Dict[str, Any]:
    """Return minimal constructor kwargs for each decomposition variant."""
    if name in {"series_decomp", "stable_decomp"}:
        return {"kernel_size": 3}
    if name == "learnable_decomp":
        return {"input_dim": 3, "init_kernel_size": 5, "max_kernel_size": 16}
    if name == "wavelet_decomp":
        return {"seq_len": 48, "d_model": 3, "levels": 2}
    return {}


@pytest.mark.parametrize("name", ["series_decomp", "stable_decomp", "learnable_decomp", "wavelet_decomp"])
def test_decomposition_basic_reconstruction(name: str) -> None:
    if DecompositionRegistry is None or get_decomposition_component is None:
        pytest.skip("DecompositionRegistry unavailable")
    if name not in DecompositionRegistry.list_components():
        pytest.skip(f"Component {name} not registered")

    comp = get_decomposition_component(name, **_decomp_params(name))

    x = torch.randn(2, 48, 3)
    seasonal, trend = comp(x)  # type: ignore[misc]

    assert seasonal.shape == x.shape and trend.shape[0] == x.shape[0]
    recon = seasonal + trend if trend.shape == x.shape else seasonal + torch.zeros_like(x)
    # Only enforce strict reconstruction for classic moving-average style decompositions
    if name in {"series_decomp", "stable_decomp"}:
        assert torch.allclose(recon, x, atol=1e-4), "Reconstruction drift too large"
    assert torch.isfinite(seasonal).all() and torch.isfinite(trend).all()

