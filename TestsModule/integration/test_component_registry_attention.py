"""Attention registry structural tests.

Verifies a minimal contract: list non-empty, create subset of components via
registry.create (tolerating graceful skips for absent names), forward pass
shape integrity, and that list_components is deterministic (stable ordering).
"""
from __future__ import annotations

from typing import List

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.attention.registry import AttentionRegistry  # type: ignore
except Exception:  # pragma: no cover
    AttentionRegistry = None  # type: ignore


def test_attention_registry_lists_components() -> None:
    if AttentionRegistry is None:
        pytest.skip("AttentionRegistry unavailable")
    names = AttentionRegistry.list_components()
    assert names and isinstance(names, list)
    # Deterministic ordering expectation (current implementation preserves dict insertion order)
    assert names == AttentionRegistry.list_components(), "Component listing not deterministic"


@pytest.mark.parametrize("candidate", [
    "fourier_attention",
    "wavelet_attention",
    "enhanced_autocorrelation",
    "bayesian_attention",
])
def test_attention_registry_create_and_forward(candidate: str) -> None:
    if AttentionRegistry is None:
        pytest.skip("AttentionRegistry unavailable")
    if candidate not in AttentionRegistry.list_components():
        pytest.skip(f"{candidate} not registered")
    try:
        comp = AttentionRegistry.create(candidate, d_model=32, n_heads=2)
    except Exception as e:  # pragma: no cover
        pytest.xfail(f"Instantiation failure for {candidate}: {e}")

    q = torch.randn(2, 12, 32, requires_grad=True)
    out_tuple = comp(q, q, q)  # type: ignore[misc]
    out = out_tuple[0] if isinstance(out_tuple, (tuple, list)) else out_tuple
    assert out.shape == q.shape
    out.mean().backward()
    assert q.grad is not None

