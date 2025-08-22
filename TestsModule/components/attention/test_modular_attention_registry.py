"""Modular attention registry structure tests.

Ensures unified registry lists are non-empty and deterministic and that
we can instantiate a subset of registered components via unified registry.
"""
from __future__ import annotations

import pytest
import torch

# Ensure registration side-effect
import layers.modular.core.register_components  # noqa: F401
from layers.modular.core import unified_registry, ComponentFamily
from layers.modular.attention.registry import get_attention_component

pytestmark = [pytest.mark.extended]


def test_unified_attention_registry_non_empty():
    names = unified_registry.list(ComponentFamily.ATTENTION)[ComponentFamily.ATTENTION.value]
    assert isinstance(names, list) and len(names) > 0
    # deterministic snapshot (same call twice)
    assert names == unified_registry.list(ComponentFamily.ATTENTION)[ComponentFamily.ATTENTION.value]


@pytest.mark.parametrize("candidate", [
    "fourier_attention",
    "wavelet_attention",
    "enhanced_autocorrelation",
    "bayesian_attention",
])
def test_unified_attention_create_and_forward(candidate: str) -> None:
    names = set(unified_registry.list(ComponentFamily.ATTENTION)[ComponentFamily.ATTENTION.value])
    if candidate not in names:
        pytest.skip(f"{candidate} not registered")

    comp = get_attention_component(candidate, d_model=32, n_heads=2)
    q = torch.randn(2, 12, 32, requires_grad=True)
    out, weights = comp(q, q, q)
    assert out.shape == q.shape
    out.mean().backward()
    assert q.grad is not None
    if isinstance(weights, torch.Tensor):
        assert torch.isfinite(weights).all()
