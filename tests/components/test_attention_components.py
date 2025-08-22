"""Smoke tests for key attention components.

Covers lightweight instantiation + forward pass to ensure registry wiring works.
Focus: fast feedback (no large tensors or performance assertions).
"""
from __future__ import annotations

import pytest
import torch

from layers.modular.core import unified_registry, ComponentFamily, get_attention_component  # type: ignore
import layers.modular.core.register_components  # noqa: F401

ATTENTION_SMOKE = [
    ("fourier_attention", {}),
    ("linear_attention", {}),
    ("enhanced_autocorrelation", {"adaptive_k": True, "multi_scale": True}),
]

@pytest.mark.smoke
@pytest.mark.parametrize("name,kwargs", ATTENTION_SMOKE)
def test_attention_smoke_forward(name: str, kwargs: dict) -> None:
    """Instantiate attention module and run a tiny forward pass.

    Ensures: creation succeeds, output shape matches input, gradients flow.
    """
    names = set(unified_registry.list(ComponentFamily.ATTENTION)['attention'])
    # include aliases
    for n in list(names):
        try:
            info = unified_registry.describe(ComponentFamily.ATTENTION, n)
            names.update(info.get('aliases', []))
        except Exception:
            pass
    if name not in names:
        pytest.skip(f"{name} not registered")
    attn = get_attention_component(name, d_model=32, n_heads=2, **kwargs)

    q = torch.randn(2, 8, 32, requires_grad=True)
    out, weights = attn(q, q, q)

    assert out.shape == q.shape, "Output shape mismatch"
    assert torch.isfinite(out).all(), "Non-finite outputs"

    loss = out.sum()
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all(), "No/invalid gradients"

    # Optional: if weights returned, check it’s finite; some modules return None intentionally
    if weights is not None:
        assert torch.isfinite(weights).all()
