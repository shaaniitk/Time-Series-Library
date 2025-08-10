"""Smoke tests for key attention components.

Covers lightweight instantiation + forward pass to ensure registry wiring works.
Focus: fast feedback (no large tensors or performance assertions).
"""
from __future__ import annotations

import pytest
import torch

try:
    from layers.modular.attention.registry import AttentionRegistry  # type: ignore
except Exception:  # pragma: no cover - flexibility during refactor
    AttentionRegistry = None  # type: ignore

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
    if AttentionRegistry is None:
        pytest.skip("AttentionRegistry unavailable")

    registry = AttentionRegistry()
    attn = registry.create(name, d_model=32, n_heads=2, **kwargs)

    q = torch.randn(2, 8, 32, requires_grad=True)
    out, weights = attn(q, q, q)

    assert out.shape == q.shape, "Output shape mismatch"
    assert torch.isfinite(out).all(), "Non-finite outputs"

    loss = out.sum()
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all(), "No/invalid gradients"

    # Optional lightweight sanity on weights object presence
    assert weights is not None
