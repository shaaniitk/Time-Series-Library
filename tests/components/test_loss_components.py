"""Smoke tests for core loss components via LossRegistry.

Covers instantiation + scalar output + gradient path.
"""
from __future__ import annotations

import pytest
import torch

try:
    from layers.modular.losses.registry import LossRegistry  # type: ignore
except Exception:  # pragma: no cover
    LossRegistry = None  # type: ignore

LOSS_SMOKE = [
    ("mse", {}),
    ("mae", {}),
    ("mape", {}),
]

@pytest.mark.smoke
@pytest.mark.parametrize("name,kwargs", LOSS_SMOKE)
def test_loss_smoke(name: str, kwargs: dict) -> None:
    """Run a tiny loss computation and backward check."""
    if LossRegistry is None:
        pytest.skip("LossRegistry unavailable")

    registry = LossRegistry()
    loss_fn = registry.create(name, **kwargs)

    pred = torch.randn(4, 6, 3, requires_grad=True)
    target = torch.randn(4, 6, 3)
    if name in {"mape"}:
        target = target.abs() + 0.1

    loss = loss_fn(pred, target)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "Loss must be scalar tensor"
    assert torch.isfinite(loss), "Loss not finite"

    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
