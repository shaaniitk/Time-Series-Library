"""Quantile loss behaviour tests migrated from monolith.

Focus: pinball (quantile) loss non‑negativity, ordering benefit, shape &
heterogeneous quantile sets.
"""
from __future__ import annotations

from typing import List

import pytest
import torch

pytestmark = [pytest.mark.extended]


def _base_target(batch: int = 3, seq: int = 12, feat: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn(batch, seq, feat, generator=g)


@pytest.mark.parametrize("quantiles", [[0.1, 0.5, 0.9], [0.5], [0.25, 0.75]])
def test_quantile_loss_basic_and_shape(quantiles: List[float]) -> None:
    try:
        from layers.modular.registry import create_component
    except Exception:  # pragma: no cover
        pytest.skip("registry not available")

    loss_fn = create_component("loss", "quantile_loss", {"quantiles": quantiles, "reduction": "mean"})
    if loss_fn is None:
        pytest.skip("quantile_loss not registered")

    target = _base_target()
    pred = torch.randn(*target.shape, len(quantiles))
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_quantile_ordering_reduces_loss() -> None:
    try:
        from layers.modular.registry import create_component
    except Exception:  # pragma: no cover
        pytest.skip("registry not available")

    qs = [0.1, 0.5, 0.9]
    loss_fn = create_component("loss", "quantile_loss", {"quantiles": qs, "reduction": "mean"})
    if loss_fn is None:
        pytest.skip("quantile loss not registered")

    target = _base_target()
    pred = torch.randn(*target.shape, len(qs))
    unordered = float(loss_fn(pred, target))
    ordered_pred = torch.sort(pred, dim=-1)[0]
    ordered = float(loss_fn(ordered_pred, target))
    # Allow equality (stochastic chance) but ordering should *not* increase loss materially
    assert ordered <= unordered + 1e-6


def test_pinball_asymmetry_reference_values() -> None:
    """Analytical pinball losses for q=0.1 small vector of errors."""
    q = 0.1
    errors = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    pinball = torch.where(errors >= 0, q * errors, (q - 1) * errors)
    # Explicit reference pattern
    # Our direct formulation yields positive losses for both sides; expected magnitudes
    expected = [0.9, 0.45, 0.0, 0.05, 0.1]
    assert torch.allclose(pinball, torch.tensor(expected), atol=1e-6)

