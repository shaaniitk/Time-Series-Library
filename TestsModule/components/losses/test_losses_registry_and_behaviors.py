from __future__ import annotations

"""Loss component tests.

Covers registry presence, aliasing, and minimal behaviors for quantile/pinball
and several advanced/adaptive losses with tiny tensors. Marked extended.
"""

from typing import Any, List

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.losses.registry import LossRegistry, get_loss_component  # type: ignore
except Exception:  # pragma: no cover
    LossRegistry = None  # type: ignore
    get_loss_component = None  # type: ignore


def test_loss_registry_non_empty_and_aliases() -> None:
    if LossRegistry is None:
        pytest.skip("LossRegistry unavailable")
    names = set(LossRegistry.list_available())
    assert names, "loss registry should not be empty"
    # Aliases present
    assert "quantile" in names and "pinball" in names


@pytest.mark.parametrize("quantiles", [[0.1, 0.5, 0.9], [0.5]])
def test_pinball_quantile_loss_shapes_and_nonneg(quantiles: List[float]) -> None:
    if get_loss_component is None:
        pytest.skip("Loss components unavailable")

    # Ensure monotonic sorted quantiles for stable behavior
    q = sorted(quantiles)
    loss, mult = get_loss_component("quantile", quantiles=q)
    assert mult == len(q)

    B, L, T = 2, 6, 4
    preds = torch.randn(B, L, T, len(q))
    target = torch.randn(B, L, T)
    val = loss(preds, target)
    assert isinstance(val, torch.Tensor)
    assert torch.isfinite(val).all()
    assert val.item() >= 0.0

    # Also test flattened [B, L, T*Q]
    preds_flat = preds.reshape(B, L, T * len(q))
    val2 = loss(preds_flat, target)
    assert torch.isfinite(val2).all() and val2.item() >= 0.0


@pytest.mark.parametrize("name,kwargs", [
    ("mse", {}),
    ("mae", {}),
    ("huber", {}),
])
def test_standard_losses_basic(name: str, kwargs: dict[str, Any]) -> None:
    if LossRegistry is None or get_loss_component is None:
        pytest.skip("LossRegistry unavailable")
    # Use registry create alias through get_loss_component wrapper
    loss, mult = get_loss_component(name, **kwargs)
    assert mult == getattr(loss, "output_dim_multiplier", 1)

    x = torch.randn(2, 8, 3, requires_grad=True)
    y = torch.randn(2, 8, 3)
    v = loss(x, y)
    assert isinstance(v, torch.Tensor) and torch.isfinite(v).all()
    v.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("name,kwargs", [
    ("mape", {}),
    ("smape", {}),
    ("mase", {"freq": 1}),
    ("ps_loss", {"pred_len": 8}),
    ("focal", {}),
])
def test_advanced_losses_minimal(name: str, kwargs: dict[str, Any]) -> None:
    if LossRegistry is None:
        pytest.skip("LossRegistry unavailable")
    # Some advanced losses are classes directly; use registry.create path
    loss_cls = LossRegistry.get(name)
    loss = loss_cls(**kwargs) if callable(loss_cls) else loss_cls

    x = torch.abs(torch.randn(2, 8, 3)) + 0.1 if name in {"mape", "smape"} else torch.randn(2, 8, 3)
    y = torch.abs(torch.randn(2, 8, 3)) + 0.1 if name in {"mape", "smape"} else torch.randn(2, 8, 3)
    v = loss(x, y) if name != "ps_loss" else loss(x, y)
    assert isinstance(v, torch.Tensor) and torch.isfinite(v).all()


@pytest.mark.parametrize("name,kwargs", [
    ("adaptive_autoformer", {}),
    ("frequency_aware", {}),
])
def test_adaptive_losses_minimal(name: str, kwargs: dict[str, Any]) -> None:
    if LossRegistry is None:
        pytest.skip("LossRegistry unavailable")
    loss_cls = LossRegistry.get(name)
    loss = loss_cls(**kwargs) if callable(loss_cls) else loss_cls

    x = torch.randn(2, 16, 4)
    y = torch.randn(2, 16, 4)
    v = loss(x, y)
    assert isinstance(v, torch.Tensor) and torch.isfinite(v).all()
