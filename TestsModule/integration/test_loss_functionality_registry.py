"""Registry / meta tests for loss components (migrated)."""
from __future__ import annotations

import importlib
from typing import Iterable

import pytest
import torch

pytestmark = [pytest.mark.extended]


ESSENTIAL_LOSSES: tuple[str, ...] = (
    "mse",
    "mae",
    "quantile_loss",
)


def _has_registry() -> bool:
    try:
        import layers.modular.registry  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


@pytest.mark.smoke
def test_loss_registry_contains_essentials() -> None:
    if not _has_registry():
        pytest.skip("registry unavailable")
    from layers.modular.registry import create_component

    missing = [name for name in ESSENTIAL_LOSSES if create_component("loss", name, {"reduction": "mean"}) is None]
    assert not missing, f"Missing essential losses: {missing}"


def test_registered_losses_basic_forward() -> None:
    if not _has_registry():
        pytest.skip("registry unavailable")
    from layers.modular.registry import list_components, create_component

    registered: Iterable[str] = list_components("loss")  # type: ignore[assignment]
    # Only sample a subset for speed
    sample = [n for n in registered if any(key in n for key in ("mse", "mae", "quantile"))][:5]
    if not sample:
        pytest.skip("no sample losses found in registry")
    pred = torch.randn(2, 6, 3)
    target = torch.randn(2, 6, 3)
    for name in sample:
        cfg = {"reduction": "mean"}
        if "quantile" in name:
            cfg["quantiles"] = [0.5]
        loss_fn = create_component("loss", name, cfg)
        if loss_fn is None:
            continue
        out = loss_fn(pred.unsqueeze(-1) if "quantile" in name else pred, target)
        assert torch.isfinite(out)
        assert out.dim() == 0

