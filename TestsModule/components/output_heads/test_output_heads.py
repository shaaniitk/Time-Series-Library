from __future__ import annotations

"""Output head component tests.

Validates registry listing and minimal forward shape for standard and quantile
output heads using tiny tensors. Lightweight and marked extended.
"""

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.output_heads.registry import (
        OutputHeadRegistry,
        get_output_head_component,
    )  # type: ignore
except Exception:  # pragma: no cover
    OutputHeadRegistry = None  # type: ignore
    get_output_head_component = None  # type: ignore


def _has(name: str) -> bool:
    if OutputHeadRegistry is None:
        return False
    try:
        return name in OutputHeadRegistry.list_components()
    except Exception:
        return False


def test_output_head_registry_non_empty() -> None:
    if OutputHeadRegistry is None:
        pytest.skip("OutputHeadRegistry unavailable")
    names = OutputHeadRegistry.list_components()
    assert isinstance(names, list) and names
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "name,kwargs,expected_last_dim",
    [
        ("standard", {"d_model": 16, "c_out": 4}, 4),
        ("quantile", {"d_model": 16, "c_out": 3, "num_quantiles": 5}, 15),
    ],
)
def test_output_head_forward_shape(name: str, kwargs: dict[str, Any], expected_last_dim: int) -> None:
    if OutputHeadRegistry is None or get_output_head_component is None:
        pytest.skip("OutputHeadRegistry unavailable")
    if not _has(name):
        pytest.skip(f"{name} output head not registered")

    head = get_output_head_component(name, **kwargs)
    x = torch.randn(2, 10, 16, requires_grad=True)
    y = head(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[:2] == x.shape[:2]
    assert y.shape[-1] == expected_last_dim
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
