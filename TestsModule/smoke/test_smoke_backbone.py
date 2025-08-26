"""Smoke test for backbone construction & forward.
Marks: smoke
"""
from __future__ import annotations
import pytest
import torch

try:
    from layers.modular.core.registry import unified_registry, ComponentFamily
except Exception:  # pragma: no cover
    unified_registry = None  # type: ignore

@pytest.mark.smoke
def test_backbone_minimal_forward(device: torch.device) -> None:
    if unified_registry is None:
        pytest.skip("Backbone implementation not available yet")
    # Use standard FFN from unified registry as a simple backbone-like block
    model = unified_registry.create(ComponentFamily.FEEDFORWARD, 'standard_ffn', d_model=16, hidden_factor=2, dropout=0.0)
    x = torch.randn(2, 8, 16, device=device)
    y = model(x)
    assert y.shape == x.shape
