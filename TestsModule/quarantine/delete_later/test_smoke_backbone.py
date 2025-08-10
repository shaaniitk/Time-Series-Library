"""Smoke test for backbone construction & forward.
Marks: smoke
"""
from __future__ import annotations
import pytest
import torch

# Placeholder import path; adjust once backbone factory available
try:
    from utils.modular_components.implementations.feedforward import StandardFeedForward as _Backbone
except Exception:  # pragma: no cover - early scaffold
    _Backbone = None  # type: ignore

@pytest.mark.smoke
def test_backbone_minimal_forward(device: torch.device) -> None:
    if _Backbone is None:
        pytest.skip("Backbone implementation not available yet")
    model = _Backbone(d_model=16, hidden_factor=2, dropout=0.0)  # type: ignore[arg-type]
    x = torch.randn(2, 8, 16, device=device)
    y = model(x)
    assert y.shape == x.shape
