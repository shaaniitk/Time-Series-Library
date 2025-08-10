from __future__ import annotations

"""Quarantined duplicate of active smoke backbone test.

Reason: Original file duplicated into quarantine; kept only as reference while migration proceeds.
Not part of normal execution.
Note: Filename altered to prevent import collision with active smoke test.
"""

import pytest
import torch

__deprecated_duplicate__ = True  # renamed file originally 'test_smoke_backbone.py'

# Placeholder import path; adjust once backbone factory available
try:
    from utils.modular_components.implementations.feedforward import StandardFeedForward as _Backbone
except Exception:  # pragma: no cover - early scaffold
    _Backbone = None  # type: ignore

@pytest.mark.skip(reason="Quarantined duplicate; use active smoke test instead")  # also implicitly quarantine
def test_backbone_minimal_forward(device: torch.device) -> None:
    if _Backbone is None:
        pytest.skip("Backbone implementation not available yet")
    model = _Backbone(d_model=16, hidden_factor=2, dropout=0.0)  # type: ignore[arg-type]
    x = torch.randn(2, 8, 16, device=device)
    y = model(x)
    assert y.shape == x.shape
