from __future__ import annotations

"""Bayesian and uncertainty-related losses.

Covers basic execution for bayesian, bayesian_quantile, and uncertainty_calibration
losses using tiny tensors and minimal scaffolding. Marked extended.
"""

from typing import Any, List

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.losses.registry import LossRegistry  # type: ignore
except Exception:  # pragma: no cover
    LossRegistry = None  # type: ignore


class _DummyModel:
    def __init__(self) -> None:
        self._quantiles = None
    def set_quantile_levels(self, q):
        self._quantiles = q
    def get_quantile_levels(self):
        return self._quantiles


def test_bayesian_mse_like_loss_minimal() -> None:
    if LossRegistry is None:
        pytest.skip("LossRegistry unavailable")
    loss_factory = LossRegistry.get("bayesian")
    loss = loss_factory(kl_weight=1e-6)
    model = _DummyModel()
    x = torch.randn(2, 8, 3)
    y = torch.randn(2, 8, 3)
    out = loss(model, x, y)
    # BayesianLoss returns a dict with 'total_loss'
    assert isinstance(out, dict) and 'total_loss' in out
    v = out['total_loss']
    assert isinstance(v, torch.Tensor) and torch.isfinite(v).all()


def test_bayesian_quantile_minimal() -> None:
    if LossRegistry is None:
        pytest.skip("LossRegistry unavailable")
    loss_cls = LossRegistry.get("bayesian_quantile")
    q = [0.1, 0.5, 0.9]
    model = _DummyModel()
    loss = loss_cls(quantiles=q, kl_weight=1e-6)

    B, L, T = 2, 6, 4
    preds = torch.randn(B, L, T, len(q))
    target = torch.randn(B, L, T)
    out = loss(model, preds, target)
    assert isinstance(out, dict) and 'total_loss' in out
    v = out['total_loss']
    assert isinstance(v, torch.Tensor) and torch.isfinite(v).all()


def test_uncertainty_calibration_minimal() -> None:
    if LossRegistry is None:
        pytest.skip("LossRegistry unavailable")
    loss_cls = LossRegistry.get("uncertainty_calibration")
    loss = loss_cls(calibration_weight=1.0)
    x = torch.randn(2, 8, 3)
    y = torch.randn(2, 8, 3)
    sigma = torch.abs(torch.randn(2, 8, 3)) + 0.1
    v = loss(x, y, sigma)
    assert isinstance(v, torch.Tensor) and torch.isfinite(v).all()
