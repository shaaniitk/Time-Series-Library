"""Output / Bayesian / Uncertainty related loss tests migrated from monolith."""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.extended]


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(123)
    return torch.randn(3, 12, 4, generator=g), torch.randn(3, 12, 4, generator=g)


def test_bayesian_mse_optional_kl_and_uncertainty_weights() -> None:
    try:
        from layers.modular.registry import create_component
    except Exception:  # pragma: no cover
        pytest.skip("registry not available")

    loss_fn = create_component(
        "loss",
        "bayesian_mse",
        {"kl_weight": 1e-5, "uncertainty_weight": 0.1, "reduction": "mean"},
    )
    if loss_fn is None:
        pytest.skip("bayesian_mse not registered")

    pred, target = _pair()
    base = float(loss_fn(pred, target))
    assert base >= 0

    # If compute_loss supports model; simulate optional existence
    if hasattr(loss_fn, "compute_loss"):
        class _MockModel:
            def kl_divergence(self) -> torch.Tensor:  # type: ignore[override]
                return torch.tensor(0.01)

        with_model = float(loss_fn.compute_loss(pred, target, model=_MockModel()))  # type: ignore[attr-defined]
        # Allow equality if KL tiny, but it should not be negative
        assert with_model >= 0


def test_uncertainty_calibration_relative_behaviour() -> None:
    try:
        from layers.modular.losses.adaptive_bayesian_losses import UncertaintyCalibrationLoss
    except Exception:  # pragma: no cover
        pytest.skip("adaptive bayesian losses unavailable")

    pred, target = _pair()
    loss_fn = UncertaintyCalibrationLoss(calibration_weight=1.0)
    # High error sample
    pred_bad = target + 2.0 * torch.randn_like(target)
    low_u = torch.ones_like(pred) * 0.1
    high_u = torch.ones_like(pred) * 2.0
    # Poor calibration: large error, low uncertainty vs high uncertainty scenario
    loss_mis = float(loss_fn(pred_bad, target, low_u))
    loss_cal = float(loss_fn(pred_bad, target, high_u))
    # We expect calibrated (higher self‑reported uncertainty) to not exceed miscalibrated loss dramatically
    assert loss_cal <= loss_mis * 1.2

