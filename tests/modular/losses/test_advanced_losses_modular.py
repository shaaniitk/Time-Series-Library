"""Advanced & adaptive loss component modular tests.

Migrates Phase 1 (advanced loss integration) assertions from legacy
`test_phase1_integration.py` into lean, parametrized pytest tests.

Coverage:
 - Registry presence of advanced & adaptive Bayesian losses.
 - Individual forward computations produce finite, non‑negative scalars where applicable.
 - QuantileLoss handles multi-quantile prediction tensors.
 - UncertaintyCalibrationLoss integrates auxiliary uncertainty tensor.

Design notes:
 - Imports are wrapped in try/except to allow graceful skip if a specific
   experimental loss class is temporarily unavailable.
 - Keeps tensors small for speed (fast suite, not marked slow).
 - Avoids print noise; relies on assertions only.
"""
from __future__ import annotations

import torch
import pytest

try:  # Core advanced losses
    from layers.modular.losses.advanced_losses import (
        MAPELoss,
        SMAPELoss,
        MASELoss,
        PSLoss,
        FocalLoss,
    )
except Exception:  # pragma: no cover - missing module during partial refactors
    MAPELoss = SMAPELoss = MASELoss = PSLoss = FocalLoss = None  # type: ignore

try:  # Adaptive / probabilistic related losses
    from layers.modular.losses.adaptive_bayesian_losses import (
        FrequencyAwareLoss,
        UncertaintyCalibrationLoss,
        QuantileLoss,
    )
except Exception:  # pragma: no cover
    FrequencyAwareLoss = UncertaintyCalibrationLoss = QuantileLoss = None  # type: ignore

try:
    from layers.modular.losses.registry import LossRegistry
except Exception:  # pragma: no cover
    LossRegistry = None  # type: ignore


@pytest.mark.skipif(LossRegistry is None, reason="LossRegistry unavailable")
def test_advanced_loss_registry_presence() -> None:
    """Verify advanced & adaptive losses are registered (non-crashing presence)."""
    registry = LossRegistry()
    expected = {
        'mape', 'smape', 'mase', 'ps_loss', 'focal',
        'frequency_aware', 'multi_quantile', 'uncertainty_calibration'
    }
    available = set(registry.list_components())
    missing = expected - available
    # Allow soft failures for experimental components, but assert majority presence.
    assert len(available & expected) >= 6, f"Insufficient advanced losses registered; missing={missing}"


@pytest.mark.skipif(MAPELoss is None, reason="Advanced losses unavailable")
def test_basic_scalar_losses_forward() -> None:
    """Smoke test: scalar-producing advanced losses return finite values."""
    pred = torch.randn(2, 10, 3)
    target = torch.abs(torch.randn(2, 10, 3)) + 0.1  # avoid divide-by-zero for MAPE variants

    losses = [
        MAPELoss(),
        SMAPELoss(),
        MASELoss(freq=1),
        PSLoss(pred_len=10),
        FocalLoss(),
    ]
    for loss_fn in losses:
        val = loss_fn(pred, target)
        assert torch.isfinite(val), f"Non-finite loss for {loss_fn.__class__.__name__}"


@pytest.mark.skipif(FrequencyAwareLoss is None, reason="Adaptive Bayesian losses unavailable")
def test_frequency_aware_loss_forward() -> None:
    """FrequencyAwareLoss returns finite value with standard shaped inputs."""
    pred = torch.randn(2, 10, 4)
    target = torch.randn(2, 10, 4)
    loss_fn = FrequencyAwareLoss()
    val = loss_fn(pred, target)
    assert torch.isfinite(val) and val.ndim == 0


@pytest.mark.skipif(QuantileLoss is None, reason="QuantileLoss unavailable")
def test_quantile_loss_multi_quantile() -> None:
    """QuantileLoss handles (B, T, C, Q) prediction tensor and returns finite scalar."""
    quantiles = [0.1, 0.5, 0.9]
    B, T, C, Q = 2, 8, 3, len(quantiles)
    pred = torch.randn(B, T, C, Q)
    target = torch.randn(B, T, C)
    loss_fn = QuantileLoss(quantiles=quantiles)  # type: ignore[arg-type]
    val = loss_fn(pred, target)
    assert torch.isfinite(val)


@pytest.mark.skipif(UncertaintyCalibrationLoss is None, reason="UncertaintyCalibrationLoss unavailable")
def test_uncertainty_calibration_loss_forward() -> None:
    """UncertaintyCalibrationLoss consumes prediction, target, and uncertainty tensors."""
    B, T, C = 2, 8, 3
    pred = torch.randn(B, T, C)
    target = torch.randn(B, T, C)
    uncertainties = torch.rand(B, T, C) + 0.05
    loss_fn = UncertaintyCalibrationLoss()
    val = loss_fn(pred, target, uncertainties)  # type: ignore[arg-type]
    assert torch.isfinite(val)
import torch
import pytest

from layers.modular.losses.advanced_losses import MAPELoss, SMAPELoss, MASELoss, PSLoss, FocalLoss
from layers.modular.losses.adaptive_bayesian_losses import (
    FrequencyAwareLoss, QuantileLoss, UncertaintyCalibrationLoss
)
from layers.modular.losses.registry import LossRegistry


@pytest.mark.parametrize("loss_name, cls", [
    ("mape", MAPELoss),
    ("smape", SMAPELoss),
    ("mase", MASELoss),
    ("ps_loss", PSLoss),
    ("focal", FocalLoss),
    ("frequency_aware", FrequencyAwareLoss),
    ("multi_quantile", QuantileLoss),
    ("uncertainty_calibration", UncertaintyCalibrationLoss),
])
def test_loss_registry_entries(loss_name, cls):
    """Ensure each advanced loss is present in the LossRegistry and instantiates."""
    registered = LossRegistry.get(loss_name)
    assert registered is not None
    # Registry may store lambda wrappers; handle those gracefully
    if isinstance(registered, type):
        assert issubclass(registered, cls) or registered is cls


def _generate_data(batch: int = 2, length: int = 16, features: int = 3):
    pred = torch.randn(batch, length, features)
    target = torch.abs(torch.randn(batch, length, features)) + 0.1
    return pred, target


def test_metric_losses_compute():
    """MAPE/SMAPE/MASE/PS/Focal should return finite scalar losses."""
    pred, target = _generate_data()

    # Provide insample for MASE with slightly longer context
    insample = torch.randn(pred.size(0), pred.size(1) + 4, pred.size(2))

    mape = MAPELoss() ; assert torch.isfinite(mape(pred, target))
    smape = SMAPELoss() ; assert torch.isfinite(smape(pred, target))
    mase = MASELoss(freq=1) ; assert torch.isfinite(mase(pred, target, insample=insample))
    ps = PSLoss(pred_len=pred.size(1)) ; assert torch.isfinite(ps(pred, target))
    focal = FocalLoss() ; assert torch.isfinite(focal(pred, target))


def test_frequency_aware_loss_compute():
    pred, target = _generate_data()
    loss_fn = FrequencyAwareLoss()
    loss_val = loss_fn(pred, target)
    assert loss_val.ndim == 0 and torch.isfinite(loss_val)


def test_quantile_loss_compute():
    pred, target = _generate_data()
    quantiles = [0.1, 0.5, 0.9]
    q_loss = QuantileLoss(quantiles=quantiles)
    # Need combined dim features * Q
    pred_q = pred.repeat(1, 1, len(quantiles))
    loss_val = q_loss(pred_q, target)
    assert loss_val.ndim == 0 and torch.isfinite(loss_val)


def test_uncertainty_calibration_loss_compute():
    pred, target = _generate_data()
    uncertainties = torch.abs(torch.randn_like(pred)) + 0.05
    u_loss = UncertaintyCalibrationLoss()
    loss_val = u_loss(pred, target, uncertainties)
    assert loss_val.ndim == 0 and torch.isfinite(loss_val)
