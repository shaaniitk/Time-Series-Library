"""Output head invariants (Phase 3 - Output Heads).

Covers:
1. Determinism (eval mode): two forward passes with identical input produce (near) identical outputs.
2. Probabilistic head logvar clamp range and std consistency with logvar.
3. Multi-task head task shapes and forecasting reshape consistency.

Skips gracefully if a given head implementation is unavailable.
"""
from __future__ import annotations

import pytest
import torch

from .thresholds import get_threshold


def _make_hidden(batch: int = 2, seq: int = 32, d_model: int = 64) -> torch.Tensor:
    return torch.randn(batch, seq, d_model)


def _try_import(name: str):  # returns class or skips
    try:
        from utils.modular_components.implementations.outputs import (
            ForecastingHead, RegressionHead, ClassificationHead,
            ProbabilisticForecastingHead, MultiTaskHead, OutputConfig
        )  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("Output heads module unavailable")
    mapping = {
        'forecasting': ForecastingHead,
        'prob_forecasting': ProbabilisticForecastingHead,
        'multi_task': MultiTaskHead,
    }
    if name not in mapping:
        pytest.skip(f"{name} head not mapped")
    return mapping[name], OutputConfig


@pytest.mark.invariant
@pytest.mark.parametrize("head_name", ["forecasting"])  # extend if needed
def test_output_head_determinism(head_name: str):
    HeadCls, OutputConfig = _try_import(head_name)
    cfg = OutputConfig(d_model=64, output_dim=3, horizon=5, dropout=0.0)
    head = HeadCls(cfg)
    head.eval()
    hidden = _make_hidden(batch=2, seq=40, d_model=64)
    with torch.no_grad():
        out1 = head(hidden)
        out2 = head(hidden)
    rel_err = torch.linalg.norm(out1 - out2) / (torch.linalg.norm(out1) + 1e-12)
    assert rel_err <= get_threshold("output_head_determinism_rel_err"), rel_err


@pytest.mark.invariant
def test_probabilistic_head_logvar_and_std_consistency():
    HeadCls, OutputConfig = _try_import("prob_forecasting")
    cfg = OutputConfig(d_model=48, output_dim=2, horizon=4, dropout=0.0)
    head = HeadCls(cfg)
    head.eval()
    hidden = _make_hidden(batch=3, seq=20, d_model=48)
    with torch.no_grad():
        out = head(hidden)
    mean, logvar, std = out['mean'], out['logvar'], out['std']
    # Clamp bounds
    assert logvar.min().item() >= get_threshold("output_prob_logvar_min"), logvar.min().item()
    assert logvar.max().item() <= get_threshold("output_prob_logvar_max"), logvar.max().item()
    # Consistency: std ~ exp(0.5 * logvar)
    recon = torch.exp(0.5 * logvar)
    rel = torch.linalg.norm(std - recon) / (torch.linalg.norm(recon) + 1e-12)
    assert rel <= get_threshold("output_prob_std_consistency_rel_err"), rel


@pytest.mark.invariant
def test_multi_task_head_shapes():
    HeadCls, OutputConfig = _try_import("multi_task")
    task_cfgs = {
        'reg': {'type': 'regression', 'output_dim': 2},
        'cls': {'type': 'classification', 'num_classes': 3, 'output_dim': 3},
        'fcst': {'type': 'forecasting', 'output_dim': 1, 'horizon': 4},
    }
    cfg = OutputConfig(d_model=32, output_dim=0, horizon=0, dropout=0.0)
    head = HeadCls(cfg, task_configs=task_cfgs)
    head.eval()
    hidden = _make_hidden(batch=2, seq=16, d_model=32)
    with torch.no_grad():
        outputs = head(hidden)
    assert set(outputs.keys()) == set(task_cfgs.keys())
    assert outputs['reg'].shape == (2, 2)
    assert outputs['cls'].shape == (2, 3)
    assert outputs['fcst'].shape == (2, 4, 1)


__all__ = []
