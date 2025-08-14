from __future__ import annotations

"""Focused tests for ChronosXAutoformer multivariate & covariate adapter.

Kept minimal, non-duplicated coverage:
1. Shape + diversity with adapter enabled (deterministic monkeypatch).
2. Fallback shape when adapter disabled.
3. Covariate residual effect.
"""

import torch
import pytest
from argparse import Namespace
from typing import List

from models.chronosx_autoformer import ChronosXAutoformer


def _build_base_config(enc_in: int = 5, c_out: int = 3) -> Namespace:
    """Create a minimal Namespace config for ChronosXAutoformer under test.

    Parameters
    ----------
    enc_in : int
        Number of encoder input channels.
    c_out : int
        Number of target output channels.
    """
    return Namespace(
        # Core sequence settings
        seq_len=32,
        label_len=16,
        pred_len=8,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=c_out,
        d_model=64,
        # Task / framework flags
        task_name='long_term_forecast',
        chronosx_model_size='tiny',
        uncertainty_enabled=False,
        num_samples=4,
        device='cpu',
    force_mock_pipeline=True,
    disable_hf_integration=True,
        # Adapter flags
        enable_multivariate_adapter=True,
        enable_covariate_residual=False,
        adapter_context_channel_map=list(range(c_out)),
        cov_residual_lookback=6,
    )


def test_multivariate_adapter_shape_and_diversity() -> None:
    """Shape + diversity with deterministic monkeypatched pipeline."""
    enc_in, c_out = 5, 3
    config = _build_base_config(enc_in=enc_in, c_out=c_out)
    model = ChronosXAutoformer(config)
    assert model.multivariate_adapter is not None

    def _deterministic_predict(context: List[torch.Tensor], prediction_length: int, num_samples: int = 1, **_) -> torch.Tensor:
        return torch.stack([series[-prediction_length:] for series in context], dim=0)

    model.multivariate_adapter.chronos_pipeline.predict = _deterministic_predict  # type: ignore[attr-defined]

    B = 3
    x_enc = torch.randn(B, config.seq_len, enc_in)
    x_mark_enc = torch.randn(B, config.seq_len, 2)
    x_dec = torch.randn(B, config.label_len + config.pred_len, enc_in)
    x_mark_dec = torch.randn(B, config.label_len + config.pred_len, 2)
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert out.shape == (B, config.pred_len, c_out)
    ref = out[..., 0]
    for ch in range(1, c_out):
        assert (ref - out[..., ch]).abs().mean().item() > 1e-6


def test_multivariate_adapter_disabled_fallback() -> None:
    """When adapter disabled, model still returns correct multivariate shape.

    Diversity is not enforced in fallback path (replication allowed).
    """
    config = _build_base_config(enc_in=4, c_out=3)
    config.enable_multivariate_adapter = False
    model = ChronosXAutoformer(config)
    assert model.multivariate_adapter is None

    B = 2
    x_enc = torch.randn(B, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(B, config.seq_len, 2)
    x_dec = torch.randn(B, config.label_len + config.pred_len, config.enc_in)
    x_mark_dec = torch.randn(B, config.label_len + config.pred_len, 2)

    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert out.shape == (B, config.pred_len, config.c_out)
def test_covariate_residual_effect() -> None:
    """Sanity check that covariate residual changes output when toggled.

    Run model twice: once with residual enabled, once disabled; same inputs -> outputs differ.
    """
    torch.manual_seed(123)
    batch_size = 2
    seq_len = 16
    label_len = 8
    pred_len = 4
    enc_in = 4
    c_out = 2

    base_cfg_kwargs = dict(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=c_out,
        d_model=32,
        chronosx_model_size="tiny",
        cov_residual_lookback=4,
        task_name="long_term_forecast",
    )

    cfg_resid = Namespace(**base_cfg_kwargs, enable_multivariate_adapter=True, enable_covariate_residual=True,
                          force_mock_chronosx=True, disable_hf_integration=True)
    cfg_no_resid = Namespace(**base_cfg_kwargs, enable_multivariate_adapter=True, enable_covariate_residual=False,
                             force_mock_chronosx=True, disable_hf_integration=True)

    model_resid = ChronosXAutoformer(cfg_resid).eval()
    model_no_resid = ChronosXAutoformer(cfg_no_resid).eval()

    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 3)
    x_dec = torch.randn(batch_size, label_len + pred_len, enc_in)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 3)

    with torch.no_grad():
        y_resid = model_resid(x_enc, x_mark_enc, x_dec, x_mark_dec)
        y_no_resid = model_no_resid(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert y_resid.shape == y_no_resid.shape
    diff = torch.mean(torch.abs(y_resid - y_no_resid)).item()
    assert diff > 1e-4, f"Covariate residual had no measurable effect (diff={diff:.6f})"
