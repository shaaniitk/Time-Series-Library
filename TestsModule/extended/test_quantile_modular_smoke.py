"""Quantile ModularAutoformer smoke test.

Ensures a tiny quantile + Bayesian (sampling) ModularAutoformer configuration:
1. Builds successfully with structured quantile config.
2. Produces output of shape (B, pred_len, num_targets * num_quantiles).
3. Supports quantiles via model.supports_quantiles().
4. Executes a forward/backward optimization step without errors.
5. Maintains loss stability (no explosion) after one optimizer step.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from models.modular_autoformer import ModularAutoformer
from configs.schemas import create_quantile_bayesian_config

pytestmark = [pytest.mark.extended]


def test_quantile_modular_autoformer_smoke() -> None:
    torch.manual_seed(321)

    # Minimal dimensions for speed
    seq_len = 32
    label_len = 16
    pred_len = 8
    num_targets = 2
    num_covariates = 0
    quantile_levels = [0.1, 0.5, 0.9]
    d_model = 16  # divisible by attention heads used in helper config (8)

    config = create_quantile_bayesian_config(
        num_targets=num_targets,
        num_covariates=num_covariates,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=d_model,
        n_samples=2,  # keep sampling cheap
        quantile_levels=quantile_levels,
    )

    # Reduce feedforward dims for speed
    config.encoder.d_ff = 64  # type: ignore[attr-defined]
    config.decoder.d_ff = 64  # type: ignore[attr-defined]

    model = ModularAutoformer(config)
    model.train()

    batch_size = 2
    x_enc = torch.randn(batch_size, seq_len, config.enc_in)
    x_dec = torch.randn(batch_size, label_len + pred_len, config.dec_in)
    x_mark_enc = None
    x_mark_dec = None

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    expected_c_out = num_targets * len(quantile_levels)

    def forward_loss() -> torch.Tensor:
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # type: ignore[arg-type]
        assert pred is not None, "Forward returned None for quantile forecast"
        assert pred.shape == (batch_size, pred_len, expected_c_out), (
            f"Unexpected prediction shape {pred.shape}, expected {(batch_size, pred_len, expected_c_out)}"
        )
        assert model.supports_quantiles(), "supports_quantiles() should be True for quantile config"
        # Simple target for MSE (random); real quantile loss tested elsewhere
        target = torch.randn_like(pred)
        return criterion(pred, target)

    loss_before = forward_loss().item()
    optimizer.zero_grad()
    forward_loss().backward()
    optimizer.step()
    loss_after = forward_loss().item()

    # Allow modest stochastic increase but guard against explosion
    assert loss_after < loss_before * 1.3, f"Loss unstable: {loss_before} -> {loss_after}"
