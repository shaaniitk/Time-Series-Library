"""Hierarchical (cross-resolution + wavelet) ModularAutoformer smoke test.

Validates a hierarchical configuration can:
1. Forward pass with hierarchical attention + wavelet decomposition.
2. Produce correct forecast shape (B, pred_len, c_out).
3. Expose stored native wavelet levels (if decomposition supports it) in descending length order.
4. Run a single optimization step.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from models.modular_autoformer import ModularAutoformer
from configs.schemas import create_hierarchical_config

pytestmark = [pytest.mark.extended]


def test_hierarchical_modular_autoformer_smoke() -> None:
    torch.manual_seed(789)

    seq_len = 48
    label_len = 16
    pred_len = 8
    num_targets = 1
    num_covariates = 0
    d_model = 16

    config = create_hierarchical_config(
        num_targets=num_targets,
        num_covariates=num_covariates,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=d_model,
        n_levels=3,
    )

    # Shrink FFN dims
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

    def forward_loss() -> torch.Tensor:
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # type: ignore[arg-type]
        assert pred is not None
        assert pred.shape == (batch_size, pred_len, num_targets)
        target = torch.randn_like(pred)
        return criterion(pred, target)

    loss_before = forward_loss().item()
    optimizer.zero_grad()
    forward_loss().backward()
    optimizer.step()
    loss_after = forward_loss().item()

    assert loss_after < loss_before * 1.25

    # Wavelet hierarchical decomposition introspection (if available)
    wavelet_levels = getattr(model, 'init_decomp', None)
    if wavelet_levels and hasattr(wavelet_levels, 'last_native_levels'):
        levels = wavelet_levels.last_native_levels or []
        # Ensure descending length order property (monotonic non-increasing)
        if levels:
            lengths = [lvl.shape[1] for lvl in levels]
            assert all(lengths[i] >= lengths[i+1] for i in range(len(lengths)-1)), "Wavelet levels not in descending order"
