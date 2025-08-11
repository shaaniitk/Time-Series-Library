"""ModularAutoformer smoke test.

Validates that a tiny ModularAutoformer instance instantiated via the
structured configuration can:
1. Perform a forward pass with None temporal marks.
2. Produce output of expected shape (batch, pred_len, c_out).
3. Execute a single optimization step without runtime errors.
4. Yield a stable (non-exploding) loss that is not significantly worse
   after one training step (weak reduction/stability criterion).

Kept intentionally small for speed: reduced d_model and feedforward dims.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from models.modular_autoformer import ModularAutoformer
from configs.schemas import create_enhanced_config

pytestmark = [pytest.mark.extended]


def test_modular_autoformer_forward_and_train_step() -> None:
    """Run a minimal training step and basic shape assertions.

    The test ensures forward/backward works with deterministic sampling
    and that loss is numerically stable (weak reduction condition).
    """
    torch.manual_seed(123)

    # Tiny configuration for speed
    seq_len = 32
    label_len = 16
    pred_len = 8
    num_targets = 1
    num_covariates = 0
    d_model = 16  # Must be divisible by n_heads (8 in helper config)

    config = create_enhanced_config(
        num_targets=num_targets,
        num_covariates=num_covariates,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=d_model,
    )

    # Shrink feedforward dimensions to keep test fast
    config.encoder.d_ff = 64  # type: ignore[attr-defined]
    config.decoder.d_ff = 64  # type: ignore[attr-defined]

    model = ModularAutoformer(config)
    model.train()

    batch_size = 2
    # Encoder input: (B, seq_len, enc_in)
    x_enc = torch.randn(batch_size, seq_len, config.enc_in)
    # Decoder input: (B, label_len + pred_len, dec_in)
    x_dec = torch.randn(batch_size, label_len + pred_len, config.dec_in)

    # Temporal marks omitted (None) to use value embedding only
    x_mark_enc = None
    x_mark_dec = None

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    def forward_loss() -> torch.Tensor:
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # type: ignore[arg-type]
        assert pred is not None, "Model forward returned None for forecasting task"
        assert pred.shape == (batch_size, pred_len, num_targets), (
            f"Unexpected prediction shape {pred.shape}, expected {(batch_size, pred_len, num_targets)}"
        )
        target = torch.randn_like(pred)
        return criterion(pred, target)

    loss_before = forward_loss().item()
    optimizer.zero_grad()
    forward_loss().backward()
    optimizer.step()
    loss_after = forward_loss().item()

    # Allow slight increase (stochastic) but ensure no explosion
    assert loss_after < loss_before * 1.2, f"Loss unstable: {loss_before} -> {loss_after}"
