"""Imputation workflow test (split from monolithic workflows)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from integration.helpers.workflow_data import make_base_config, generate_synthetic_data  # type: ignore

try:  # pragma: no cover
    from models.Autoformer_Fixed import Model as AutoformerFixed  # type: ignore
except Exception:  # pragma: no cover
    AutoformerFixed = None  # type: ignore


def test_imputation_workflow_masked_recovery() -> None:
    if AutoformerFixed is None:
        pytest.skip("AutoformerFixed unavailable")

    cfg = make_base_config()
    cfg.task_name = "imputation"
    cfg.mask_rate = 0.25

    model = AutoformerFixed(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    x_enc, x_mark_enc, x_dec, x_mark_dec, _ = generate_synthetic_data(cfg, batch_size=4)

    batch_size, seq_len, feats = x_enc.shape
    mask = torch.rand(batch_size, seq_len, feats) > cfg.mask_rate
    masked_x_enc = x_enc.clone()
    masked_x_enc[~mask] = 0

    target = x_enc

    model.train()
    losses = []
    for _ in range(3):
        opt.zero_grad()
        out = model(masked_x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = crit(out, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] <= losses[0], "Loss should not increase"
    assert out.shape == target.shape
