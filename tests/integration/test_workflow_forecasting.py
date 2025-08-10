"""Forecasting workflow test (split from monolithic workflows)."""
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

@pytest.mark.smoke
def test_forecasting_workflow_decreases_loss() -> None:
    if AutoformerFixed is None:
        pytest.skip("AutoformerFixed unavailable")

    cfg = make_base_config()
    cfg.task_name = "long_term_forecast"

    model = AutoformerFixed(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    x_enc, x_mark_enc, x_dec, x_mark_dec, target = generate_synthetic_data(cfg, batch_size=4)

    model.train()
    losses = []
    for _ in range(3):  # reduced epochs for speed
        opt.zero_grad()
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = crit(out, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] <= losses[0], "Loss should not increase"
    assert out.shape == target.shape
