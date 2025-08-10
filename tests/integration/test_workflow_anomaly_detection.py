"""Anomaly detection workflow test (split from monolithic workflows)."""
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


def test_anomaly_detection_reconstruction_gap() -> None:
    if AutoformerFixed is None:
        pytest.skip("AutoformerFixed unavailable")

    cfg = make_base_config()
    cfg.task_name = "anomaly_detection"

    model = AutoformerFixed(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    x_enc, x_mark_enc, x_dec, x_mark_dec, _ = generate_synthetic_data(cfg, batch_size=4)
    batch_size, seq_len, feats = x_enc.shape

    anomaly_mask = torch.rand(batch_size, seq_len, feats) < 0.05
    anomalous = x_enc.clone()
    anomalous[anomaly_mask] = anomalous[anomaly_mask] * 5

    target = x_enc

    model.train()
    for _ in range(3):
        opt.zero_grad()
        out = model(anomalous, x_mark_enc, x_dec, x_mark_dec)
        loss = crit(out, target)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        recon = model(anomalous, x_mark_enc, x_dec, x_mark_dec)
    recon_error = (recon - anomalous).abs()
    normal_err = recon_error[~anomaly_mask].mean()
    anomaly_err = recon_error[anomaly_mask].mean()

    assert anomaly_err > normal_err, "Anomalies should produce higher reconstruction error"
