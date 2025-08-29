"""Classification workflow test (split from monolithic workflows)."""
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


def test_classification_basic_accuracy() -> None:
    if AutoformerFixed is None:
        pytest.skip("AutoformerFixed unavailable")

    cfg = make_base_config()
    cfg.task_name = "classification"
    cfg.num_classes = 5

    model = AutoformerFixed(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    batch_size = 4
    x_enc, x_mark_enc, x_dec, x_mark_dec, _ = generate_synthetic_data(cfg, batch_size=batch_size)
    target = torch.randint(0, cfg.num_classes, (batch_size,))

    model.train()
    for _ in range(5):  # short training
        opt.zero_grad()
        logits = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = crit(logits, target)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert logits.shape == (batch_size, cfg.num_classes)

    _, pred = torch.max(logits, 1)
    acc = (pred == target).float().mean().item()
    assert acc >= 1.0 / cfg.num_classes, "Accuracy should beat random baseline"
