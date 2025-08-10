"""Model persistence (save/load) workflow test (split from monolithic workflows)."""
from __future__ import annotations

import pytest
import torch
import os
import tempfile

from integration.helpers.workflow_data import make_base_config, generate_synthetic_data  # type: ignore

try:  # pragma: no cover
    from models.Autoformer_Fixed import Model as AutoformerFixed  # type: ignore
except Exception:  # pragma: no cover
    AutoformerFixed = None  # type: ignore


def test_model_persistence_round_trip() -> None:
    if AutoformerFixed is None:
        pytest.skip("AutoformerFixed unavailable")

    cfg = make_base_config()

    model = AutoformerFixed(cfg)
    x_enc, x_mark_enc, x_dec, x_mark_dec, _ = generate_synthetic_data(cfg, batch_size=2)

    model.eval()
    with torch.no_grad():
        original = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
        torch.save(model.state_dict(), path)

    new_model = AutoformerFixed(cfg)
    new_model.load_state_dict(torch.load(path))
    new_model.eval()
    with torch.no_grad():
        loaded = new_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert torch.allclose(original, loaded, atol=1e-6)
    os.remove(path)
