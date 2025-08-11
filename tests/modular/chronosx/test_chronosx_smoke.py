"""ChronosX Autoformer smoke tests (Batch D).

Validates basic instantiation and forward pass for tiny model size (mock pipeline
fallback acceptable). Keeps runtime minimal; heavier benchmarks will reside in an
optional slow suite later.
"""
from __future__ import annotations

import torch
import pytest

from configs.autoformer.chronosx_config import create_chronosx_config
from models.chronosx_autoformer import ChronosXAutoformer


@pytest.mark.parametrize("model_size,uncertainty", [
    ("tiny", False),
    ("tiny", True),
])
def test_chronosx_forward_smoke(model_size: str, uncertainty: bool) -> None:
    """Ensure ChronosXAutoformer constructs & produces forecast tensor.

    Args:
        model_size: ChronosX pretrained size alias.
        uncertainty: Whether to enable uncertainty sampling path.
    """
    cfg = create_chronosx_config(
        seq_len=32,
        pred_len=8,
        enc_in=1,
        c_out=1,
        chronosx_model_size=model_size,
        uncertainty_enabled=uncertainty,
        num_samples=5 if uncertainty else 1,
    )

    model = ChronosXAutoformer(cfg)

    x_enc = torch.randn(1, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.zeros(1, cfg.seq_len, 1)
    x_dec = torch.randn(1, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.zeros(1, cfg.label_len + cfg.pred_len, 1)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert out.shape == (1, cfg.pred_len, cfg.c_out)
    if uncertainty:
        # If uncertainty path enabled, model should report support
        assert model.supports_uncertainty()
