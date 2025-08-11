"""ChronosX scaling tests (Batch D).

Param sweep across a few model sizes ensuring forward path remains functional.
Uses smallest seq/pred lengths to minimize network load; real benchmarking will
be moved to slow/extended suites.
"""
from __future__ import annotations

import torch
import pytest

from configs.autoformer.chronosx_config import create_chronosx_config
from models.chronosx_autoformer import ChronosXAutoformer


@pytest.mark.parametrize("model_size", ["tiny", "mini"])  # keep small for CI speed
def test_chronosx_forward_scaling(model_size: str) -> None:
    """Test forward pass for selected ChronosX sizes."""
    cfg = create_chronosx_config(
        seq_len=24,
        pred_len=6,
        enc_in=1,
        c_out=1,
        chronosx_model_size=model_size,
        uncertainty_enabled=False,
    )
    model = ChronosXAutoformer(cfg)

    x_enc = torch.randn(1, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.zeros(1, cfg.seq_len, 1)
    x_dec = torch.randn(1, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.zeros(1, cfg.label_len + cfg.pred_len, 1)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert out.shape == (1, cfg.pred_len, cfg.c_out)
