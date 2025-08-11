"""HF training smoke test (Batch E).

Executes a single optimization step on a tiny HF variant (if available) to ensure
backward compatibility with training loop expectations.
"""
from __future__ import annotations

import importlib
import pytest
import torch


def _get_enhanced():  # type: ignore[no-untyped-def]
    try:
        module = importlib.import_module("models.HFAutoformerSuite")
        return getattr(module, "HFEnhancedAutoformer")
    except Exception:
        return None


@pytest.mark.skipif(_get_enhanced() is None, reason="HF suite not installed")
def test_hf_training_single_step() -> None:
    cls = _get_enhanced()
    assert cls is not None

    class Cfg:
        def __init__(self):
            self.task_name = 'long_term_forecast'
            self.seq_len = 16
            self.label_len = 8
            self.pred_len = 4
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
            self.d_model = 32
            self.n_heads = 2
            self.e_layers = 1
            self.d_layers = 1
            self.d_ff = 64
            self.dropout = 0.0
            self.activation = 'gelu'
            self.output_attention = False
            self.mix = False
            self.backbone_model = 'chronos_tiny'
            self.mode = 'MS'

    cfg = Cfg()
    model = cls(cfg)

    x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.zeros(2, cfg.seq_len, 1)
    x_dec = torch.randn(2, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.zeros(2, cfg.label_len + cfg.pred_len, 1)
    target = torch.randn(2, cfg.pred_len, cfg.c_out)

    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert out.shape == target.shape

    loss = (out - target).pow(2).mean()
    loss.backward()
    # If we reach here without exception the smoke passes.
