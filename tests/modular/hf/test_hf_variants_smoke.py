"""HF Autoformer variant smoke tests (Batch E).

If HF models not installed, the code should degrade gracefully (either by skipping
or using internal mock wrappers). We assert basic output shape parity across variant
types with a shared tiny configuration.
"""
from __future__ import annotations

import importlib
import pytest
import torch


HF_VARIANTS = [
    ("Enhanced", "HFEnhancedAutoformer"),
    ("Bayesian", "HFBayesianAutoformer"),
]


def _maybe_get_class(class_name: str):  # type: ignore[no-untyped-def]
    try:
        module = importlib.import_module("models.HFAutoformerSuite")
        return getattr(module, class_name)
    except Exception:
        return None


@pytest.mark.parametrize("variant,label", HF_VARIANTS)
def test_hf_variant_smoke(variant: str, label: str) -> None:
    cls = _maybe_get_class(label)
    if cls is None:
        pytest.skip("HF suite not available")

    # Minimal mock config
    class Cfg:  # noqa: D401 - minimal internal container
        def __init__(self):
            self.task_name = 'long_term_forecast'
            self.seq_len = 32
            self.label_len = 16
            self.pred_len = 8
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
            self.d_model = 64
            self.n_heads = 4
            self.e_layers = 1
            self.d_layers = 1
            self.d_ff = 128
            self.dropout = 0.0
            self.activation = 'gelu'
            self.output_attention = False
            self.mix = False
            self.backbone_model = 'chronos_tiny'
            self.mode = 'MS'

    cfg = Cfg()
    model = cls(cfg)

    x_enc = torch.randn(1, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.zeros(1, cfg.seq_len, 1)
    x_dec = torch.randn(1, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.zeros(1, cfg.label_len + cfg.pred_len, 1)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert out.shape[-1] == cfg.c_out
