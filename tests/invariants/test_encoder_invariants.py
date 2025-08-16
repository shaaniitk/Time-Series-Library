"""Encoder invariants (Phase 2 start)."""
from __future__ import annotations

import pytest
import torch

from tests.helpers import time_series_generators as gen
from .thresholds import get_threshold
from . import metrics as M


def _get_encoder_impl():  # type: ignore[return-type]
    try:
        from tools.unified_component_registry import ensure_initialized, get_component  # type: ignore
        ensure_initialized()
        enc_cls = get_component("encoder", "autoformer")
        if enc_cls is not None:
            inst = enc_cls()
            setattr(inst, "_fallback", False)
            return inst
    except Exception:
        pass
    try:
        from layers.Autoformer_EncDec import EncoderLayer, Encoder  # type: ignore
        import torch.nn as nn
        attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        layer = EncoderLayer(attention=attn, d_model=32, moving_avg=7)
        enc = Encoder([layer])
        setattr(enc, "_fallback", True)
        return enc
    except Exception as e:  # pragma: no cover
        pytest.skip(f"No encoder implementation available: {e}")


def _hf_energy(x: torch.Tensor) -> float:
    return M.high_frequency_energy(x)


def test_encoder_determinism():
    enc = _get_encoder_impl()
    # Force evaluation mode to eliminate dropout / training randomness
    if hasattr(enc, 'eval'):
        enc.eval()
    x = gen.seasonal_with_trend(batch=1, length=96, dim=1)
    if x.shape[-1] != 32:
        x = torch.nn.functional.pad(x, (0, 32 - x.shape[-1]))
    out1, _ = enc(x)
    out2, _ = enc(x)
    diff = torch.linalg.norm(out1 - out2) / (torch.linalg.norm(out1) + 1e-12)
    assert diff < get_threshold("encoder_determinism_tol"), diff


def test_encoder_high_freq_non_increase():
    enc = _get_encoder_impl()
    if hasattr(enc, 'eval'):
        enc.eval()
    x = gen.seasonal_with_trend(batch=1, length=128, dim=1)
    if x.shape[-1] != 32:
        x = torch.nn.functional.pad(x, (0, 32 - x.shape[-1]))
    try:
        layers = list(enc.attn_layers)  # type: ignore[attr-defined]
    except Exception:
        layers = []
    hf_input = _hf_energy(x)
    is_fallback = bool(getattr(enc, "_fallback", False))
    prev = hf_input
    cur = None
    with torch.no_grad():
        temp = x
        if layers:
            for layer in layers:
                temp, _ = layer(temp)
                cur = _hf_energy(temp)
                tol = get_threshold("encoder_hf_increase_tol") * (10 if is_fallback else 1)
                assert cur <= prev + tol, (prev, cur)
                prev = cur
        else:
            out, _ = enc(x)
            cur = _hf_energy(out)
            tol = get_threshold("encoder_hf_increase_tol") * (10 if is_fallback else 1)
            assert cur <= prev + tol, (prev, cur)
    assert cur is not None


@pytest.mark.gradcheck
def test_encoder_gradient_flow():
    enc = _get_encoder_impl()
    if hasattr(enc, 'train'):
        enc.train()  # ensure gradients propagate (some encoders may behave differently in eval)
    x = gen.seasonal_with_trend(batch=1, length=48, dim=1)
    if x.shape[-1] != 32:
        x = torch.nn.functional.pad(x, (0, 32 - x.shape[-1]))
    x = x.detach().requires_grad_(True)
    out = enc(x)  # type: ignore
    if isinstance(out, tuple):  # handle (output, attn) style
        out_primary = out[0]
    else:
        out_primary = out
    loss = out_primary.sum()
    loss.backward()
    grad_norm = float(x.grad.norm()) if x.grad is not None else 0.0
    assert grad_norm > get_threshold("encoder_input_grad_min_norm"), grad_norm

__all__ = []
