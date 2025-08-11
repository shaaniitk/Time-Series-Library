"""Extended encoder-decoder roundtrip smoke test."""
from __future__ import annotations

import pytest
import torch

from TestsModule.utils import make_encoder, make_decoder, random_series

pytestmark = [pytest.mark.extended]


@pytest.mark.parametrize("enc_name", ["standard", "enhanced", "stable"])  # hierarchical covered elsewhere
@pytest.mark.parametrize("dec_name", ["standard", "enhanced", "stable"])
def test_roundtrip_forward(enc_name: str, dec_name: str) -> None:
    d_model = 16
    c_out = 16
    x_enc = random_series(2, 64, d_model)
    x_dec = random_series(2, 32, d_model, requires_grad=True)

    enc = make_encoder(enc_name, d_model=d_model, n_heads=2)
    enc_out_raw = enc(x_enc)  # type: ignore[misc]
    enc_out = enc_out_raw[0] if isinstance(enc_out_raw, (tuple, list)) else enc_out_raw
    dec = make_decoder(dec_name, d_model=d_model, n_heads=2, c_out=c_out)

    init_trend = torch.zeros_like(x_dec)
    out = dec(x_dec, enc_out, trend=init_trend)  # type: ignore[misc]
    assert isinstance(out, tuple) and len(out) in (2, 3)
    seasonal = out[0]
    assert seasonal.shape == x_dec.shape
    seasonal.mean().backward()
    assert x_dec.grad is not None and torch.isfinite(x_dec.grad).all()
