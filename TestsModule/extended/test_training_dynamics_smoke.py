"""Training dynamics smoke test.

Ensures a tiny encoder-decoder pair performs one optimization step without
runtime errors and that loss decreases for a second step (very weak signal).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from TestsModule.utils import make_encoder, make_decoder, random_series, make_output_head

pytestmark = [pytest.mark.extended]


@pytest.mark.parametrize("enc_name", ["standard"])  # keep minimal for speed
@pytest.mark.parametrize("dec_name", ["standard"])
def test_one_step_training_loss_reduction(enc_name: str, dec_name: str) -> None:
    torch.manual_seed(42)
    d_model = 8
    seq_enc, seq_dec = 32, 16
    enc = make_encoder(enc_name, d_model=d_model, n_heads=2)
    dec = make_decoder(dec_name, d_model=d_model, n_heads=2, c_out=d_model)
    head = make_output_head("standard", d_model=d_model, c_out=d_model)

    params = list(enc.parameters()) + list(dec.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    criterion = nn.MSELoss()

    x_enc = random_series(2, seq_enc, d_model)
    x_dec = random_series(2, seq_dec, d_model)
    y_true = random_series(2, seq_dec, d_model)

    def forward_loss() -> torch.Tensor:
        enc_out_raw = enc(x_enc)  # type: ignore[misc]
        enc_out = enc_out_raw[0] if isinstance(enc_out_raw, (tuple, list)) else enc_out_raw
        init_trend = torch.zeros_like(x_dec)
        dec_out = dec(x_dec, enc_out, trend=init_trend)  # type: ignore[misc]
        seasonal = dec_out[0]
        pred = head(seasonal)  # identity shape
        return criterion(pred, y_true)

    loss1 = forward_loss().item()
    optimizer.zero_grad()
    forward_loss().backward()
    optimizer.step()
    loss2 = forward_loss().item()
    # Allow equality (stochastic) but ensure not exploding
    assert loss2 < loss1 * 1.05, f"Loss not stable/reducing: {loss1} -> {loss2}"
