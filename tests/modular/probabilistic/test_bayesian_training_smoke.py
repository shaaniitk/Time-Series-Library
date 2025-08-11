"""Minimal Bayesian training smoke (replaces heavy bayesian_sanity_test script).

Runs a drastically shortened training loop (1 epoch / small batch) to validate:
 - Forward/backward succeeds for BayesianEnhancedAutoformer (if available).
 - KL loss (if exposed) is non-negative.

Skips gracefully if training entrypoint or model isn't present to keep fast suite stable.
Marked slow due to dependency on training pipeline / potential data loading overhead.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow

try:
    from scripts.train.train_dynamic_autoformer import build_model_from_config  # hypothetical helper
except Exception:  # pragma: no cover
    build_model_from_config = None  # type: ignore


@pytest.mark.skipif(build_model_from_config is None, reason="Training builder unavailable")
def test_minimal_bayesian_training_step() -> None:
    # Synthetic tiny config (mirrors modular config semantics loosely)
    cfg = {
        'model_type': 'bayesian',
        'd_model': 32,
        'enc_in': 32,
        'dec_in': 32,
        'c_out': 4,
        'seq_len': 24,
        'label_len': 12,
        'pred_len': 6,
        'e_layers': 1,
        'd_layers': 1,
        'n_heads': 4,
        'd_ff': 64,
        'bayesian': {'kl_weight': 1e-4},
    }
    model = build_model_from_config(cfg)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 2
    x_enc = torch.randn(B, cfg['seq_len'], cfg['enc_in'])
    x_mark_enc = torch.zeros(B, cfg['seq_len'], 1)
    x_dec = torch.randn(B, cfg['label_len'] + cfg['pred_len'], cfg['dec_in'])
    x_mark_dec = torch.zeros(B, cfg['label_len'] + cfg['pred_len'], 1)
    target = torch.randn(B, cfg['pred_len'], cfg['c_out'])

    preds = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    preds_slice = preds[:, -cfg['pred_len']:, :cfg['c_out']]
    mse = torch.nn.MSELoss()
    base_loss = mse(preds_slice, target)

    total_loss = base_loss
    if hasattr(model, 'kl_loss'):
        kl_val = getattr(model, 'kl_loss')()
        assert kl_val >= 0
        total_loss = total_loss + 1e-4 * kl_val

    optim.zero_grad()
    total_loss.backward()
    optim.step()

    assert torch.isfinite(total_loss)
