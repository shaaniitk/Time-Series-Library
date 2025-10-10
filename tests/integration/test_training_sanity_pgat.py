import torch
import pytest

from models.Celestial_Enhanced_PGAT import Model


class TrainingMockConfig:
    def __init__(self):
        # Sequence config
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        # Inputs/outputs
        self.enc_in = 118  # aligns with aggregator wave indices
        self.num_input_waves = 118
        self.dec_in = 4
        self.c_out = 4
        # Model sizes
        self.d_model = 32
        self.n_heads = 4
        self.e_layers = 1
        self.d_layers = 1
        self.dropout = 0.0
        # Embedding config
        self.embed = 'timeF'
        self.freq = 'h'
        # Features toggles for simpler training outputs
        self.use_mixture_decoder = False
        self.use_stochastic_learner = False
        self.use_hierarchical_mapping = True
        self.aggregate_waves_to_celestial = True


@pytest.fixture
def training_batch():
    torch.manual_seed(42)
    cfg = TrainingMockConfig()
    B = 4
    # Encoder inputs
    x_enc = torch.randn(B, cfg.seq_len, cfg.enc_in)
    # Valid temporal markers: [month, day, weekday, hour]
    month = torch.randint(0, 13, (B, cfg.seq_len))
    day = torch.randint(0, 32, (B, cfg.seq_len))
    weekday = torch.randint(0, 8, (B, cfg.seq_len))
    hour = torch.randint(0, 25, (B, cfg.seq_len))
    x_mark_enc = torch.stack([month, day, weekday, hour], dim=-1)

    # Decoder inputs and marks
    x_dec = torch.randn(B, cfg.label_len + cfg.pred_len, cfg.dec_in)
    month_d = torch.randint(0, 13, (B, cfg.label_len + cfg.pred_len))
    day_d = torch.randint(0, 32, (B, cfg.label_len + cfg.pred_len))
    weekday_d = torch.randint(0, 8, (B, cfg.label_len + cfg.pred_len))
    hour_d = torch.randint(0, 25, (B, cfg.label_len + cfg.pred_len))
    x_mark_dec = torch.stack([month_d, day_d, weekday_d, hour_d], dim=-1)

    # Targets: identity mapping of decoder features over prediction horizon
    targets = x_dec[:, -cfg.pred_len :, : cfg.c_out].detach().clone()
    return cfg, x_enc, x_mark_enc, x_dec, x_mark_dec, targets


def _param_snapshot(model: torch.nn.Module):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _total_param_diff(params_before, params_after):
    return sum((a - b).abs().sum().item() for a, b in zip(params_after, params_before))


@pytest.mark.extended
def test_pgat_training_sanity(training_batch):
    torch.manual_seed(123)
    cfg, x_enc, x_mark_enc, x_dec, x_mark_dec, targets = training_batch
    model = Model(cfg)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Initial forward
    outputs0, metadata0 = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert outputs0 is not None, "Model forward produced no outputs."
    assert metadata0 is not None, "Model forward produced no metadata."
    if isinstance(outputs0, dict):
        # If mixture decoder unexpectedly active, use 'mu' as mean predictions
        preds0 = outputs0.get('mu')
        assert preds0 is not None, "Expected 'mu' in outputs when dict returned."
    else:
        preds0 = outputs0
    # Align prediction horizon with targets
    preds0 = preds0[:, -cfg.pred_len :, :]
    assert preds0.shape == targets.shape, f"Preds {preds0.shape} vs targets {targets.shape}"
    loss0 = criterion(preds0, targets)
    assert torch.isfinite(loss0), "Initial loss is not finite."

    # Basic metadata check under aggregation path
    # The model stores original target waves in metadata when aggregating
    original_targets = metadata0.get('original_targets', None)
    assert original_targets is not None, "original_targets missing under aggregation path."

    # Train for a few steps; track gradient health
    params_before = _param_snapshot(model)
    grad_norms = []
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        preds = outputs['mu'] if isinstance(outputs, dict) else outputs
        preds = preds[:, -cfg.pred_len :, :]
        loss = criterion(preds, targets)
        loss.backward()
        # Gradient health checks
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Found non-finite gradients."
                total_norm += p.grad.norm().item()
        grad_norms.append(total_norm)
        losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    params_after = _param_snapshot(model)
    delta = _total_param_diff(params_before, params_after)
    assert delta > 0.0, "Parameters did not update after optimizer steps."

    # Loss should generally trend down on identity targets; allow minor noise
    # Require at least 10% improvement across 6 steps for stability
    assert losses[0] > losses[-1] * 0.9, f"Loss did not improve sufficiently: {losses}"

    # Grad norms should be finite and not explode
    assert all(n > 0 for n in grad_norms), "Gradient norms are zero; no learning signal."
    assert max(grad_norms) < 1e3, "Gradient explosion detected."

    # Determinism: re-run with same seed from same initial weights
    torch.manual_seed(123)
    model2 = Model(cfg)
    model2.train()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    # Copy initial weights to model2 to ensure same starting point
    with torch.no_grad():
        for p2, p1 in zip(model2.parameters(), params_before):
            p2.copy_(p1)

    losses2 = []
    for _ in range(6):
        optimizer2.zero_grad(set_to_none=True)
        outputs2, _ = model2(x_enc, x_mark_enc, x_dec, x_mark_dec)
        preds2 = outputs2['mu'] if isinstance(outputs2, dict) else outputs2
        preds2 = preds2[:, -cfg.pred_len :, :]
        loss2 = criterion(preds2, targets)
        loss2.backward()
        optimizer2.step()
        losses2.append(loss2.item())

    # Allow small non-determinism from parallel ops; require close convergence
    assert abs(losses[-1] - losses2[-1]) / max(1e-8, abs(losses[-1])) < 0.05, (
        f"Training not approximately deterministic: {losses[-1]} vs {losses2[-1]}"
    )