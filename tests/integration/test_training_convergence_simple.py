import torch
import pytest

from models.Celestial_Enhanced_PGAT import Model


class SimpleConfig:
    def __init__(self):
        # Sequence config
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        # Inputs/outputs
        self.enc_in = 118  # aligns with aggregator indices
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
        # Advanced features OFF
        self.use_mixture_decoder = False
        self.use_stochastic_learner = False
        self.use_hierarchical_mapping = False
        self.aggregate_waves_to_celestial = True


@pytest.mark.extended
def test_pgat_converges_in_5_epochs_mse_only():
    torch.manual_seed(7)
    cfg = SimpleConfig()
    model = Model(cfg)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    B = 8
    # Encoder inputs
    x_enc = torch.randn(B, cfg.seq_len, cfg.enc_in)
    # Valid temporal markers
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

    # Identity-like targets over prediction horizon
    targets = x_dec[:, -cfg.pred_len :, : cfg.c_out].detach().clone()

    # Verify outputs are tensor (advanced features off)
    outputs, _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert isinstance(outputs, torch.Tensor), "Outputs should be tensor when advanced features are off."

    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        preds, _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        preds = preds[:, -cfg.pred_len :, :]
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(loss.item())

    # Convergence: final loss at least 20% lower than initial
    assert losses[-1] <= losses[0] * 0.8, f"Insufficient convergence over 5 epochs: {losses}"