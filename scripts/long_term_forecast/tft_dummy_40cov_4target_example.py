import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TemporalFusionTransformer import Model


def make_dummy_batch(batch_size, seq_len, label_len, pred_len, enc_in, c_out, known_len, device):
    x_enc = torch.randn(batch_size, seq_len, enc_in, device=device)
    x_mark_enc = torch.randn(batch_size, seq_len, known_len, device=device)

    # Decoder input is shaped like target channels (c_out), matching framework usage.
    x_dec = torch.randn(batch_size, label_len + pred_len, c_out, device=device)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, known_len, device=device)

    # Synthetic target for the prediction horizon.
    y_future = torch.randn(batch_size, pred_len, c_out, device=device)
    return x_enc, x_mark_enc, x_dec, x_mark_dec, y_future


def build_model_config():
    # Generic setup example: 40 covariates (10*4), 4 targets.
    # Targets are sourced from encoder feature indices 0..3 for de-normalization.
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="custom_tft_generic_40x4",
        seq_len=30,
        label_len=16,
        pred_len=4,
        enc_in=40,
        dec_in=4,
        c_out=4,
        d_model=128,
        n_heads=8,
        dropout=0.1,
        embed="timeF",
        freq="h",
        # Required for unregistered datasets (no fallbacks allowed)
        tft_observed_pos=list(range(40)),
        tft_static_pos=[],
        tft_target_pos=[0, 1, 2, 3],
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_model_config()

    model = Model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    known_len = 4  # from embed=timeF and freq='h'

    # Build a tiny in-memory dataset to demonstrate training wiring.
    all_tensors = [
        make_dummy_batch(
            batch_size=1,
            seq_len=cfg.seq_len,
            label_len=cfg.label_len,
            pred_len=cfg.pred_len,
            enc_in=cfg.enc_in,
            c_out=cfg.c_out,
            known_len=known_len,
            device=torch.device("cpu"),
        )
        for _ in range(64)
    ]

    x_enc = torch.cat([item[0] for item in all_tensors], dim=0)
    x_mark_enc = torch.cat([item[1] for item in all_tensors], dim=0)
    x_dec = torch.cat([item[2] for item in all_tensors], dim=0)
    x_mark_dec = torch.cat([item[3] for item in all_tensors], dim=0)
    y_future = torch.cat([item[4] for item in all_tensors], dim=0)

    loader = DataLoader(TensorDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y_future), batch_size=8, shuffle=True)

    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for bx, bxm, bd, bdm, by in loader:
            bx = bx.to(device)
            bxm = bxm.to(device)
            bd = bd.to(device)
            bdm = bdm.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            out_full = model(bx, bxm, bd, bdm)
            pred = out_full[:, -cfg.pred_len:, :]
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss={running_loss / len(loader):.6f}")

    # Interpretation example on one mini-batch.
    model.eval()
    with torch.no_grad():
        bx, bxm, bd, bdm, _ = next(iter(loader))
        payload = model(
            bx.to(device),
            bxm.to(device),
            bd.to(device),
            bdm.to(device),
            return_interpretation=True,
        )
        print("predictions_full shape:", tuple(payload["predictions_full"].shape))
        print("predictions shape:", tuple(payload["predictions"].shape))
        print("attention_weights shape:", tuple(payload["attention_weights"].shape))
        print("history_vsn_weights shape:", tuple(payload["history_vsn_weights"].shape))


if __name__ == "__main__":
    main()
