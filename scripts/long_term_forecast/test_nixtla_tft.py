import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TFT_Nixtla import Model

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dummy_batch(batch_size, seq_len, label_len, pred_len, enc_in, c_out, known_len, device):
    x_enc = torch.randn(batch_size, seq_len, enc_in, device=device)
    x_mark_enc = torch.randn(batch_size, seq_len, known_len, device=device)
    x_dec = torch.randn(batch_size, label_len + pred_len, c_out, device=device)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, known_len, device=device)
    y_future = torch.randn(batch_size, pred_len, c_out, device=device)
    return x_enc, x_mark_enc, x_dec, x_mark_dec, y_future

def build_model_config():
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=30,
        label_len=16,
        pred_len=4,
        enc_in=4,
        dec_in=4,
        c_out=4,
        d_model=64,
        n_heads=8,
        dropout=0.1,
        embed="timeF",
        freq="h",
        batch_size=4
    )

def main():
    set_seed(42)
    device = torch.device("cpu")
    cfg = build_model_config()

    epochs = 2
    samples = 16
    batch_size = cfg.batch_size

    model = Model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    known_len = model.time_features_dim

    all_tensors = [
        make_dummy_batch(
            batch_size=1,
            seq_len=cfg.seq_len,
            label_len=cfg.label_len,
            pred_len=cfg.pred_len,
            enc_in=cfg.enc_in,
            c_out=cfg.c_out,
            known_len=known_len,
            device=device,
        )
        for _ in range(samples)
    ]

    x_enc = torch.cat([item[0] for item in all_tensors], dim=0)
    x_mark_enc = torch.cat([item[1] for item in all_tensors], dim=0)
    x_dec = torch.cat([item[2] for item in all_tensors], dim=0)
    x_mark_dec = torch.cat([item[3] for item in all_tensors], dim=0)
    y_future = torch.cat([item[4] for item in all_tensors], dim=0)

    loader = DataLoader(TensorDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y_future), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for bx, bxm, bd, bdm, by in loader:
            optimizer.zero_grad()
            out_full = model(bx, bxm, bd, bdm)
            loss = criterion(out_full, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss={running_loss / len(loader):.6f}")

if __name__ == "__main__":
    main()
