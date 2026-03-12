import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TemporalFusionTransformer import Model


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation for TFT full-attention and VSN residual bypass with stronger bug-detection checks."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default="7,13,29,43,71")
    parser.add_argument("--overfit-steps", type=int, default=60)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quick", action="store_true", help="Run a very fast version for smoke tests.")
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_seed_list(seed_text: str):
    parts = [x.strip() for x in seed_text.split(",") if x.strip()]
    return [int(x) for x in parts]


def build_base_config():
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="custom_tft_generic_40x4",
        seq_len=30,
        label_len=16,
        pred_len=4,
        enc_in=40,
        dec_in=4,
        c_out=4,
        d_model=64,
        n_heads=8,
        dropout=0.1,
        embed="timeF",
        freq="h",
        e_layers=1,
        tft_use_swiglu=True,
        tft_cross_variable_mixing=True,
        tft_allow_custom_known=True,
        tft_known_len=40,
        tft_known_max_channels=128,
        tft_observed_pos=list(range(40)),
        tft_static_pos=[],
        tft_target_pos=[0, 1, 2, 3],
    )


def make_learnable_dataset(cfg, n_samples=128, noise_std=0.01, device=torch.device("cpu")):
    # Build a target with deterministic structure from known future covariates + recent history.
    # This is intentionally learnable and much better for diagnosing architectural bugs than random labels.
    x_enc = torch.randn(n_samples, cfg.seq_len, cfg.enc_in, device=device)
    x_mark_enc = torch.randn(n_samples, cfg.seq_len, cfg.tft_known_len, device=device)
    x_dec = torch.randn(n_samples, cfg.label_len + cfg.pred_len, cfg.c_out, device=device)
    x_mark_dec = torch.randn(n_samples, cfg.label_len + cfg.pred_len, cfg.tft_known_len, device=device)

    future_known = x_mark_dec[:, -cfg.pred_len:, :]  # [N, pred_len, known_len]
    recent_targets = x_enc[:, -cfg.pred_len:, :cfg.c_out]  # [N, pred_len, c_out]
    known_weights = torch.randn(cfg.tft_known_len, cfg.c_out, device=device)
    known_signal = torch.matmul(future_known, known_weights)  # [N, pred_len, c_out]
    nonlinear_signal = torch.sin(known_signal)

    y_future = 0.55 * recent_targets + 0.30 * known_signal + 0.15 * nonlinear_signal
    y_future = y_future + noise_std * torch.randn_like(y_future)
    return TensorDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y_future)


def run_experiment(base_cfg, dataset, full_attention: bool, residual_bypass: bool, seed: int, epochs=8, batch_size=16, lr=1e-3, device=None):
    set_seed(seed)
    cfg = SimpleNamespace(**vars(base_cfg))
    cfg.tft_full_attention = full_attention
    cfg.tft_vsn_residual_bypass = residual_bypass
    cfg.tft_dual_attention_fusion = False

    model = Model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    epoch_losses = []
    for _ in range(epochs):
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

        epoch_losses.append(running_loss / len(loader))

    return {
        "full_attention": full_attention,
        "vsn_residual_bypass": residual_bypass,
        "seed": seed,
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1],
    }


def overfit_single_batch_check(base_cfg, dataset, full_attention: bool, residual_bypass: bool, seed: int, overfit_steps: int, lr: float, device):
    set_seed(seed)
    cfg = SimpleNamespace(**vars(base_cfg))
    cfg.tft_full_attention = full_attention
    cfg.tft_vsn_residual_bypass = residual_bypass
    cfg.tft_dual_attention_fusion = False

    model = Model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False)
    bx, bxm, bd, bdm, by = next(iter(loader))
    bx = bx.to(device)
    bxm = bxm.to(device)
    bd = bd.to(device)
    bdm = bdm.to(device)
    by = by.to(device)

    model.train()
    start_loss = None
    end_loss = None
    for step in range(overfit_steps):
        optimizer.zero_grad()
        out_full = model(bx, bxm, bd, bdm)
        pred = out_full[:, -cfg.pred_len:, :]
        loss = criterion(pred, by)
        if step == 0:
            start_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        end_loss = float(loss.item())

    ratio = end_loss / max(start_loss, 1e-12)
    return {
        "start_loss": start_loss,
        "end_loss": end_loss,
        "ratio": ratio,
        "passes": ratio < 0.60,
    }


def format_bool(v: bool) -> str:
    return "on" if v else "off"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    seeds = parse_seed_list(args.seeds)
    if not seeds:
        seeds = [42]

    cfg = build_base_config()

    epochs = args.epochs
    samples = args.samples
    batch_size = args.batch_size
    overfit_steps = args.overfit_steps
    if args.quick:
        cfg.d_model = 32
        cfg.n_heads = 4
        epochs = 1
        samples = min(samples, 16)
        batch_size = min(batch_size, 8)
        overfit_steps = min(overfit_steps, 8)
        seeds = seeds[:1]

    set_seed(seeds[0])

    # Keep one fixed learnable dataset across all runs for fair and meaningful comparison.
    dataset = make_learnable_dataset(cfg, n_samples=samples, device=torch.device("cpu"))

    experiments = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]

    aggregated_results = []
    for full_attention, residual_bypass in experiments:
        per_seed = []
        for seed in seeds:
            result = run_experiment(
                base_cfg=cfg,
                dataset=dataset,
                full_attention=full_attention,
                residual_bypass=residual_bypass,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=args.lr,
                device=device,
            )
            per_seed.append(result)

        final_losses = [x["final_loss"] for x in per_seed]
        mean_loss = statistics.fmean(final_losses)
        std_loss = statistics.pstdev(final_losses) if len(final_losses) > 1 else 0.0
        overfit = overfit_single_batch_check(
            base_cfg=cfg,
            dataset=dataset,
            full_attention=full_attention,
            residual_bypass=residual_bypass,
            seed=seeds[0] if seeds else 42,
            overfit_steps=overfit_steps,
            lr=args.lr,
            device=device,
        )

        aggregated_results.append(
            {
                "full_attention": full_attention,
                "vsn_residual_bypass": residual_bypass,
                "mean_final_loss": mean_loss,
                "std_final_loss": std_loss,
                "seed_losses": final_losses,
                "overfit": overfit,
            }
        )

    aggregated_results = sorted(aggregated_results, key=lambda x: x["mean_final_loss"])

    print("Ablation: full attention x VSN residual bypass")
    print(f"Dataset: learnable synthetic target | samples={samples} | epochs={epochs} | batch_size={batch_size}")
    print(f"Seeds: {seeds}")
    print("-" * 120)
    print(
        f"{'rank':<6}{'full_attn':<12}{'vsn_bypass':<12}{'mean_loss':<14}{'std_loss':<14}"
        f"{'overfit_ratio':<16}{'overfit_pass':<14}{'seed_losses'}"
    )
    for idx, item in enumerate(aggregated_results, start=1):
        seed_losses = ", ".join(f"{v:.6f}" for v in item["seed_losses"])
        overfit_ratio = item["overfit"]["ratio"]
        overfit_pass = "yes" if item["overfit"]["passes"] else "no"
        print(
            f"{idx:<6}{format_bool(item['full_attention']):<12}{format_bool(item['vsn_residual_bypass']):<12}"
            f"{item['mean_final_loss']:<14.6f}{item['std_final_loss']:<14.6f}"
            f"{overfit_ratio:<16.6f}{overfit_pass:<14}{seed_losses}"
        )

    best = aggregated_results[0]
    print("-" * 120)
    print(
        "Best config by mean loss => "
        f"full_attention={format_bool(best['full_attention'])}, "
        f"vsn_residual_bypass={format_bool(best['vsn_residual_bypass'])}, "
        f"mean_final_loss={best['mean_final_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
