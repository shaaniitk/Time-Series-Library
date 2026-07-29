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
from utils.tft_synthetic import make_multiscale_tft_dataset


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation for TFT attention, VSN bypass, and advanced dependency-modeling upgrades."
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
        tft_known_feature_names=[f"known_{i}" for i in range(40)],
        tft_use_lag_attention=False,
        tft_lag_scales=[1, 2, 4],
        tft_use_higher_order=False,
        tft_interaction_order=2,
        tft_interaction_rank=16,
        tft_use_regime_moe=False,
        tft_num_regimes=3,
        tft_num_moe_experts=4,
        tft_moe_top_k=2,
        tft_moe_hidden_size=96,
        tft_moe_noise_epsilon=1e-2,
        tft_moe_aux_loss_coeff=0.05,
        tft_payload_stack_layers=False,
        tft_observed_pos=list(range(40)),
        tft_static_pos=[],
        tft_target_pos=[0, 1, 2, 3],
    )


def make_learnable_dataset(cfg, n_samples=128, noise_std=0.01, device=torch.device("cpu")):
    return make_multiscale_tft_dataset(
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        c_out=cfg.c_out,
        known_len=cfg.tft_known_len,
        n_samples=n_samples,
        device=device,
        noise_std=noise_std,
    )


def run_experiment(base_cfg, dataset, toggles, seed: int, epochs=8, batch_size=16, lr=1e-3, device=None):
    set_seed(seed)
    cfg = SimpleNamespace(**vars(base_cfg))
    cfg.tft_full_attention = toggles["full_attention"]
    cfg.tft_vsn_residual_bypass = toggles["vsn_residual_bypass"]
    cfg.tft_cross_variable_mixing = toggles["graph"]
    cfg.tft_use_lag_attention = toggles["lag"]
    cfg.tft_use_higher_order = toggles["interaction"]
    cfg.tft_use_regime_moe = toggles["moe"]
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
            aux_loss = getattr(model, "last_moe_aux_loss", None)
            if torch.is_tensor(aux_loss):
                loss = loss + cfg.tft_moe_aux_loss_coeff * aux_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_losses.append(running_loss / len(loader))

    return {
        **toggles,
        "seed": seed,
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1],
    }


def overfit_single_batch_check(base_cfg, dataset, toggles, seed: int, overfit_steps: int, lr: float, device):
    set_seed(seed)
    cfg = SimpleNamespace(**vars(base_cfg))
    cfg.tft_full_attention = toggles["full_attention"]
    cfg.tft_vsn_residual_bypass = toggles["vsn_residual_bypass"]
    cfg.tft_cross_variable_mixing = toggles["graph"]
    cfg.tft_use_lag_attention = toggles["lag"]
    cfg.tft_use_higher_order = toggles["interaction"]
    cfg.tft_use_regime_moe = toggles["moe"]
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
        aux_loss = getattr(model, "last_moe_aux_loss", None)
        if torch.is_tensor(aux_loss):
            loss = loss + cfg.tft_moe_aux_loss_coeff * aux_loss
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


def build_experiments():
    return [
        {
            "name": "baseline",
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": False,
            "lag": False,
            "interaction": False,
            "moe": False,
        },
        {
            "name": "full_off_bypass_off",
            "full_attention": False,
            "vsn_residual_bypass": False,
            "graph": False,
            "lag": False,
            "interaction": False,
            "moe": False,
        },
        {
            "name": "graph_only",
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": True,
            "lag": False,
            "interaction": False,
            "moe": False,
        },
        {
            "name": "lag_only",
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": False,
            "lag": True,
            "interaction": False,
            "moe": False,
        },
        {
            "name": "interaction_only",
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": False,
            "lag": False,
            "interaction": True,
            "moe": False,
        },
        {
            "name": "moe_only",
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": False,
            "lag": False,
            "interaction": False,
            "moe": True,
        },
        {
            "name": "all_upgrades",
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": True,
            "lag": True,
            "interaction": True,
            "moe": True,
        },
    ]


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
    experiments = build_experiments()
    if args.quick:
        cfg.d_model = 32
        cfg.n_heads = 4
        epochs = 1
        samples = min(samples, 8)
        batch_size = min(batch_size, 4)
        overfit_steps = min(overfit_steps, 12)
        seeds = seeds[:1]
        experiments = [
            item for item in experiments
            if item["name"] in {"baseline", "graph_only", "interaction_only", "moe_only", "all_upgrades"}
        ]

    set_seed(seeds[0])

    # Keep one fixed learnable dataset across all runs for fair and meaningful comparison.
    dataset = make_learnable_dataset(cfg, n_samples=samples, device=torch.device("cpu"))

    aggregated_results = []
    for toggles in experiments:
        per_seed = []
        for seed in seeds:
            result = run_experiment(
                base_cfg=cfg,
                dataset=dataset,
                toggles=toggles,
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
            toggles=toggles,
            seed=seeds[0] if seeds else 42,
            overfit_steps=overfit_steps,
            lr=args.lr,
            device=device,
        )

        aggregated_results.append(
            {
                **toggles,
                "mean_final_loss": mean_loss,
                "std_final_loss": std_loss,
                "seed_losses": final_losses,
                "overfit": overfit,
            }
        )

    aggregated_results = sorted(aggregated_results, key=lambda x: x["mean_final_loss"])

    print("Ablation: TFT dependency-upgrade matrix")
    print(f"Dataset: learnable synthetic target | samples={samples} | epochs={epochs} | batch_size={batch_size}")
    print(f"Seeds: {seeds}")
    print("-" * 120)
    print(
        f"{'rank':<6}{'name':<20}{'full_attn':<12}{'vsn_bypass':<12}{'graph':<8}{'lag':<8}{'interact':<10}{'moe':<8}{'mean_loss':<14}{'std_loss':<14}"
        f"{'overfit_ratio':<16}{'overfit_pass':<14}{'seed_losses'}"
    )
    for idx, item in enumerate(aggregated_results, start=1):
        seed_losses = ", ".join(f"{v:.6f}" for v in item["seed_losses"])
        overfit_ratio = item["overfit"]["ratio"]
        overfit_pass = "yes" if item["overfit"]["passes"] else "no"
        print(
            f"{idx:<6}{item['name']:<20}{format_bool(item['full_attention']):<12}{format_bool(item['vsn_residual_bypass']):<12}"
            f"{format_bool(item['graph']):<8}{format_bool(item['lag']):<8}{format_bool(item['interaction']):<10}{format_bool(item['moe']):<8}"
            f"{item['mean_final_loss']:<14.6f}{item['std_final_loss']:<14.6f}"
            f"{overfit_ratio:<16.6f}{overfit_pass:<14}{seed_losses}"
        )

    best = aggregated_results[0]
    print("-" * 120)
    print(
        "Best config by mean loss => "
        f"name={best['name']}, "
        f"full_attention={format_bool(best['full_attention'])}, "
        f"vsn_residual_bypass={format_bool(best['vsn_residual_bypass'])}, "
        f"graph={format_bool(best['graph'])}, "
        f"lag={format_bool(best['lag'])}, "
        f"interaction={format_bool(best['interaction'])}, "
        f"moe={format_bool(best['moe'])}, "
        f"mean_final_loss={best['mean_final_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
