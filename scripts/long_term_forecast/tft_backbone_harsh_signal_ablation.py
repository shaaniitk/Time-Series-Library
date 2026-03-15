import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.long_term_forecast.tft_ablation_full_attention_vs_vsn_bypass import (
    build_base_config,
    make_learnable_dataset,
    overfit_single_batch_check,
    resolve_device,
    run_experiment,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation for TFT temporal backbones on the harsh synthetic signal.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default="7,13,29")
    parser.add_argument("--overfit-steps", type=int, default=24)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_text):
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def build_experiments():
    return [
        {"name": "lstm", "temporal_backbone": "lstm"},
        {"name": "gated_tcn", "temporal_backbone": "gated_tcn"},
        {"name": "hybrid_tcn_lstm", "temporal_backbone": "hybrid_tcn_lstm"},
    ]


def _run_backbone_experiment(base_cfg, dataset, exp_cfg, seed, epochs, batch_size, lr, device):
    run_cfg = argparse.Namespace(**vars(base_cfg))
    run_cfg.tft_temporal_backbone = exp_cfg["temporal_backbone"]
    run_cfg.tft_temporal_backbone_layers = 2
    run_cfg.tft_temporal_kernel_size = 3
    run_cfg.tft_temporal_hidden_size = run_cfg.d_model
    return run_experiment(
        base_cfg=run_cfg,
        dataset=dataset,
        toggles={
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": True,
            "lag": True,
            "interaction": True,
            "moe": True,
        },
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )


def _overfit_backbone(base_cfg, dataset, exp_cfg, seed, overfit_steps, lr, device):
    run_cfg = argparse.Namespace(**vars(base_cfg))
    run_cfg.tft_temporal_backbone = exp_cfg["temporal_backbone"]
    run_cfg.tft_temporal_backbone_layers = 2
    run_cfg.tft_temporal_kernel_size = 3
    run_cfg.tft_temporal_hidden_size = run_cfg.d_model
    return overfit_single_batch_check(
        base_cfg=run_cfg,
        dataset=dataset,
        toggles={
            "full_attention": True,
            "vsn_residual_bypass": True,
            "graph": True,
            "lag": True,
            "interaction": True,
            "moe": True,
        },
        seed=seed,
        overfit_steps=overfit_steps,
        lr=lr,
        device=device,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    seeds = parse_seed_list(args.seeds) or [42]
    cfg = build_base_config()
    cfg.tft_use_lag_attention = True
    cfg.tft_use_higher_order = True
    cfg.tft_use_regime_moe = True
    cfg.tft_dual_attention_fusion = True
    cfg.tft_use_explicit_cross_attention = True
    cfg.tft_cross_attention_type = "full"
    cfg.tft_use_revin = True
    cfg.tft_attention_backend = "sdpa"
    cfg.tft_attention_position_bias = "rope"

    epochs = args.epochs
    samples = args.samples
    batch_size = args.batch_size
    overfit_steps = args.overfit_steps
    if args.quick:
        cfg.d_model = 32
        cfg.n_heads = 4
        epochs = min(epochs, 2)
        samples = min(samples, 16)
        batch_size = min(batch_size, 4)
        overfit_steps = min(overfit_steps, 12)
        seeds = seeds[:1]

    set_seed(seeds[0])
    dataset = make_learnable_dataset(cfg, n_samples=samples, device="cpu")

    results = []
    for exp_cfg in build_experiments():
        per_seed = [
            _run_backbone_experiment(cfg, dataset, exp_cfg, seed, epochs, batch_size, args.lr, device)
            for seed in seeds
        ]
        final_losses = [item["final_loss"] for item in per_seed]
        results.append(
            {
                **exp_cfg,
                "mean_final_loss": statistics.fmean(final_losses),
                "std_final_loss": statistics.pstdev(final_losses) if len(final_losses) > 1 else 0.0,
                "seed_losses": final_losses,
                "overfit": _overfit_backbone(cfg, dataset, exp_cfg, seeds[0], overfit_steps, args.lr, device),
            }
        )

    results = sorted(results, key=lambda item: item["mean_final_loss"])
    print("Ablation: TFT temporal backbone on harsh synthetic signal")
    print(f"Dataset: harsh synthetic target | samples={samples} | epochs={epochs} | batch_size={batch_size}")
    print(f"Seeds: {seeds}")
    print("-" * 100)
    print(f"{'rank':<6}{'name':<20}{'mean_loss':<14}{'std_loss':<14}{'overfit_ratio':<16}{'overfit_pass':<14}{'seed_losses'}")
    for idx, item in enumerate(results, start=1):
        seed_losses = ", ".join(f"{value:.6f}" for value in item["seed_losses"])
        print(
            f"{idx:<6}{item['name']:<20}{item['mean_final_loss']:<14.6f}{item['std_final_loss']:<14.6f}"
            f"{item['overfit']['ratio']:<16.6f}{('yes' if item['overfit']['passes'] else 'no'):<14}{seed_losses}"
        )
    print("-" * 100)
    print(f"Best config by mean loss => name={results[0]['name']}, mean_final_loss={results[0]['mean_final_loss']:.6f}")
    print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()