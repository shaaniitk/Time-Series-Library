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
    parser = argparse.ArgumentParser(description='Targeted ablation for TFT cross-attention, RevIN, and timestep regime routing.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seeds', type=str, default='7,13,29')
    parser.add_argument('--overfit-steps', type=int, default=24)
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()


def parse_seed_list(seed_text):
    return [int(item.strip()) for item in seed_text.split(',') if item.strip()]


def build_experiments():
    return [
        {'name': 'baseline', 'cross': False, 'revin': False, 'regime': False},
        {'name': 'cross_only', 'cross': True, 'revin': False, 'regime': False},
        {'name': 'revin_only', 'cross': False, 'revin': True, 'regime': False},
        {'name': 'regime_only', 'cross': False, 'revin': False, 'regime': True},
        {'name': 'cross_revin', 'cross': True, 'revin': True, 'regime': False},
        {'name': 'cross_regime', 'cross': True, 'revin': False, 'regime': True},
        {'name': 'revin_regime', 'cross': False, 'revin': True, 'regime': True},
        {'name': 'all_three', 'cross': True, 'revin': True, 'regime': True},
    ]


def main():
    args = parse_args()
    device = resolve_device(args.device)
    seeds = parse_seed_list(args.seeds)
    if not seeds:
        seeds = [42]

    cfg = build_base_config()
    cfg.tft_use_lag_attention = False
    cfg.tft_use_higher_order = False
    cfg.tft_dual_attention_fusion = False
    cfg.tft_full_attention = True
    cfg.tft_vsn_residual_bypass = True
    cfg.tft_cross_variable_mixing = False
    cfg.tft_cross_attention_type = 'interpretable'
    cfg.tft_use_explicit_cross_attention = False
    cfg.tft_use_revin = False
    cfg.tft_revin_affine = True
    cfg.tft_use_regime_moe = False

    epochs = args.epochs
    samples = args.samples
    batch_size = args.batch_size
    overfit_steps = args.overfit_steps
    experiments = build_experiments()
    if args.quick:
        cfg.d_model = 32
        cfg.n_heads = 4
        epochs = min(epochs, 2)
        samples = min(samples, 16)
        batch_size = min(batch_size, 4)
        overfit_steps = min(overfit_steps, 12)
        seeds = seeds[:1]

    set_seed(seeds[0])
    dataset = make_learnable_dataset(cfg, n_samples=samples, device='cpu')

    results = []
    for exp_cfg in experiments:
        cfg_result = []
        for seed in seeds:
            toggles = {
                'full_attention': True,
                'vsn_residual_bypass': True,
                'graph': False,
                'lag': False,
                'interaction': False,
                'moe': exp_cfg['regime'],
            }
            run_cfg = argparse.Namespace(**vars(cfg))
            run_cfg.tft_use_explicit_cross_attention = exp_cfg['cross']
            run_cfg.tft_cross_attention_type = 'interpretable'
            run_cfg.tft_use_revin = exp_cfg['revin']
            run_cfg.tft_use_regime_moe = exp_cfg['regime']
            result = run_experiment(
                base_cfg=run_cfg,
                dataset=dataset,
                toggles=toggles,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=args.lr,
                device=device,
            )
            cfg_result.append(result)
        overfit_cfg = argparse.Namespace(**vars(cfg))
        overfit_cfg.tft_use_explicit_cross_attention = exp_cfg['cross']
        overfit_cfg.tft_cross_attention_type = 'interpretable'
        overfit_cfg.tft_use_revin = exp_cfg['revin']
        overfit_cfg.tft_use_regime_moe = exp_cfg['regime']
        overfit = overfit_single_batch_check(
            base_cfg=overfit_cfg,
            dataset=dataset,
            toggles={
                'full_attention': True,
                'vsn_residual_bypass': True,
                'graph': False,
                'lag': False,
                'interaction': False,
                'moe': exp_cfg['regime'],
            },
            seed=seeds[0],
            overfit_steps=overfit_steps,
            lr=args.lr,
            device=device,
        )
        final_losses = [item['final_loss'] for item in cfg_result]
        results.append({
            **exp_cfg,
            'mean_final_loss': statistics.fmean(final_losses),
            'std_final_loss': statistics.pstdev(final_losses) if len(final_losses) > 1 else 0.0,
            'seed_losses': final_losses,
            'overfit': overfit,
        })

    results = sorted(results, key=lambda item: item['mean_final_loss'])
    print('Ablation: TFT cross-attention + RevIN + timestep regime routing')
    print(f'Dataset: harsh synthetic target | samples={samples} | epochs={epochs} | batch_size={batch_size}')
    print(f'Seeds: {seeds}')
    print('-' * 120)
    print(f"{'rank':<6}{'name':<18}{'cross':<10}{'revin':<10}{'regime':<10}{'mean_loss':<14}{'std_loss':<14}{'overfit_ratio':<16}{'overfit_pass':<14}{'seed_losses'}")
    for idx, item in enumerate(results, start=1):
        seed_losses = ', '.join(f"{value:.6f}" for value in item['seed_losses'])
        print(
            f"{idx:<6}{item['name']:<18}{str(item['cross']):<10}{str(item['revin']):<10}{str(item['regime']):<10}"
            f"{item['mean_final_loss']:<14.6f}{item['std_final_loss']:<14.6f}{item['overfit']['ratio']:<16.6f}"
            f"{('yes' if item['overfit']['passes'] else 'no'):<14}{seed_losses}"
        )
    print('-' * 120)
    print(f"Best config by mean loss => name={results[0]['name']}, mean_final_loss={results[0]['mean_final_loss']:.6f}")
    print(json.dumps(results[0], indent=2))


if __name__ == '__main__':
    main()