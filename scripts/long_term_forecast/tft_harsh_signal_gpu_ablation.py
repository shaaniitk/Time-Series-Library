import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.long_term_forecast.tft_ablation_full_attention_vs_vsn_bypass import (
    build_base_config,
    build_experiments,
    make_learnable_dataset,
    overfit_single_batch_check,
    resolve_device,
    run_experiment,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the harsher TFT ablation on a phase-coupled, regime-switched synthetic signal."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default="7,13,29,43,71")
    parser.add_argument("--overfit-steps", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/tft_harsh_signal_ablation")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--quick", action="store_true", help="Run a smaller version for wiring checks.")
    return parser.parse_args()


def parse_seed_list(seed_text):
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def _result_gap(best_loss, other_loss):
    return 100.0 * (other_loss - best_loss) / max(abs(best_loss), 1e-12)


def _interpret_top_result(results):
    best = results[0]
    baseline = next((item for item in results if item["name"] == "baseline"), None)
    notes = []

    if baseline is not None:
        gap_vs_baseline = _result_gap(best["mean_final_loss"], baseline["mean_final_loss"])
        notes.append(
            f"Best-vs-baseline improvement: {gap_vs_baseline:.2f}% lower mean loss."
        )

    if best["name"] == "all_upgrades":
        notes.append("All upgrades won jointly; the harder signal likely rewards complementary modeling capacity.")
    elif best["name"] == "graph_only":
        notes.append("Graph-only winning suggests cross-variable dependency structure is the main bottleneck.")
    elif best["name"] == "lag_only":
        notes.append("Lag-only winning suggests multi-scale temporal alignment is the main missing inductive bias.")
    elif best["name"] == "interaction_only":
        notes.append("Interaction-only winning suggests higher-order covariate-target interactions dominate this signal.")
    elif best["name"] == "moe_only":
        notes.append("MoE-only winning suggests regime switching is strong enough that conditional specialization helps immediately.")
    elif best["name"] == "full_off_bypass_off":
        notes.append("Turning off full attention and VSN bypass winning usually indicates the richer stack is under-optimized at the current budget.")

    unstable = [item["name"] for item in results if item["std_final_loss"] > 0.01]
    if unstable:
        notes.append("High seed variance: " + ", ".join(unstable))

    hard_to_fit = [item["name"] for item in results if item["overfit"]["ratio"] >= 0.60]
    if hard_to_fit:
        notes.append("Configs failing the overfit sanity threshold: " + ", ".join(hard_to_fit))

    return notes


def _markdown_summary(run_payload):
    lines = [
        "# TFT Harsh-Signal Ablation Summary",
        "",
        f"Device: {run_payload['device']}",
        f"Samples: {run_payload['samples']} | Epochs: {run_payload['epochs']} | Batch size: {run_payload['batch_size']}",
        f"Seeds: {run_payload['seeds']}",
        "",
        "Signal characteristics:",
        "- Low-, medium-, and high-frequency waves with time-varying phase velocities.",
        "- Target depends on phase differences and ratios of phase velocities.",
        "- High-frequency bursts are regime-switched using phase-coupled gates.",
        "",
        "## Ranking",
        "",
        "| Rank | Name | Mean Loss | Std Loss | Overfit Ratio | Overfit Pass |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, item in enumerate(run_payload["results"], start=1):
        lines.append(
            f"| {idx} | {item['name']} | {item['mean_final_loss']:.6f} | {item['std_final_loss']:.6f} | {item['overfit']['ratio']:.6f} | {'yes' if item['overfit']['passes'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Interpretation Notes",
        "",
    ])
    lines.extend([f"- {note}" for note in run_payload["interpretation_notes"]])
    lines.extend([
        "",
        "## Best Config",
        "",
        json.dumps(run_payload["best_result"], indent=2),
        "",
    ])
    return "\n".join(lines)


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
        epochs = min(epochs, 2)
        samples = min(samples, 16)
        batch_size = min(batch_size, 4)
        overfit_steps = min(overfit_steps, 12)
        seeds = seeds[:1]

    set_seed(seeds[0])
    dataset = make_learnable_dataset(cfg, n_samples=samples, device="cpu")

    results = []
    for toggles in experiments:
        per_seed = []
        for seed in seeds:
            per_seed.append(
                run_experiment(
                    base_cfg=cfg,
                    dataset=dataset,
                    toggles=toggles,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=args.lr,
                    device=device,
                )
            )

        final_losses = [item["final_loss"] for item in per_seed]
        results.append(
            {
                **toggles,
                "mean_final_loss": statistics.fmean(final_losses),
                "std_final_loss": statistics.pstdev(final_losses) if len(final_losses) > 1 else 0.0,
                "seed_losses": final_losses,
                "overfit": overfit_single_batch_check(
                    base_cfg=cfg,
                    dataset=dataset,
                    toggles=toggles,
                    seed=seeds[0],
                    overfit_steps=overfit_steps,
                    lr=args.lr,
                    device=device,
                ),
            }
        )

    results = sorted(results, key=lambda item: item["mean_final_loss"])
    best = results[0]
    interpretation_notes = _interpret_top_result(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"tft_harsh_signal_ablation_{timestamp}{tag_suffix}.json"
    md_path = output_dir / f"tft_harsh_signal_ablation_{timestamp}{tag_suffix}.md"

    run_payload = {
        "device": str(device),
        "epochs": epochs,
        "samples": samples,
        "batch_size": batch_size,
        "seeds": seeds,
        "learning_rate": args.lr,
        "overfit_steps": overfit_steps,
        "signal_description": {
            "low_frequency": "slow wave with drifting phase velocity",
            "medium_frequency": "mid-scale wave with stronger phase drift",
            "high_frequency": "fast wave with regime-switched burst gating",
            "dependencies": [
                "phase differences between waves",
                "ratios of phase velocities",
                "regime-switched burst gating tied to phase alignment",
            ],
        },
        "results": results,
        "best_result": best,
        "interpretation_notes": interpretation_notes,
    }

    json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(run_payload), encoding="utf-8")

    print("TFT harsh-signal ablation completed")
    print(f"Device: {device}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    print("Top 3 configs:")
    for idx, item in enumerate(results[:3], start=1):
        print(
            f"  {idx}. {item['name']} | mean_loss={item['mean_final_loss']:.6f} | std={item['std_final_loss']:.6f} | "
            f"overfit_ratio={item['overfit']['ratio']:.6f} | overfit_pass={'yes' if item['overfit']['passes'] else 'no'}"
        )
    print("Interpretation notes:")
    for note in interpretation_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    main()