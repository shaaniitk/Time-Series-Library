# TFT Harsh-Signal Ablation

Run [scripts/long_term_forecast/tft_harsh_signal_gpu_ablation.py](scripts/long_term_forecast/tft_harsh_signal_gpu_ablation.py) directly on the target machine. It generates a hard synthetic forecasting task where:

- a low-frequency wave, a medium-frequency wave, and a high-frequency wave all have drifting phase velocities
- the target depends on phase differences between waves
- the target depends on ratios of phase velocities
- the high-frequency component is regime-switched, so burst intensity changes with phase alignment

Recommended direct run on GPU:

```bash
./.venv/Scripts/python.exe scripts/long_term_forecast/tft_harsh_signal_gpu_ablation.py --device cuda --epochs 8 --samples 128 --batch-size 16 --seeds 7,13,29,43,71 --overfit-steps 32
```

Quick wiring run:

```bash
./.venv/Scripts/python.exe scripts/long_term_forecast/tft_harsh_signal_gpu_ablation.py --quick --device cpu
```

What the script writes:

- a JSON file in `results/tft_harsh_signal_ablation/`
- a Markdown summary in `results/tft_harsh_signal_ablation/`

What another agent should inspect in the JSON:

- `results[*].mean_final_loss`: primary ranking metric, lower is better
- `results[*].std_final_loss`: seed stability, lower is better
- `results[*].overfit.ratio`: optimization sanity check
- `best_result`: best-ranked configuration
- `interpretation_notes`: auto-generated high-level reading of the run

Interpretation rules for another agent:

- If `all_upgrades` wins by a clear margin, the richer stack is helping jointly on phase-coupled nonlinear structure.
- If `graph_only` wins, cross-variable dependency modeling is the main gain.
- If `lag_only` wins, multi-scale temporal alignment matters more than cross-variable modeling.
- If `interaction_only` wins, higher-order covariate-target coupling dominates.
- If `moe_only` wins, regime switching is strong enough that conditional specialization matters.
- If `full_off_bypass_off` wins, the advanced stack is probably under-trained or over-regularized for the chosen budget.
- If the best loss beats baseline by less than about 3%, treat it as weak evidence and look at `std_final_loss` before claiming a winner.
- If `overfit.ratio >= 0.60`, that configuration is not fitting even a single batch well enough; interpret ranking cautiously.

Minimum information another agent should report back:

- machine type and whether the run used CPU or GPU
- exact command used
- top 3 configs by `mean_final_loss`
- baseline loss and best-vs-baseline percentage gap
- which configs failed the overfit sanity threshold
- whether seed variance changes the conclusion