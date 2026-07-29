# Current Status

> This is the first file an agent reads when resuming the project.
>
> Updated: 2026-07-29, 21:07 Asia/Kolkata.

## Project State

```text
phase: PLANNING / THEORY DISCUSSION
implementation_authorized: no
financial_astrology_code_written: no
data_sample_received: no
hypothesis_registry_frozen: no
locked_holdout_opened: no
```

The user explicitly asked to inspect the current TFT run first and to discuss the
financial-astrology design before implementation. Do not begin loader, feature,
model, or experiment code until that state changes in this file and the tracker.

## Live Native-TFT Matrix

The production advanced-feature matrix launched by:

```bash
bash scripts/long_term_forecast/ETT_script/TFT_ETTh1_OT_feature_matrix.sh
```

was inspected read-only and was still alive. The active case was:

```text
tft_ot_p24_xattn_interp
explicit interpretable cross-attention
```

Its checkpoint advanced at 20:52:09, showing continued validation improvement.
At 21:07 the parent process was still active and using the accelerator; fresh
short-lived child processes were data-loader/evaluation workers, not duplicate
matrix launches. The exact epoch is not recoverable non-intrusively because
output is attached to the user's terminal. Do not signal, kill, restart, or
attach a debugger to it.

Completed cases at the last inspection:

| Rank by MSE | Case | MSE | MAE | Relative to baseline |
|---:|---|---:|---:|---|
| 1 | quantile-only | 0.03435775 | 0.14262331 | MSE -3.73%, MAE -2.97% |
| 2 | ALiBi | 0.03467850 | 0.14418337 | MSE -2.83%, MAE -1.91% |
| 3 | baseline | 0.03568881 | 0.14699268 | reference |
| 4 | SDPA | 0.03568881 | 0.14699268 | bit-for-bit baseline parity |
| 5 | joint quantile | 0.03587884 | 0.14622945 | MSE +0.53%, MAE -0.52% |

Quantile diagnostics:

| Case | Pinball | Nominal interval | Coverage | Width | Crossing |
|---|---:|---:|---:|---:|---:|
| quantile-only | 0.04626653 | 80% | 71.91% | 0.39879317 | 0 |
| joint quantile | 0.04921214 | 80% | 68.04% | 0.36935251 | 0 |

Both intervals under-cover and would need calibration. Quantile-only is
nevertheless the current best point forecaster in this matrix.

Eight cases were still queued after cross-attention:

1. lag attention;
2. sparse cross-variable mixing;
3. learned FFT;
4. higher-order interactions;
5. regime MoE;
6. temporal compression;
7. covariate reattention;
8. experimental-full profile.

## Architecture Conclusion Reached

The earlier `seq_len=128` proposal is withdrawn as a description of the complete
astrological memory.

The current design uses:

- a local market sequence, starting at 252 trading days;
- exact target-date known astronomical state;
- fast daily and medium event/weekly representations;
- slow recursive/calendar-time memory and optional monthly pooling;
- phase and relative-phase encodings for cycles longer than the market record.

This represents Saturn and outer-planet state without claiming that the available
NIFTY record contains enough independent cycles to identify their full effects.

## Immediate Next Actions

While the TFT matrix runs:

1. continue theory/data discussion only;
2. ask for a representative merged data sample and generator when the user is
   ready;
3. settle the disputed tradition choices in `HYPOTHESIS_REGISTRY.md`;
4. update this file with the final TFT matrix scorecard when the process exits.

After the matrix and discussion:

1. execute `FA-DATA-001`;
2. freeze `FA-THEORY-001`;
3. execute the leakage-only loader slice `FA-LOAD-001`;
4. build non-astrological baselines before adding any planet feature.

## Resume Safety Check

Before doing any work, an agent must run read-only checks equivalent to:

```bash
ps -eo pid,ppid,etime,stat,cmd | rg 'TFT_ETTh1_OT_feature_matrix|run.py'
git status --short
```

Existing modified files belong to the user and prior sessions. Preserve them.
