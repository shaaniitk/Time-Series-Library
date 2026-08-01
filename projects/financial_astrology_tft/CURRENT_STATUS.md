# Current Status

> This is the first file an agent reads when resuming the project.
>
> Updated: 2026-08-01, 14:05 Asia/Kolkata.

## Project State

```text
phase: NATIVE SEMANTIC HARDENING / DATA REMEDIATION
planning_authorized: yes
native_semantic_repairs_implemented: partial (TFT-SR00 and TFT-SR01 complete)
native_active_task: TFT-SR02
financial_astrology_code_written: no
financial_astrology_neural_training_authorized: no
data_sample_received: yes
data_audit_state: WAITING_EXTERNAL
hypothesis_registry_frozen: no
locked_holdout_opened: no
```

The detailed architecture and execution plan is authorized. The native matrix
has exited, `TFT-SR00`/`TFT-SR01` passed, and `TFT-SR02` is active. The supplied data audit found blocking
semantic defects and is waiting for raw OHLC plus generator provenance. Do not
launch a NIFTY neural run until `FA-TFT-SEM-001`, `FA-DATA-001`,
`FA-THEORY-001`, `FA-LEAK-001`, `FA-LOAD-001`, and the baseline gate are
complete.

## Completed Native-TFT Matrix

The production advanced-feature matrix launched by:

```bash
bash scripts/long_term_forecast/ETT_script/TFT_ETTh1_OT_feature_matrix.sh
```

finished at 23:22. No matching parent or `run.py` process remained at the
23:36 read-only check. The final case was:

```text
tft_ot_p24_experimental_profile
profile: experimental_full
```

All 14 result and checkpoint directories exist. The final case wrote
`metrics.npy`, `pred.npy`, `true.npy`, quantile outputs, and `checkpoint.pth`.
`TFT-SR00` froze 65 producer/script/data/result/checkpoint records in
[`legacy_v1_matrix_manifest.json`](../../metadata/tft/legacy_v1_matrix_manifest.json)
and added guarded v1/v2 checkpoint identities.

Completed point metrics:

| Case | MSE | MAE | MSE vs baseline | Semantic disposition |
|---|---:|---:|---:|---|
| quantile-only | 0.03435775 | 0.14262331 | -3.73% | Promising single seed; interval under-coverage remains |
| covariate reattention | 0.03445451 | 0.14307196 | -3.46% | Promising single seed; verify semantics and seeds |
| ALiBi | 0.03467850 | 0.14418337 | -2.83% | Promising single seed |
| regime MoE | 0.03547186 | 0.14576912 | -0.61% | Near neutral; not a default |
| baseline | 0.03568881 | 0.14699268 | reference | Reference only |
| SDPA | 0.03568881 | 0.14699268 | 0.00% | Exact baseline parity |
| joint quantile | 0.03587884 | 0.14622945 | +0.53% | Mixed; interval under-coverage |
| learned FFT | 0.03607463 | 0.14592449 | +1.08% | Selector semantics defective; result not dispositive |
| cross-attention | 0.03616716 | 0.14684366 | +1.34% | Semantic/neutrality repair required |
| higher-order | 0.03687086 | 0.14817350 | +3.31% | Latent polynomial is not named covariate interaction |
| lag attention | 0.03693667 | 0.14863680 | +3.50% | Shifted-prefix semantics do not equal causal lag response |
| temporal compression | 0.03730480 | 0.15012075 | +4.53% | Short-sequence/decompression semantics defective |
| sparse cross-mixing | 0.03858219 | 0.15343992 | +8.11% | Graph semantics/integration require repair |
| experimental profile | 0.04245717 | 0.16271803 | +18.96% | Worst case; stacked defective/non-neutral extensions are not a clean ablation |

Quantile-only coverage was 71.91% and joint-quantile coverage was 68.04% for a
nominal 80% interval; crossing was zero. These are useful diagnostics, not
calibrated production intervals.

## Post-Matrix Semantic Conclusion

The adverse advanced-feature results do **not** by themselves prove that every
idea is useless. Code audit found that several switches have a mismatch between
their label and their implemented scientific meaning. The canonical native gate
is now:

```text
FA-TFT-SEM-001
  -> TFT-SR00 legacy freeze/versioning
  -> TFT-SR01 reproducibility and paired initialization
  -> TFT-SR02 neutral extension and coordinate contract
  -> TFT-SR03 truthful FFT semantics
  -> TFT-SR04 cross-attention semantics
  -> TFT-SR05 lag/response semantics
  -> TFT-SR06 named interactions and per-feature VSN
  -> TFT-SR07 temporal compression semantics
  -> TFT-SR08 graph semantics
  -> TFT-SR09 semantic release gate
```

Acceptance is contract-driven: exact disabled parity, true no-op initialization,
known-answer synthetic fixtures, nonzero gradients, mask/prefix invariance,
deterministic paired seeds, and a single micro-run. Do not repeat the 36-hour
ETTh1 matrix merely to close this gate.

## Financial-Astrology Architecture Decision

`seq_len` is local market memory, not planetary orbital memory. The starting
primary configuration uses 252 trading sessions, a one-session stationary
target, and separate known-future planetary representations:

- exact state and motion over the decision-to-target interval;
- circular phase and relative phase, including retrograde/station state;
- fast and medium event clocks;
- low-capacity calendar-time response banks for prolonged effects;
- optional coarse slow-state summaries for Jupiter, Saturn, nodes, and outer
  planets;
- matched smooth/null ephemerides.

The first neural comparison freezes a market/calendar TFT, copies the same
checkpoint for all arms, and compares `disabled`, matched `null`, and real
planetary residuals. Generic FFT, graph, lag, compression, MoE, higher-order,
and cross-attention extensions remain off initially even after semantic repair;
they must earn entry through a named hypothesis.

## Supplied-Data Verdict

The source audit is recorded in [DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md).
The disposition is:

- retain the continuous ephemeris fields only provisionally, after timestamp,
  units, frame, and generator confirmation;
- reject every supplied `*_sign_sin/cos` column because all 12 families obey
  the same off-by-one/clipping rashi defect on all 18,251 rows;
- rebuild the market table from raw OHLC because at least 412
  value-fingerprinted rows are labelled one calendar day before their session;
- quarantine Shadbala and omit any claimed Hilbert arm;
- derive Ketu from one Rahu/node axis instead of learning duplicate numbers;
- replace `time_delta = calendar_gap / 5` with actual calendar and
  trading-session deltas.

After remediation, the first admissible planet slice is the 35 continuous
Sun-through-Saturn fields plus one two-dimensional mean-node longitude axis.

## Immediate Next Actions

1. complete active `TFT-SR02`: exact-zero residual adapters and physical
   time/feature-coordinate contracts;
2. obtain raw session-dated NIFTY OHLC and the exact PySwissEph generator plus
   convention manifest; do not relabel the derived CSV heuristically;
3. implement `TFT-SR02`–`TFT-SR08` in dependency order;
4. close `TFT-SR09` with focused tests and one deterministic micro-run;
5. complete corrected data/convention/leakage/loader work in parallel where
   dependencies allow;
6. run non-neural and market/calendar baselines;
7. begin the staged NIFTY comparisons in `IMPLEMENTATION_PLAN.md`.

## Resume Safety Check

Before doing work, run read-only checks equivalent to:

```bash
ps -eo pid,ppid,etime,stat,cmd | rg 'TFT_ETTh1_OT_feature_matrix|run.py'
git status --short
```

Existing modified files belong to the user and prior sessions. Preserve them.
