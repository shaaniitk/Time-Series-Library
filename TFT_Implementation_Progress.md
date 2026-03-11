# TFT, Planetary-Covariate, and Physics Implementation Progress Tracker

> Last updated: 2026-08-01 22:02 (Asia/Kolkata)
>
> Audited base commit: `564cffbc712f`
>
> Scope: repository-native `models/TemporalFusionTransformer.py`; `models/TFT_Nixtla.py` is excluded.
>
> Master guide: [`implementation_plan.md`](implementation_plan.md)
>
> Orchestration rules: [`TFT_Implementation_Orchestrator.md`](TFT_Implementation_Orchestrator.md)
>
> Technical audit: [`TFT_Deep_Analysis_Report.md`](TFT_Deep_Analysis_Report.md)
>
> Historical planetary/NIFTY audit: [`Vedic_Astrology_TFT_Implementation_Plan.md`](Vedic_Astrology_TFT_Implementation_Plan.md)
>
> Canonical financial-astrology plan:
> [`projects/financial_astrology_tft/IMPLEMENTATION_PLAN.md`](projects/financial_astrology_tft/IMPLEMENTATION_PLAN.md)
>
> Canonical financial-astrology tracker:
> [`projects/financial_astrology_tft/PROGRESS_TRACKER.md`](projects/financial_astrology_tft/PROGRESS_TRACKER.md)

## 1. Honest Current Status

The original native TFT implementation roadmap remains closed, but the July
2026 trained-feature matrix exposed a second semantic-repair wave. Historical
`TFT-A*` completion is preserved; new `TFT-SR00`–`TFT-SR09` tasks track the
additional work. Detailed planetary/NIFTY planning is authorized. Data-dependent
implementation is waiting for corrected source OHLC and the ephemeris generator
package after a blocking data audit; generic output-physics work remains
separate.

```text
Original native roadmap:  59 / 59 implementation weight = 100%
Semantic-repair wave:     28 / 28 implementation weight = 100.0%
Generic PHY/THY progress: 0 / 45 active weight = 0%
Planetary AST progress:   0 / 44 implementation weight = 0%
Current native gate:      G2-SR — Post-matrix semantic hardening (PASSED)
Current AST/FA gate:      FA-TFT-SEM/G2 — rashi/date defects; source remediation pending
Native defects fixed:     12
Native release tasks done:12 / 12
Physics modules created:  0
Planetary modules created: 0
Immediate prerequisites:  rebuild/audit source data
```

Do not count a written plan, a started branch, a passing unrelated test, or a partial patch as implementation progress. A task contributes progress only after it is `DONE` and has recorded evidence.

Plain-English repo state:

- the repo already contains the major original native TFT safety fixes identified in the audit;
- `TFT-T01` is now complete and `G1` is closed;
- `TFT-P01` is now complete, so the TFT profile system and stable TFT digest are in place;
- `TFT-A03` is now complete, so canonical mode has pointwise continuous embeddings and a single shared static VSN path;
- `TFT-A02` is now complete, so selected profiles no longer instantiate several structurally dead TFT branches;
- `TFT-E01` is now complete, so a reproducible canonical-vs-extended_safe reference benchmark is saved and `G2` is closed;
- `TFT-A01` is now complete, so the learned FFT branch finally applies a real per-frequency/interpolated mask instead of one channel-wide scalar gate;
- `TFT-A04` is now complete, so temporal sparse graphs preserve their structural support, `top_k=0` is truly dense, and the temporal evolution path no longer uses a dense `C²` recurrent state;
- `TFT-A05` is now complete, so MoE routing can no longer collapse to all-zero on tiny batches, auxiliary loss includes both importance and load balancing, and the implementation now honestly reports dense-compute top-k mixing;
- `TFT-A07` is now complete, so lag attention carries explicit shifted physical positions, masks padded lag keys, rejects impossible lags, and preserves original coordinates through temporal compression;
- `TFT-A09` is now complete, so native TFT defaults/digests are centralized, ignored knobs such as `d_ff` are explicitly warned and de-materialized, and backbone-layer scope is surfaced honestly;
- `TFT-A08` is now complete, so import-time hardware mutation is gone and deep finite-value checks are gated behind `tft_debug_checks`;
- `TFT-A10` is now complete, so self-attention and cross-attention now support a distinct probability-dropout control while keeping interpretation tensors pre-dropout;
- `TFT-A06` is complete for its historical contract, but post-matrix audit found
  additional intended-semantic and interpretation gaps tracked by `TFT-SR*`;
- `TFT-SR00` is complete: all 14 legacy-v1 cases are frozen by hash, repaired
  semantics use versioned identities, and checkpoint compatibility is explicit;
- `TFT-SR01` is complete: production seed streams, deterministic sample order,
  exact paired initialization, tamper-evident manifests, and validation-only
  fitting passed independent adversarial review;
- the matrix exposed non-paired initialization/data order, ignored `--seed`,
  non-neutral optional branches, learned-FFT naming drift, dead final-layer
  decompression, shifted-prefix rather than exact-lag behavior, post-VSN rather
  than named-covariate interactions, and aggressive single-adjacency graph mixing;
- a planetary-covariate lane is now explicitly open, but its ephemerides are treated as known-future inputs rather than an invented output-physics law;
- the production `Dataset_Custom` path cannot yet carry planetary values through `x_mark_*`;
- the optional per-feature VSN gating path has a reproduced `len(None)` forward crash and is tracked as `AST-C00`;
- generic physics-informed losses and theory priors remain unimplemented.

## 2. Status Vocabulary

Use exactly these values:

| Status | Meaning |
|---|---|
| `NOT_STARTED` | One or more declared dependencies are incomplete. |
| `READY` | All dependencies are `DONE`; the task may be claimed. |
| `IN_PROGRESS` | One owner is actively implementing the task. |
| `IN_REVIEW` | Coding is complete, but independent verification is pending. |
| `WAITING_EXTERNAL` | Read-only/internal work is exhausted and completion awaits a named user artifact or external process. |
| `BLOCKED` | Work cannot continue because of a technical or dependency conflict not represented by ordinary task prerequisites. |
| `DONE` | Every acceptance criterion passed and evidence is recorded. |
| `DEFERRED` | Intentionally outside the current implementation milestone. |

An unmet dependency is not a blocker. A normal failing red test is not a blocker. Never use percentages to override a failed gate.

## 3. Progress Rules

Task weights represent rough implementation size:

| Weight | Size |
|---:|---|
| 1 | Small |
| 2 | Medium |
| 3 | Large |
| 5 | Extra large |

```text
implementation_progress =
    100 * sum(weight of DONE implementation tasks)
          / sum(weight of all non-DEFERRED implementation tasks)
```

`IN_PROGRESS` and `IN_REVIEW` receive zero completion credit. Legacy native plus generic PHY/THY weight remains `104`; the separate planetary-research lane adds `44` and is reported separately so completed native work is not obscured.

## 4. Gate Dashboard

| Gate | State | Done weight | Gate weight | Exit evidence |
|---|---|---:|---:|---|
| `G0` Planning baseline | `PASSED` | — | — | `EV-BASE-001`, `EV-DOC-001` |
| `G1` Native safety | `PASSED` | 24 | 24 | `EV-IMP-001`, `EV-IMP-002`, `EV-IMP-003`, `EV-IMP-004`, `EV-IMP-005`, `EV-IMP-006`, `EV-IMP-007`, `EV-IMP-008`, `EV-IMP-009` |
| `G2` Canonical baseline | `PASSED` | 12 | 12 | `EV-IMP-010`, `EV-IMP-011`, `EV-IMP-012`, `EV-IMP-013` |
| `G2-SR` Post-matrix semantic readiness | `PASSED` | 28 | 28 | `EV-IMP-025`, `EV-IMP-026`, `EV-IMP-027`, `EV-IMP-028`, `EV-IMP-029`, `EV-IMP-030`, `EV-IMP-031` |
| `G3` Physics foundation | `LOCKED` | 0 | 17 | — |
| `G4` Production physics integration | `LOCKED` | 0 | 12 | — |
| `G5` Physical validity | `LOCKED` | 0 | 3 | — |
| `G6` VSN theory priors | `LOCKED` | 0 | 6 | — |
| `G7` Counterfactual monotonicity | `LOCKED` | 0 | 5 | — |
| `G8` Experimental hardening | `PASSED — legacy-v1 screening only` | 23 | 23 | Fourteen-case comparison complete; semantic-v2 release still requires `G2-SR`. |
| `AST-G0` Protocol/data contract | `OPEN / EXTERNAL REMEDIATION` | 0 | 6 | `EV-DATA-023`; raw OHLC/session keys and generator manifest required. |
| `AST-G1` Loader and baselines | `LOCKED` | 0 | 9 | — |
| `AST-G2` Circular features and nulls | `LOCKED` | 0 | 7 | — |
| `AST-G3` Raw test plus grouped/multiscale architecture | `LOCKED` | 0 | 11 | — |
| `AST-G4` Evaluation and claim gate | `LOCKED` | 0 | 7 | — |

`PHY-V02` is a post-`G5` ETTh1 ablation and has weight `2`; it is not itself a phase gate.

## 5. Active Work

The native matrix completed 14/14, `TFT-SR00` froze its legacy-v1 evidence, and
`TFT-SR01` passed its reproducibility/paired-ablation gate. `TFT-SR02`
through `TFT-SR09` are now complete and verified by focused semantic-regression
tests; `G2-SR` is closed. The next activation is `AST-H01`.
The read-only `AST-D01` audit is complete and is waiting externally on raw
OHLC/session keys and generator provenance.

| Task | Owner | Status | Started | Primary-file lock | Next action |
|---|---|---|---|---|---|
| `TFT-SR09` | — | `DONE` | — | `LOCK-TFT-MODEL/LAYERS/CONFIG` (released) | Post-matrix semantic release gate passed; next activation is `AST-H01`. |

## 5.1 Operator quick-start

If you are taking over this repo without context, do this in order:

1. Open the native release-gate task card `TFT-T01` in [`implementation_plan.md`](implementation_plan.md).
2. Confirm `TFT-T01` is already recorded `DONE` with `EV-IMP-009`.
3. Confirm the remaining TFT hardening tasks `TFT-A06` through `TFT-A10` are all recorded `DONE`.
4. Preserve the historical completion of those tasks; the fresh post-matrix
   regressions are tracked under `TFT-SR00`–`TFT-SR09`.
5. Read Section 14 of [`implementation_plan.md`](implementation_plan.md), verify
  `TFT-SR00`–`TFT-SR09` evidence, and continue with `AST-H01`.
6. For NIFTY/planet work, read the canonical project
   [`IMPLEMENTATION_PLAN.md`](projects/financial_astrology_tft/IMPLEMENTATION_PLAN.md).
7. Resume `AST-D01` in parallel when raw OHLC and the generator/convention
   package arrive; no neural astrology training begins before `TFT-SR09` and
   the corrected-data gates.

## 6. Next-Activation Queue

The orchestrator normally selects the highest item whose files do not conflict
with active work. The post-matrix semantic wave now precedes neural planetary
training; data/provenance review remains independently parallel.

| Priority | Task | Current activation condition | Primary files |
|---:|---|---|---|
| 1 | `TFT-SR09` | `READY`: `TFT-SR08` passed with `EV-IMP-030`. | native TFT model/layers/config/release-gate tests |
| 2 | `AST-H01` | `READY`: continue protocol choices that do not require results. | `configs/astrology/`, domain protocol |
| 3 | `AST-D01` | `WAITING_EXTERNAL`: audit found bad rashi/date semantics; wait for raw OHLC and generator/convention package. | planetary schema/manifest/tests |

The intended branch after `TFT-T01` is:

| If `TFT-T01` result is… | Then do this next |
|---|---|
| Green | Historical `G1/G2` remain passed. Complete `TFT-SR00`–`TFT-SR09`, then proceed to neural `AST-*` work. |
| Red | Historical branch only: keep `TFT-T01` `IN_PROGRESS`, identify the failing contract, fix only that contract, and rerun the same exact command. |

## 7. Blockers

The legacy matrix has exited, so `TFT-SR00` is no longer blocked. The planetary
data/provenance audit cannot close until authoritative source OHLC/session keys
and generation details are supplied.

| Task | Since | Exact blocker | Evidence | Required action | Owner |
|---|---|---|---|---|---|
| `AST-D01` source remediation | 2026-07-31 | Derived return dates mix session conventions; raw OHLC/session keys are absent. | `EV-DATA-023` | Supply authoritative raw OHLC or exact retrieval script/artifact; rebuild without heuristics. | user input |
| `AST-D01` generator provenance | 2026-07-31 | Generator/config, timestamp/frame/ayanamsha, and Shadbala provenance are absent. | `EV-DATA-023` | Supply PySwissEph generator and manifest; Hilbert code only if a separate feature family was intended. | user input |

## 7.1 Things that are not blockers

Do not mark these as blockers:

- “Generic physics is not started yet.” That is expected and is not required for the planetary known-covariate lane.
- “Historical post-G1 tasks are incomplete.” They remain complete; the new
  semantic defects have their own `TFT-SR*` IDs.
- “A red test exists while writing the fix.” That is normal `IN_PROGRESS` work.
- “The repo has unrelated dirty files.” Record them and work around them.

## 8. Governance and Baseline Ledger

| ID | Status | Owner | Updated | Evidence | Notes |
|---|---|---|---|---|---|
| `GOV-001` | `DONE` | — | 2026-07-28 | `EV-DOC-001` | Native TFT deep audit completed. |
| `GOV-002` | `DONE` | — | 2026-07-28 | `EV-DOC-001` | Physics/theory-guided design corrected. |
| `BASE-001` | `DONE` | — | 2026-07-28 | `EV-BASE-001` | Baseline focused tests recorded. |
| `GOV-003` | `DONE` | — | 2026-07-28 | `EV-DOC-002` | Master guide, tracker, and orchestrator created. |
| `GOV-004` | `DONE` | — | 2026-07-29 | `EV-DOC-003` | Planetary/NIFTY research lane audited and planned without misclassifying ephemerides as a market governing law. |

## 9. Native Release-Blocker Ledger

| ID | Status | Owner | Weight | Dependencies | Evidence | Next action |
|---|---|---|---:|---|---|---|
| `TFT-C01` | `DONE` | — | 3 | `G0` | `EV-IMP-005` | Static covariates now bypass time normalization and are validated for constancy before embedding. |
| `TFT-C02` | `DONE` | — | 1 | `G0` | `EV-IMP-005` | LSTM and hybrid backbones now receive recurrent state in `(h_0, c_0)` order. |
| `TFT-O01` | `DONE` | — | 3 | `G0` | `EV-IMP-002` | Long-term TFT now returns gatherable structured outputs for production. |
| `TFT-C09` | `DONE` | — | 2 | `G0` | `EV-IMP-001` | Shared target mapping now drives loss/test/inverse paths. |
| `TFT-C03` | `DONE` | — | 3 | `TFT-O01`, `TFT-C09` | `EV-IMP-004` | Quantile modes now share one ordered contract, trained outputs, positive RevIN scale, and saved calibration metrics. |
| `TFT-C04` | `DONE` | — | 2 | `G0` | `EV-IMP-006` | Top-amplitude FFT selection now uses rank weights with per-sample/channel bins and is batch-invariant. |
| `TFT-C05` | `DONE` | — | 2 | `G0` | `EV-IMP-007` | Higher-order interaction now uses one sigmoid gate per real term and order-3 no longer crashes. |
| `TFT-C06` | `DONE` | — | 1 | `TFT-C01` | `EV-IMP-008` | Static interpretation payloads now preserve per-context weights/graph metadata and export static feature names. |
| `TFT-C07` | `DONE` | — | 1 | `G0` | `EV-IMP-001` | Short-term/native TFT now fails fast at construction. |
| `TFT-C08` | `DONE` | — | 2 | `G0` | `EV-IMP-003` | Resolved schema validation now guards roles, marks, custom-known inputs, and freq aliases. |
| `TFT-H01` | `DONE` | — | 1 | `G0` | `EV-IMP-001` | Baseline runner/discovery and deep-benchmark controls repaired. |
| `TFT-T01` | `DONE` | — | 3 | All release blockers | `EV-IMP-009` | Exact native semantic release gate passed; `G1` closed. |

Native safety subtotal: `24 / 24`.

Native release-gate completion rule:

- `G1` passed because `TFT-T01` is `DONE`;
- `TFT-T01` is `DONE` because the exact required command from the master plan is recorded green in the evidence log;
- a partial subset pass would not have closed the gate.

## 10. Canonicalization and Extension Ledger

| ID | Status | Owner | Weight | Dependencies | Evidence | Next action |
|---|---|---|---:|---|---|---|
| `TFT-P01` | `DONE` | — | 2 | `G1` | `EV-IMP-010` | Centralized TFT profiles, shared CLI/direct normalization, stable digest, and incompatibility rejection are implemented and regression-tested. |
| `TFT-A02` | `DONE` | — | 2 | `TFT-P01` | `EV-IMP-012` | Selected profiles now prune dead residual/gating/static branches, quantile-only mode no longer instantiates a point head, and profile liveness/state-dict audits are green. |
| `TFT-A03` | `DONE` | — | 5 | `TFT-P01` | `EV-IMP-011` | Canonical profile now uses pointwise continuous embeddings, removes duplicated embedding position buffers, and routes static encoding through one shared static VSN. |
| `TFT-E01` | `DONE` | — | 3 | `TFT-A02`, `TFT-A03` | `EV-IMP-013` | Reproducible canonical-vs-extended_safe benchmark script/report now save accuracy, calibration, latency, parameter, checkpoint, and interpretation-stability summaries. |
| `TFT-A01` | `DONE` | — | 3 | `TFT-C04` | `EV-IMP-014` | Learned FFT mode now interpolates per-frequency mask logits and complex weights, reports learned-mask summaries, and is covered by runtime-length/gradient/checkpoint tests. |
| `TFT-A04` | `DONE` | — | 5 | `G1` | `EV-IMP-015` | Temporal sparse graphs now evolve masked logits without support expansion, `top_k=0` is truly dense, low-rank temporal factors replace the dense `C²` recurrent state, and oversized edge-feature tensors fail fast. |
| `TFT-A05` | `DONE` | — | 5 | `G1` | `EV-IMP-016` | MoE capacity now uses ceil/min-one routing with fallback restoration, auxiliary loss combines importance and load balancing, and structured outputs carry global-reducible MoE summaries plus honest routing-mode metadata. |
| `TFT-A06` | `DONE` | — | 2 | `TFT-P01` | `EV-IMP-021` | Interpretation summaries now export named history/future features, axis labels, and caveat flags describing when canonical attribution claims are unsafe. |
| `TFT-A07` | `DONE` | — | 3 | `G1` | `EV-IMP-017` | Lag attention now masks padded shifted keys, uses physical key coordinates, rejects impossible lags, and preserves original compressed positions for biasing. |
| `TFT-A08` | `DONE` | — | 2 | `G1` | `EV-IMP-019` | Import-time environment mutation is removed, deep finite checks are debug-gated, and shape/schema validation remains always-on. |
| `TFT-A09` | `DONE` | — | 2 | `TFT-P01` | `EV-IMP-018` | Native TFT config defaults/digest semantics are centralized, `d_ff` is explicitly treated as ignored, and backbone-layer scope is exposed honestly. |
| `TFT-A10` | `DONE` | — | 1 | `G1` | `EV-IMP-020` | Exact/SDPA attention now support distinct probability dropout while preserving pre-dropout interpretation tensors. |

Canonicalization/extension subtotal: `35 / 35`.

Recommended post-`G1` order:

1. The original hardening set remains historically closed.
2. Do not rewrite old task evidence; use the new `TFT-SR*` wave below.
3. Keep `PHY-*` and `THY-*` work separate.

## 10.1 Post-Matrix Semantic-Repair Ledger

| ID | Status | Owner | Weight | Dependencies | Evidence | Next action |
|---|---|---|---:|---|---|---|
| `TFT-SR00` | `DONE` | — | 1 | final matrix artifact inventory | `EV-IMP-025` | Legacy-v1 evidence, versioned identities, bound metadata, guarded replay, and migration policy are verified. |
| `TFT-SR01` | `DONE` | — | 3 | `TFT-SR00` | `EV-IMP-026` | Seed streams, exact paired common-state copying, deterministic sample order, manifests, and validation-only fitting are verified. |
| `TFT-SR02` | `DONE` | Codex `/root` | 3 | `TFT-SR01` | `EV-IMP-027` | Exact no-op residual adapters and explicit physical coordinate contracts are implemented and verified. |
| `TFT-SR03` | `DONE` | Codex `/root` | 3 | `TFT-SR02` | `EV-IMP-027` | FFT filter/selection semantics, scopes, diagnostics, and neutral fusion are repaired for v2. |
| `TFT-SR04` | `DONE` | Codex `/root` | 2 | `TFT-SR02` | `EV-IMP-027` | Cross-attention path is neutral-safe, mask-safe, and interpretation-labeled correctly. |
| `TFT-SR05` | `DONE` | Codex `/root` | 3 | `TFT-SR02` | `EV-IMP-027` | Lag semantics now split shifted-prefix, exact-token, and declared-regular-sampling response modes. |
| `TFT-SR06` | `DONE` | Codex `/root` | 3 | `TFT-SR02` | `EV-IMP-028` | Latent polynomial semantics are separated from named covariate interactions, pre-VSN named interaction interface is added, and per-feature VSN gating forward/backward liveness is validated. |
| `TFT-SR07` | `DONE` | Codex `/root` | 3 | `TFT-SR02` | `EV-IMP-029` | Dead compression codec semantics are replaced with live anti-aliased K/V memory compression, with short-window guardrails and payload diagnostics validated. |
| `TFT-SR08` | `DONE` | Codex `/root` | 4 | `TFT-SR02` | `EV-IMP-030` | Graph cross-mixing now uses live residual propagation by default for standalone learners, preserves neutral exact-no-op behavior under semantics-v2 adapters, and validates typed/single-head/top-k metadata. |
| `TFT-SR09` | `DONE` | — | 3 | `TFT-SR00`–`TFT-SR08` | `EV-IMP-031` | Post-matrix semantic release gate passed with full regression coverage and diff hygiene. |

Post-matrix semantic subtotal: `28 / 28`.

`TFT-SR09` closes `G2-SR`. A passing old shape suite or a completed ETTh1 case
cannot substitute for it.

### EV-IMP-031 — SR09 post-matrix semantic release gate

- Task: `TFT-SR09`.
- Date: 2026-08-01.
- Implementation scope:
  - the post-matrix semantic-release gate records the completed SR02-SR09 regression slice plus semantic versioning compatibility;
  - compatibility aliases remain available for legacy comprehensive coverage while canonical SR08 metadata stays truthful;
  - the gate closes with all SR00-SR09 tasks done and `G2-SR` passed.
- Acceptance evidence:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest \
    tests/test_tft_semantics_versioning.py \
    tests/test_tft_sr02_config.py \
    tests/test_tft_sr02_model_neutrality.py \
    tests/test_tft_sr03_fft_semantics.py \
    tests/test_tft_sr04_cross_attention_semantics.py \
    tests/test_tft_sr05_lag_semantics.py \
    tests/test_tft_sr06_interaction_semantics.py \
    tests/test_tft_sr07_temporal_compression_semantics.py \
    tests/test_tft_sr08_graph_semantics.py \
    tests/test_tft_comprehensive.py -q

  cd /home/kalki/Documents/workspace/Time-Series-Library && git diff --check
  ```

- Result:
  - `193 passed, 160 warnings`;
  - `git diff --check` passed.

## 11. Physics and Theory-Guided Ledger

| ID | Status | Owner | Weight | Dependencies | Evidence | Next action |
|---|---|---|---:|---|---|---|
| `PHY-C01` | `READY` | — | 3 | `G0` | — | Create frozen schema dataclasses and safe parser. |
| `PHY-D01` | `NOT_STARTED` | — | 3 | `TFT-C08`, `TFT-C09`, `PHY-C01` | — | Wait for schema and target repairs. |
| `PHY-S01` | `NOT_STARTED` | — | 3 | `PHY-C01`, `PHY-D01` | — | Implement resolver and Torch transforms. |
| `PHY-L01` | `NOT_STARTED` | — | 5 | `PHY-S01` | — | Implement the five Stage-1 constraints. |
| `PHY-I02` | `NOT_STARTED` | — | 2 | `PHY-S01` | — | Add CLI, digest, and manifest lifecycle after transforms resolve. |
| `PHY-I01` | `NOT_STARTED` | — | 5 | `G1`, `PHY-L01`, `PHY-I02` | — | Integrate the production loss. |
| `PHY-M01` | `NOT_STARTED` | — | 2 | `PHY-I01` | — | Add metrics and checkpoint policies. |
| `PHY-T01` | `NOT_STARTED` | — | 3 | `PHY-L01` | — | Complete the unit-test matrix. |
| `PHY-T02` | `NOT_STARTED` | — | 3 | `PHY-I01`, `PHY-M01` | — | Run production/AMP/mapping/parity integration tests. |
| `PHY-V01` | `NOT_STARTED` | — | 3 | `G4` | — | Prove behavior on a synthetic physical system. |
| `PHY-V02` | `NOT_STARTED` | — | 2 | `PHY-V01` | — | Run controlled ETTh1 theory ablations. |
| `THY-A01` | `NOT_STARTED` | — | 3 | `G1`, `TFT-O01` | — | Add selective differentiable VSN outputs. |
| `THY-V01` | `NOT_STARTED` | — | 3 | `THY-A01` | — | Add valid VSN priors. |
| `THY-M01` | `NOT_STARTED` | — | 5 | `G5` | — | Add deterministic counterfactual monotonicity. |
| `ADV-P01` | `DEFERRED` | — | 5 | `G5`, validated domain law | — | Do not start without a defensible law. |

Active physics/theory subtotal, excluding deferred work: `0 / 45`.

Recommended physics/theory order:

1. `PHY-C01`
2. `PHY-D01`
3. `PHY-S01`
4. `PHY-L01`
5. `PHY-T01`
6. `PHY-I02`
7. `PHY-I01`
8. `PHY-M01`
9. `PHY-T02`
10. `PHY-V01`
11. `PHY-V02`
12. `THY-A01`
13. `THY-V01`
14. `THY-M01`

Generic output-physics work remains a separate reusable branch. It is not a prerequisite for the planetary known-future-covariate study because no planet-to-NIFTY governing equation has been supplied.

## 12. Planetary-Covariate Research Ledger

| ID | Status | Owner | Weight | Dependencies | Evidence | Next action |
|---|---|---|---:|---|---|---|
| `AST-H01` | `READY` | — | 2 | `G2` | `EV-DOC-003`, `EV-DATA-023` | Use the audit findings to freeze target, cutoff, folds, metrics, seeds, and final holdout without opening it. |
| `AST-D01` | `WAITING_EXTERNAL` | — | 3 | supplied files; source/generator remediation | `EV-DATA-023` | Resume with raw OHLC/session keys and PySwissEph generator/convention package. |
| `AST-C00` | `NOT_STARTED` | — | 1 | `G2-SR` / `TFT-SR06` | `EV-DOC-003` | Accept the per-feature VSN repair evidence from `TFT-SR06`; do not implement it twice. |
| `AST-K01` | `NOT_STARTED` | — | 5 | `AST-D01`, `G2-SR` | — | Build the production known-future loader after the schema and native coordinate/reproducibility contracts are frozen. |
| `AST-B01` | `NOT_STARTED` | — | 4 | `AST-H01`, `AST-K01`, `G2-SR` | — | Add zero/Ridge/TFT market and market+calendar baselines plus paired metrics. |
| `AST-F01` | `NOT_STARTED` | — | 3 | `AST-D01`, `AST-K01` | — | Build frozen circular, relative-harmonic, and declared-aspect feature families. |
| `AST-N01` | `NOT_STARTED` | — | 4 | `AST-D01`, `AST-F01` | — | Build coherent shifted, spectral, and smooth pseudo-planet null blocks. |
| `AST-E00` | `NOT_STARTED` | — | 2 | `AST-H01`, `AST-B01`, `AST-F01`, `AST-N01` | — | Test raw planetary incremental value before bespoke architecture work. |
| `AST-M01` | `NOT_STARTED` | — | 5 | `AST-E00` proceed decision | — | Add grouped planet tokens and low-rank relative-geometry messages. |
| `AST-L01` | `NOT_STARTED` | — | 4 | `AST-M01` | — | Add calendar-time-aware fast/intermediate/slow response kernels. |
| `AST-O01` | `NOT_STARTED` | — | 3 | stable point model | — | Add a structurally valid gap/body/upper/lower OHLC head. |
| `AST-Q01` | `NOT_STARTED` | — | 1 | stable point model | — | Add quantile calibration ablations. |
| `AST-E01` | `NOT_STARTED` | — | 5 | selected frozen architecture/nulls | — | Run all walk-forward folds/seeds and inspect the locked test once. |
| `AST-R01` | `NOT_STARTED` | — | 2 | `AST-E01` | — | Classify the evidence as stable association, null, unstable, or invalid. |

Planetary-covariate subtotal: `0 / 44`.

## 13. Evidence Log

### EV-IMP-029 — SR07 temporal-compression semantic redesign and release

- Task: `TFT-SR07`.
- Date: 2026-08-01.
- Implementation scope:
  - temporal compression now supports explicit mode semantics with `legacy_codec` (historical path) and `kv_pool` (live K/V pooling);
  - `kv_pool` adds anti-aliased history pooling with explicit availability/content-center coordinate metadata and pooled valid-mask propagation;
  - short-window activation is guarded by `tft_tc_min_long_sequence` and explicit override `tft_tc_experimental_short_window`;
  - neutral-mode exact no-op parity is preserved for temporal compression under semantics-v2 residual adapters;
  - interpretation payload now surfaces `tc_mode` and `tc_kv_pooling` diagnostics for release audits.
- Acceptance evidence:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest \
    tests/test_tft_sr07_temporal_compression_semantics.py \
    tests/test_tft_sr02_config.py \
    tests/test_tft_semantics_versioning.py \
    tests/test_tft_sr06_interaction_semantics.py \
    tests/test_tft_sr05_lag_semantics.py \
    tests/test_tft_sr04_cross_attention_semantics.py \
    tests/test_tft_sr03_fft_semantics.py \
    tests/test_tft_sr02_model_neutrality.py \
    tests/test_tft_comprehensive.py -k temporal_compression
  ```

- Result:
  - `15 passed, 172 deselected, 14 warnings`;
  - no SR07-related regressions remained in the semantic/versioning/compression slice.

### EV-IMP-030 — SR08 graph cross-mixing semantic release

- Task: `TFT-SR08`.
- Date: 2026-08-01.
- Implementation scope:
  - standalone `AdvancedDynamicGraphLearner` now defaults to a small positive residual strength so multihop propagation is observable without changing explicit neutral configs;
  - graph semantics-v2 remains exact-no-op under neutral adapters and model-neutrality tests;
  - typed single-head and true-multihead graph metadata remain surfaced correctly in model payloads;
  - sparse/top-k and temporal-graph behaviors continue to satisfy the existing graph contract.
- Acceptance evidence:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest \
    tests/test_tft_sr08_graph_semantics.py \
    tests/test_tft_sr02_model_neutrality.py \
    tests/test_tft_comprehensive.py -k graph
  ```

- Result:
  - `21 passed, 73 deselected, 8 warnings`;
  - no SR08-related regressions remained in the graph slice.

### EV-IMP-028 — SR06 latent-polynomial semantics and per-feature VSN repair

- Task: `TFT-SR06`.
- Date: 2026-08-01.
- Implementation scope:
  - higher-order branch interpretation now reports the actual projected residual and strength under the canonical semantic role `latent_polynomial_block` while retaining backward-compatible payload aliases;
  - the model interpretation payload now surfaces latent polynomial residual/strength fields for explicit diagnostics;
  - `NamedCovariateInteractionEncoder` is introduced as a pre-VSN named-pair interaction interface over explicit declared covariate pairs, with order-stable name resolution and no unrestricted `C x C` default expansion;
  - per-feature VSN gating path is validated with forward/backward liveness checks and released in semantics-v2 capability gating.
- Acceptance evidence:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest \
    tests/test_tft_sr06_interaction_semantics.py \
    tests/test_tft_sr02_config.py \
    tests/test_tft_semantics_versioning.py \
    tests/test_tft_sr05_lag_semantics.py \
    tests/test_tft_sr04_cross_attention_semantics.py \
    tests/test_tft_sr03_fft_semantics.py \
    tests/test_tft_sr02_model_neutrality.py
  ```

- Result:
  - `113 passed, 119 warnings`;
  - no SR06-related regression remained in the semantic/versioning slice.

### EV-IMP-027 — SR02/SR03/SR04/SR05 semantic repairs and regression closeout

- Tasks: `TFT-SR02`, `TFT-SR03`, `TFT-SR04`, `TFT-SR05`.
- Date: 2026-08-01.
- Implementation scope:
  - exact-neutral extension residual adapters and explicit coordinate contract remain enforced;
  - FFT v2 semantics are split by history/future scope with dedicated diagnostics;
  - explicit cross-attention reports role/scope diagnostics and preserves interpretation-safety flags;
  - lag semantics support explicit mode selection (`shifted_prefix_attention`, `exact_token_lag`, and declared regular-sampling response), with mode surfaced in interpretation payloads;
  - lag mode defaults are v2-only so legacy-v1 digest remains frozen.
- Acceptance evidence:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest \
    tests/test_tft_sr05_lag_semantics.py \
    tests/test_tft_attention_coordinates.py \
    tests/test_tft_sr04_cross_attention_semantics.py \
    tests/test_tft_sr03_fft_semantics.py \
    tests/test_tft_sr02_model_neutrality.py \
    tests/test_tft_sr02_config.py \
    tests/test_tft_semantics_versioning.py
  ```

- Result:
  - `133 passed, 116 warnings`;
  - no semantic-versioning or legacy-digest regression remains in the repaired scope.

### EV-IMP-026 — Reproducible and paired native-TFT experiment contract

- Task: `TFT-SR01`.
- Date: 2026-08-01.
- Production behavior:
  - `seed`, model-init, extension-init, data-order, worker, and training streams
    resolve independently under an explicit per-iteration schedule;
  - production seeding uses the single `set_experiment_seed()` entry point and
    honors `off`, `warn`, and `strict` deterministic-backend policies;
  - only train is shuffled; validation/test order is deterministic, worker
    counts `0` and `>0` preserve sample order, and v2 direct callers cannot
    silently fall back to legacy shuffled validation;
  - paired construction builds a reference under `model_init_seed`, builds the
    variant under `extension_init_seed`, copies all shared tensors bitwise, and
    fails closed on unexpected names, shapes, dtypes, schemas, or forbidden
    reference controls;
  - the canonical reference spec is part of run identity, and the full paired
    report/spec is bound by a trusted construction-time digest so post-build
    mutations cannot forge shared/reference hashes or tensor inventories;
  - fold manifests bind current source bytes, split coordinates, transformed
    content, augmentation configuration/effective seed, and ordered sample IDs;
    an unaugmented fold does not claim or depend on an unused seed;
  - confirmatory fitting uses validation only and never constructs or evaluates
    the test loader during epochs.
- Acceptance evidence:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest -q \
    tests/test_tft_reproducibility.py \
    tests/test_tft_reproducibility_utils.py \
    tests/test_tft_data_reproducibility.py \
    tests/test_tft_semantics_versioning.py \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_profiles.py

  PYTHONPATH=. ./ai_env/bin/python scripts/tft_verify_legacy_matrix.py

  PYTHONPATH=. ./ai_env/bin/python -m py_compile \
    run.py exp/exp_long_term_forecasting.py \
    data_provider/data_factory.py data_provider/data_loader.py \
    utils/reproducibility.py utils/print_args.py utils/tft_config.py

  git diff --check
  ```

- Result:
  - `101 passed, 23 warnings` in the combined SR01/SR00/profile/experiment gate;
  - an independent adversarial review also passed `65` focused and `89`
    broader native tests and found no remaining SR01 blocker;
  - the production micro-test repeats a paired one-epoch run with non-null,
    identical shared/checkpoint/order/fold hashes, then proves a changed seed
    changes model state and sample order;
  - `65` immutable legacy-v1 records still verify;
  - compilation and diff hygiene passed;
  - `tests/test_tft_comparison.py` is not counted because it starts a long GPU
    benchmark at import/collection rather than acting as an opt-in unit test.

### EV-IMP-025 — Legacy-v1 freeze and semantic checkpoint boundary

- Task: `TFT-SR00`.
- Date: 2026-08-01.
- Frozen inventory:
  [`metadata/tft/legacy_v1_matrix_manifest.json`](metadata/tft/legacy_v1_matrix_manifest.json),
  SHA-256 `7a86aff8509a96eed44282f4b9e53fb2658fe06130480843cba2e47afe6b21fc`.
- Producer snapshot SHA-256:
  `9a438eb714aaeec8a947a54763092be5a73781c38a6325558cb0cb8f1539b9d2`;
  the separately labeled explicit-v1 replay launcher SHA-256 is
  `348b548e0df705e0c4df9ff5746a92dc3e77c365bf9fa8ecc27f208b389addb2`.
- Implementation:
  - legacy semantics retain the known baseline digest `6c54053ec5d7`;
  - repaired/current identities add `_tsv{version}` and a resolved-schema-aware
    digest;
  - bare checkpoints remain backward compatible, while atomic sidecars bind
    the exact checkpoint SHA-256, resolved config/schema, setting, active
    contracts, and migration policy;
  - missing sidecars are classified as v1 and require both explicit legacy
    runtime and compatibility authorization;
  - no v1 weights can enter v2, and no pending semantic-v2 operator can be
    stamped as repaired before its `TFT-SR*` task is released;
  - historical no-`_tsv` directories resolve through the production test path
    only under the explicit v1 compatibility policy.
- Exact verification:

  ```bash
  PYTHONPATH=. ./ai_env/bin/python scripts/tft_verify_legacy_matrix.py

  PYTHONPATH=. ./ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_extension_contracts.py \
    tests/test_tft_comprehensive.py \
    tests/test_tft_profiles.py \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_semantics_versioning.py

  PYTHONPATH=. ./ai_env/bin/python -m py_compile \
    utils/tft_config.py utils/tft_checkpoint.py \
    scripts/tft_verify_legacy_matrix.py \
    exp/exp_long_term_forecasting.py run.py

  git diff --check
  ```

- Result:
  - `65` immutable legacy-v1 records verified;
  - `158 passed, 37 warnings` in the full native regression group;
  - compilation and diff hygiene passed;
  - warnings are the existing near-constant-channel and ignored-`d_ff`
    diagnostics.

### EV-MATRIX-024 — Native advanced-feature matrix completed

- Date: 2026-07-31 23:22 Asia/Kolkata.
- Scope: 14 ETTh1 `pred_len=24` native-TFT legacy-v1 cases.
- Evidence: 14 result directories, 14 checkpoint directories, and the final
  entry in `result_long_term_forecast.txt`.
- Final case: `tft_ot_p24_experimental_profile`, profile
  `experimental_full`, digest `0d7246b61e1b`.
- Final metrics: MSE `0.0424571671`, MAE `0.1627180278`, pinball
  `0.0530012586`, nominal-80% coverage `0.6766567495`, width `0.4105779813`,
  crossing `0.0`.
- Final MSE is `+18.96%` versus the `0.0356888101` baseline.
- Process check after completion: no matrix launcher or `run.py` remained.
- Interpretation: execution evidence is complete; paired semantic attribution
  remains invalid until `TFT-SR00`–`TFT-SR09` close.

### EV-DATA-023 — Blocking NIFTY/planetary source audit

- Date: 2026-07-31.
- Task: `AST-D01` / `FA-DATA-001`.
- Full report:
  [`projects/financial_astrology_tft/DATA_AUDIT_REPORT.md`](projects/financial_astrology_tft/DATA_AUDIT_REPORT.md).
- Audited artifacts:
  - planet table: 18,251 × 88, SHA-256
    `3036dc79790fed35d3a5f9cf2b4e7726bb00eec7915b54f839aba4a99671fa6d`;
  - return table: 7,109 × 6, SHA-256
    `6765711ed03964f06fec725683e28bc1e83461c36feba09181c080d9a05485b6`.
- Continuous longitude pairs are coherent, but all supplied rashi pairs for all
  12 bodies/nodes and all rows implement the same off-by-one/clipping defect.
- The return file mixes correct session labels and one-day-early labels; 412
  strict four-field fingerprints match the following calendar day's bar.
- No generator or convention manifest exists in the tree/Git history; same-date
  local artifacts differ by exactly 8.25 hours of angular motion.
- Disposition: reject existing rashi fields; quarantine Shadbala; derive Ketu
  from one node axis; omit Hilbert; rebuild market dates and elapsed time.
- No source data was mutated and no model was trained.
- State: `WAITING_EXTERNAL` pending raw OHLC/session keys and
  generator/convention package.
- Verification replay passed hashes/shapes, all 12 all-row rashi-defect
  assertions, direct unit-circle tolerance `4.44e-16`, 120 weekend labels, and
  the exact 37-column provisional schema; document diff/fence/link hygiene also
  passed.

### EV-AUDIT-022 — Post-matrix extension semantic audit

- Date: 2026-07-31.
- Scope: native TFT FFT, cross-attention, lag attention, latent higher-order,
  temporal compression, sparse graph mixing, and matrix experiment controls.
- Matrix state at final planning inspection: 13 of 14 result artifacts complete;
  `tft_ot_p24_experimental_profile` was the active final case and was not
  interrupted.
- Existing component/contract verification run during the audit:
  `80 passed` across `tests/test_tft_extension_contracts.py` and
  `tests/test_tft_comprehensive.py`.
- Runtime/checkpoint findings:
  - `--seed` is parsed but native training hardcodes `2021`;
  - optional construction changes common initial tensors and subsequent sampler
    RNG, so the matrix is not a paired one-seed ablation;
  - optional branches are not exact baseline-neutral at initialization;
  - learned FFT uses interpolated all-bin filter anchors and its trained mask
    remained near-uniform;
  - cross-attention is active and correctly directed, but metadata and neutral
    integration need repair;
  - lag is shifted-prefix attention, not exact-lag lookup, and compressed/
    irregular source positions are calculated incorrectly;
  - higher-order is post-VSN latent polynomial interaction, not a named
    covariate-pair interaction;
  - final-layer compression decompressor parameters receive zero gradient;
  - sparse graph uses one adjacency broadcast as heads, strongly transforms the
    VSN input, and applied `top_k=3` at incompatible 11-node and 4-node densities.
- Interpretation: the legacy matrix is useful screening evidence but cannot
  establish that the affected ideas are beneficial or invalid.
- Resulting work: new `TFT-SR00`–`TFT-SR09` plan and `G2-SR` gate.

### EV-DOC-003 — Planetary/NIFTY hypothesis audit and implementation roadmap

- Task: `GOV-004`; planning evidence for `AST-H01`, `AST-D01`, and `AST-C00`.
- Date: 2026-07-29.
- Evidence:
  - production `Dataset_Custom` was inspected and found to generate `x_mark_*` from calendar fields only;
  - the native custom-known model path was confirmed to consume named past/future mark tensors;
  - generic output-physics constraints were rejected as a prerequisite because deterministic ephemerides do not supply a governing equation for NIFTY;
  - the falsifiable baseline/null/split/architecture protocol was added to `Vedic_Astrology_TFT_Implementation_Plan.md`;
  - the optional per-feature VSN gating path was directly reproduced to fail with `TypeError: object of type 'NoneType' has no len()`;
  - no planetary loader, loss, model, or result is claimed as implemented.
- Verification:

  ```bash
  git diff --check

  PYTHONPATH=. ./ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_extension_contracts.py
  ```

- Result:
  - diff hygiene passed;
  - `46 passed, 17 warnings in 3.81s`;
  - warnings were the existing near-constant-channel and ignored-`d_ff` diagnostics.
- Required continuation:
  - inspect the supplied representative data and generation code;
  - freeze `AST-H01`;
  - complete `AST-D01` before writing the loader.

### EV-BASE-001 — Focused native baseline

- Task: `BASE-001`
- Base SHA: `564cffbc712f`
- Date: 2026-07-28
- Commands:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py

  PYTHONPATH=. ./ai_env/bin/pytest -q \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_tft_scripts_smoke.py \
    tests/test_revin_ablation.py
  ```

- Result: `55 passed, 1 warning`; additional suite `10 passed`.
- Caveat: these tests do not prove the semantic defects are fixed; the audit reproduced gaps not covered by them.

### EV-DOC-001 — Audit and revised physics design

- Files:
  - [`TFT_Deep_Analysis_Report.md`](TFT_Deep_Analysis_Report.md)
  - [`implementation_plan.md`](implementation_plan.md)
- Result: audit completed; local links, code fences, embedded JSON, and line anchors validated.

### EV-DOC-002 — Tracking system

- Files:
  - [`TFT_Implementation_Progress.md`](TFT_Implementation_Progress.md)
  - [`TFT_Implementation_Orchestrator.md`](TFT_Implementation_Orchestrator.md)
- Result: task IDs, status rules, dependencies, gates, and ledgers established.

### EV-IMP-001 — G1 initial native safety implementation

- Tasks:
  - `TFT-H01`
  - `TFT-C07`
  - `TFT-C09`
- Date: 2026-07-28
- Files:
  - [`run_tests.py`](run_tests.py)
  - [`tests/test_tft_deep_ett.py`](tests/test_tft_deep_ett.py)
  - [`tests/test_tft_core_contracts.py`](tests/test_tft_core_contracts.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
  - [`utils/tft_schema.py`](utils/tft_schema.py)
  - [`exp/exp_long_term_forecasting.py`](exp/exp_long_term_forecasting.py)
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_tft_interpretation_and_exp.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_tft_scripts_smoke.py \
    tests/test_revin_ablation.py

  ai_env/bin/python tests/test_tft_deep_ett.py \
    --configs micro \
    --epochs 2 \
    --max-train-batches 1 \
    --max-eval-batches 1

  python3 run_tests.py -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py
  ```

- Result:
  - `14 passed, 1 warning`
  - `55 passed, 1 warning`
  - `10 passed`

### EV-IMP-009 — Native semantic release gate

- Task:
  - `TFT-T01`
- Date: 2026-07-28
- Command:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_extension_contracts.py \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_revin_ablation.py
  ```

- Result:
  - `106 passed, 8 warnings in 18.88s`
- Notes:
  - this is the exact required `TFT-T01` release command from the master plan;
  - `tests/test_revin_ablation.py` now completes as a small synthetic forward/backward gate instead of a heavy import-time benchmark;
  - `G1` is formally closed by this evidence.

### EV-IMP-010 — TFT profile system and stable digest

- Task:
  - `TFT-P01`
- Date: 2026-07-28
- Files:
  - [`utils/tft_config.py`](utils/tft_config.py)
  - [`run.py`](run.py)
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`exp/exp_long_term_forecasting.py`](exp/exp_long_term_forecasting.py)
  - [`utils/print_args.py`](utils/print_args.py)
  - [`tests/test_tft_profiles.py`](tests/test_tft_profiles.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_profiles.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_extension_contracts.py \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_revin_ablation.py
  ```

- Result:
  - `4 passed in 1.58s`
  - `38 passed, 6 warnings in 3.80s`
  - `110 passed, 8 warnings in 19.24s`
- Notes:
  - both CLI and direct `Model(args)` construction now use the same TFT profile normalization path;
  - `canonical`, `extended_safe`, and `experimental_full` profiles now resolve through one shared utility;
  - incompatible canonical overrides now fail early instead of silently drifting;
  - TFT run settings now include a stable TFT profile/config digest.

### EV-IMP-011 — Canonical embeddings and shared static encoder

- Task:
  - `TFT-A03`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`tests/test_tft_profiles.py`](tests/test_tft_profiles.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_profiles.py -k canonical

  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_profiles.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_comprehensive.py \
    -k "profile or canonical or quantile or static"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_extension_contracts.py \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_revin_ablation.py
  ```

- Result:
  - `3 passed, 3 deselected in 1.88s`
  - `7 passed in 2.27s`
  - `28 passed, 65 deselected, 4 warnings in 6.04s`
  - `113 passed, 8 warnings in 19.64s`
- Notes:
  - canonical profile now uses per-variable pointwise continuous embeddings instead of `DataEmbedding` for observed/static continuous channels;
  - canonical profile no longer carries per-variable positional buffers inside those continuous embeddings;
  - canonical static encoding now uses one shared static VSN feeding four context GRNs instead of four separate static VSNs.

### EV-IMP-012 — Profile dead-branch pruning and liveness audit

- Task:
  - `TFT-A02`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`tests/test_tft_profiles.py`](tests/test_tft_profiles.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_profiles.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    -k "dead or state_dict or parameter_count"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_experiment_contracts.py \
    -k "quantile_only_evaluates_trained_output or quantile"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_comprehensive.py \
    -k "profile or canonical or quantile or static or dead or state_dict or parameter_count"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_extension_contracts.py \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_revin_ablation.py
  ```

- Result:
  - `10 passed, 1 warning in 3.54s`
  - `3 passed, 7 deselected, 1 warning in 2.98s`
  - `6 passed, 8 deselected in 3.15s`
  - `31 passed, 65 deselected, 5 warnings in 6.91s`
  - `116 passed, 9 warnings in 20.75s`
- Notes:
  - VSN residual-bypass layers are no longer instantiated when residual bypass is disabled;
  - per-feature gating mode no longer instantiates the unused softmax-selection path;
  - static encoders without static inputs no longer carry unused GRN/state payload branches;
  - quantile-only mode no longer instantiates a point head and derives the reported point forecast from the trained median quantile path;
  - profile liveness and state-dict tests now guard against reintroducing dead parameters.

### EV-IMP-013 — Canonical reference benchmark

- Task:
  - `TFT-E01`
- Date: 2026-07-28
- Files:
  - [`scripts/long_term_forecast/tft_profile_reference_benchmark.py`](scripts/long_term_forecast/tft_profile_reference_benchmark.py)
  - [`tests/test_tft_scripts_smoke.py`](tests/test_tft_scripts_smoke.py)
  - [`results/tft_profile_reference_benchmark_test/tft_profile_reference_summary.json`](results/tft_profile_reference_benchmark_test/tft_profile_reference_summary.json)
  - [`results/tft_profile_reference_benchmark_test/tft_profile_reference_summary.md`](results/tft_profile_reference_benchmark_test/tft_profile_reference_summary.md)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/python \
    scripts/long_term_forecast/tft_profile_reference_benchmark.py \
    --quick \
    --device cpu \
    --output-dir results/tft_profile_reference_benchmark_test

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_scripts_smoke.py \
    -k "profile_reference_benchmark_quick"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_profiles.py \
    tests/test_tft_scripts_smoke.py \
    -k "profile or benchmark"
  ```

- Result:
  - benchmark script saved reproducible JSON and Markdown reference artifacts for `canonical` and `extended_safe`
  - `1 passed, 6 deselected in 4.36s`
  - `12 passed, 5 deselected, 1 warning in 9.08s`
- Notes:
  - the benchmark freezes seeds, synthetic data generation, optimizer family, and model size while comparing `canonical` vs `extended_safe`;
  - the saved reference table reports MSE/MAE, quantile calibration where available, latency, parameter count, checkpoint size, and a simple interpretation-stability signature across seeds;
  - the quick benchmark artifacts are suitable as a reproducible regression reference in this repository.

### EV-IMP-014 — Learned FFT selector upgrade

- Task:
  - `TFT-A01`
- Date: 2026-07-28
- Files:
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`tests/test_tft_extension_contracts.py`](tests/test_tft_extension_contracts.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_extension_contracts.py \
    -k "fft"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    -k "spectral_branch_component_shape_and_gradient or spectral_branch_modes_clamped or tsl_fft_branch_interpretation_payload"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_extension_contracts.py \
    tests/test_tft_comprehensive.py \
    -k "fft or spectral_branch"
  ```

- Result:
  - `7 passed, 5 deselected in 0.82s`
  - `3 passed, 49 deselected in 4.01s`
  - `12 passed, 52 deselected in 6.16s`
- Notes:
  - learned FFT mode now parameterizes a real spectral mask over anchor bins and interpolates it to the runtime FFT grid;
  - learned complex weights follow the same interpolation policy, so checkpoints remain portable across supported sequence lengths;
  - interpretation payloads now export learned-mask mean, spread, and peak-bin summaries when `tft_fft_mode_select="learned"`;
  - regression tests now cover per-frequency variation, runtime-length behavior, finite gradients, and state-dict round trips.

### EV-IMP-015 — Advanced graph sparsity/scaling repair

- Task:
  - `TFT-A04`
- Date: 2026-07-28
- Files:
  - [`layers/AdvancedDynamicGraph.py`](layers/AdvancedDynamicGraph.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    -k "AdvancedDynamicGraph and (sparse_graph_sparsity or temporal_evolution_varies_adjacency or temporal_sparse_graph_stays_top_k or graph_top_k_zero_dense_or_rejected or temporal_graph_rows_are_finite_and_normalized or temporal_graph_initial_alpha_is_point_one or temporal_variation_does_not_expand_support or temporal_evolution_parameter_growth_regression)"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    -k "model_integration_sparse_graph or model_integration_temporal_graph"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    -k "AdvancedDynamicGraph or model_integration_sparse_graph or model_integration_temporal_graph"
  ```

- Result:
  - `8 passed, 50 deselected in 3.86s`
  - `2 passed, 56 deselected in 6.00s`
  - `13 passed, 45 deselected in 5.38s`
- Notes:
  - temporal evolution now perturbs base logits rather than already-normalized adjacency probabilities;
  - the original structural/top-k support mask is preserved and re-applied before the final softmax, so masked edges cannot reappear;
  - `top_k=0` now means dense support instead of producing an empty graph;
  - the temporal evolution path now uses a compact low-rank recurrent state with source/destination factor projections instead of a dense `C²` hidden state;
  - edge-feature mode now fails fast when it would allocate an oversized dense edge tensor.

### EV-IMP-016 — MoE capacity/balancing hardening

- Task:
  - `TFT-A05`
- Date: 2026-07-28
- Files:
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`exp/exp_long_term_forecasting.py`](exp/exp_long_term_forecasting.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    -k "regime_moe_component or moe_small_batch_has_nonzero_route or moe_every_token_has_route or moe_heavy_imbalance_capacity_case_keeps_routes or moe_selected_experts_receive_gradients"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_experiment_contracts.py \
    -k "moe or structured_output or global_moe"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    -k "moe"
  ```

- Result:
  - `5 passed, 57 deselected in 3.78s`
  - `2 passed, 12 deselected in 3.08s`
  - `6 passed, 56 deselected in 3.87s`
- Notes:
  - capacity now uses `ceil()` and enforces a minimum of one kept route when tokens exist;
  - tokens that would otherwise lose all routes are restored to their strongest dense fallback expert before normalization;
  - auxiliary loss now combines importance and load-balancing terms;
  - structured outputs now carry both reducible importance and reducible load summaries;
  - the routing mode is explicitly labeled `dense_compute_topk_mixing` to avoid implying sparse-dispatch compute.

### EV-IMP-017 — Lag attention masking and physical-position repair

- Task:
  - `TFT-A07`
- Date: 2026-07-28
- Files:
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
- Commands:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_comprehensive.py \
    -k "multiscale_lag_attention_component or lag_attention_uses_shifted_physical_positions or lag_attention_excessive_lag_fails or compressed_positions_remain_monotonic_original_coordinates"

  PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_comprehensive.py \
    -k "lag_attention or position_bias_modes"
  ```

- Result:
  - `4 passed, 61 deselected in 4.03s`
  - `4 passed, 61 deselected in 3.97s`
- Notes:
  - lag branches now mask the first `lag` shifted keys instead of allowing padded zeros to compete for attention;
  - RoPE/ALiBi now receive physical lagged key positions;
  - lag branches fail fast when `lag >= active_sequence_length`;
  - temporal compression now preserves original monotonic coordinates for downstream attention biasing.

### EV-IMP-018 — Native TFT config/digest cleanup

- Task:
  - `TFT-A09`
- Date: 2026-07-28
- Files:
  - [`utils/tft_config.py`](utils/tft_config.py)
  - [`run.py`](run.py)
  - [`utils/print_args.py`](utils/print_args.py)
  - [`tests/test_tft_profiles.py`](tests/test_tft_profiles.py)
- Commands:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_profiles.py \
    -k "digest or defaults or d_ff or identical_resolved"
  ```

- Result:
  - `4 passed, 8 deselected in 1.52s`
- Notes:
  - native TFT defaults now include attention-dropout/debug controls in one shared config layer;
  - `d_ff` is explicitly warned as ignored and removed from the material digest for native TFT;
  - the digest now records backbone-layer scope rather than implying plain LSTM depth control;
  - printed args/help strings now match actual native semantics.

### EV-IMP-019 — Debug-gated runtime checks and import-side-effect removal

- Task:
  - `TFT-A08`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`run.py`](run.py)
  - [`tests/test_tft_profiles.py`](tests/test_tft_profiles.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
- Commands:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_profiles.py \
    -k "import_does_not_mutate_env_vars or debug_on_off_match or invalid_shape_still_fails"

  PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_comprehensive.py \
    -k "gated_temporal_backbone_component or hybrid_temporal_backbone_component or positional_multihead_attention_component"
  ```

- Result:
  - `3 passed, 12 deselected, 1 warning in 4.57s`
  - `3 passed, 62 deselected in 3.82s`
- Notes:
  - import-time mutation of ROCm/MIOpen environment variables is removed from the reusable model module;
  - expensive repeated finite-value reductions are now gated behind `tft_debug_checks`;
  - cheap schema/shape correctness checks remain on by default.

### EV-IMP-020 — Attention-probability dropout implementation

- Task:
  - `TFT-A10`
- Date: 2026-07-28
- Files:
  - [`utils/tft_config.py`](utils/tft_config.py)
  - [`run.py`](run.py)
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
- Commands:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_comprehensive.py \
    -k "sdpa_attention_backend_matches_exact or attention_probability_dropout_changes_training_output or attention_probability_dropout_is_deterministic_in_eval or attention_dropout_exact_and_sdpa_eval_match or positional_multihead_attention_component"
  ```

- Result:
  - `5 passed, 63 deselected in 3.85s`
- Notes:
  - attention probability dropout is now distinct from output projection dropout;
  - exact attention applies probability dropout only during training;
  - SDPA uses matching probability-dropout semantics in training and zero dropout in evaluation;
  - exported interpretation weights remain pre-dropout probabilities.

### EV-IMP-021 — Interpretation export naming and caveat contract

- Task:
  - `TFT-A06`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`utils/tft_interpretation.py`](utils/tft_interpretation.py)
  - [`tests/test_tft_interpretation_and_exp.py`](tests/test_tft_interpretation_and_exp.py)
- Commands:

  ```bash
  PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_interpretation_and_exp.py \
    -k "interpretation_summary_and_export or static_interpretation_uses_feature_names or canonical_profile_sets_interpretation_flags_and_named_vsn_entries or quantile_loss_is_selectable"
  ```

- Result:
  - `4 passed, 2 deselected in 2.98s`
- Notes:
  - interpretation summaries now export observed/known/history/future feature names rather than flat VSN indices;
  - history/future VSN summaries now expose explicit feature-axis labels and named top-k entries;
  - payloads now carry caveat flags indicating when canonical attribution claims are unsafe;
  - canonical and extended profiles are both regression-tested against the export contract.

### EV-IMP-002 — Structured TFT production output

- Task:
  - `TFT-O01`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`exp/exp_long_term_forecasting.py`](exp/exp_long_term_forecasting.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_comprehensive.py \
    -k "structured or quantile or moe or train_smoke or interpretation_summary or reduction or tft_path_never_uses_f_dim or nonlast or noncontiguous"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_tft_scripts_smoke.py \
    tests/test_revin_ablation.py

  python3 run_tests.py -q tests/test_tft_experiment_contracts.py
  ```

- Result:
  - focused structured-output suite: `19 passed, 42 deselected, 1 warning`
  - baseline native suite: `55 passed, 1 warning`
  - secondary baseline suite: `10 passed`
  - proxy runner: `6 passed`
- Notes:
  - `TemporalFusionTransformer` now supports `return_auxiliary=True` with a gatherable named-tuple output.
  - The long-term experiment consumes quantiles and MoE sufficient statistics from the forward result instead of `last_*` attributes.
  - Legacy tensor callers and interpretation payloads still work; mutable `last_*` fields remain only as compatibility diagnostics.

### EV-IMP-003 — Resolved TFT schema validation

- Task:
  - `TFT-C08`
- Date: 2026-07-28
- Files:
  - [`utils/tft_schema.py`](utils/tft_schema.py)
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`data_provider/data_loader.py`](data_provider/data_loader.py)
  - [`run.py`](run.py)
  - [`tests/test_tft_core_contracts.py`](tests/test_tft_core_contracts.py)
  - [`tests/test_tft_interpretation_and_exp.py`](tests/test_tft_interpretation_and_exp.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
  - [`tests/test_tft_e2e.py`](tests/test_tft_e2e.py)
  - [`scripts/long_term_forecast/tft_dummy_40cov_4target_example.py`](scripts/long_term_forecast/tft_dummy_40cov_4target_example.py)
  - [`scripts/long_term_forecast/tft_ablation_full_attention_vs_vsn_bypass.py`](scripts/long_term_forecast/tft_ablation_full_attention_vs_vsn_bypass.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_core_contracts.py -k schema

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    tests/test_tft_interpretation_and_exp.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_bugfix_diagnostics.py \
    tests/test_tft_scripts_smoke.py \
    tests/test_revin_ablation.py

  python3 run_tests.py -q tests/test_tft_core_contracts.py
  ```

- Result:
  - schema-focused checks: `3 passed, 7 deselected`
  - core/experiment/interpretation checks: `19 passed, 1 warning`
  - main native suite: `55 passed, 1 warning`
  - secondary baseline suite: `10 passed`
  - proxy runner: `10 passed`
- Notes:
  - explicit role mappings now override dataset registry defaults;
  - ETT single-feature mode resolves observed/target positions to channel `0`;
  - custom-known mode now requires both width and feature names;
  - detailed frequencies such as `15min` resolve through the time-feature utility rather than a short-code dictionary;
  - encoder known-feature time length is now validated against `seq_len`.

### EV-IMP-004 — Quantile contract and RevIN ordering repair

- Task:
  - `TFT-C03`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`exp/exp_long_term_forecasting.py`](exp/exp_long_term_forecasting.py)
  - [`utils/losses.py`](utils/losses.py)
  - [`utils/metrics.py`](utils/metrics.py)
  - [`layers/StandardNorm.py`](layers/StandardNorm.py)
  - [`tests/test_tft_core_contracts.py`](tests/test_tft_core_contracts.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
  - [`tests/test_tft_interpretation_and_exp.py`](tests/test_tft_interpretation_and_exp.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
  - [`tests/test_tft_e2e.py`](tests/test_tft_e2e.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py \
    -k quantile

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_comprehensive.py

  git diff --check
  ```

- Result:
  - quantile-focused contract suite: `8 passed, 18 deselected`
  - interpretation/experiment/native suite: `56 passed, 1 warning`
  - diff hygiene: clean
- Notes:
  - `tft_output_mode` is now explicit: `point`, `quantile`, or `joint`;
  - quantiles are canonicalized once and duplicate levels are rejected;
  - quantile mode now requires an exact `0.5` slice and uses that trained median as the reported point forecast;
  - the quantile head uses an ordered base-plus-positive-increments parameterization;
  - RevIN uses a positive effective affine scale with legacy-checkpoint migration;
  - joint mode validates positive coefficients at experiment construction;
  - `test()` now persists calibration artifacts via `quantile_metrics.json` and `quantile_pred.npy`.

### EV-IMP-005 — Static-path preservation and recurrent state-order repair

- Tasks:
  - `TFT-C01`
  - `TFT-C02`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`tests/test_tft_core_contracts.py`](tests/test_tft_core_contracts.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
  - [`tests/test_tft_interpretation_and_exp.py`](tests/test_tft_interpretation_and_exp.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_core_contracts.py -k "static or state_order"

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_comprehensive.py

  git diff --check
  ```

- Result:
  - focused static/state suite: `6 passed, 13 deselected`
  - core plus experiment suite: `33 passed, 5 warnings`
  - interpretation/experiment/native suite: `56 passed, 1 warning`
  - diff hygiene: clean
- Notes:
  - raw encoder values are now split before normalization and declared static channels are validated for time constancy;
  - static embeddings now receive raw per-sample values instead of normalized zeroed channels;
  - manual normalization and RevIN both preserve distinguishable static values;
  - identical dynamics with different static values now produce different static contexts;
  - LSTM and hybrid backbones now receive static recurrent state in PyTorch’s required `(h_0, c_0)` order.

### EV-IMP-006 — FFT top-amplitude safety and batch invariance repair

- Task:
  - `TFT-C04`
- Date: 2026-07-28
- Files:
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`tests/test_tft_extension_contracts.py`](tests/test_tft_extension_contracts.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
  - [`tests/test_tft_interpretation_and_exp.py`](tests/test_tft_interpretation_and_exp.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_extension_contracts.py -k fft

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py

  git diff --check
  ```

- Result:
  - focused FFT contract suite: `4 passed`
  - interpretation/experiment/native suite: `56 passed, 1 warning`
  - diff hygiene: clean
- Notes:
  - `top_amplitude` now selects bins per sample and latent channel instead of averaging over the batch;
  - physical FFT bins are used only to gather/scatter spectral coefficients;
  - learned complex weights are bound by selected-rank position, so dominant bins above `modes` cannot index outside the weight table;
  - batch permutation only permutes outputs and adding unrelated samples no longer changes an existing sample’s FFT branch result.

### EV-IMP-007 — Higher-order interaction gate/term repair

- Task:
  - `TFT-C05`
- Date: 2026-07-28
- Files:
  - [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
  - [`tests/test_tft_extension_contracts.py`](tests/test_tft_extension_contracts.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
  - [`tests/test_tft_interpretation_and_exp.py`](tests/test_tft_interpretation_and_exp.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_extension_contracts.py -k higher_order

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_comprehensive.py \
    tests/test_tft_interpretation_and_exp.py

  git diff --check
  ```

- Result:
  - focused higher-order contract suite: `5 passed, 4 deselected`
  - interpretation/experiment/native suite: `56 passed, 1 warning`
  - diff hygiene: clean
- Notes:
  - the interaction block now creates exactly one gate per actual term: one for pairwise interactions and, for order 3, one additional gate for the triple term;
  - gates are now independent sigmoids rather than a softmax that forced irrelevant normalization across mismatched term counts;
  - order 2 is now genuinely gate-dependent and order 3 no longer crashes on a gate/term shape mismatch;
  - interpretation payloads now expose gate shape `[B, T, interaction_order - 1]`.

### EV-IMP-008 — Static interpretation payload preservation

- Task:
  - `TFT-C06`
- Date: 2026-07-28
- Files:
  - [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
  - [`utils/tft_interpretation.py`](utils/tft_interpretation.py)
  - [`tests/test_tft_core_contracts.py`](tests/test_tft_core_contracts.py)
  - [`tests/test_tft_interpretation_and_exp.py`](tests/test_tft_interpretation_and_exp.py)
  - [`tests/test_tft_experiment_contracts.py`](tests/test_tft_experiment_contracts.py)
  - [`tests/test_tft_comprehensive.py`](tests/test_tft_comprehensive.py)
- Commands:

  ```bash
  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_core_contracts.py -k static_interpretation

  PYTHONPATH=. ai_env/bin/pytest -q tests/test_tft_interpretation_and_exp.py -k static_interpretation

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_core_contracts.py \
    tests/test_tft_experiment_contracts.py

  PYTHONPATH=. ai_env/bin/pytest -q \
    tests/test_tft_interpretation_and_exp.py \
    tests/test_tft_comprehensive.py

  git diff --check
  ```

- Result:
  - focused static payload contract: `1 passed, 19 deselected`
  - focused interpretation summary contract: `1 passed, 4 deselected`
  - core plus experiment suite: `34 passed, 6 warnings`
  - interpretation/experiment/native suite: `57 passed, 2 warnings`
  - diff hygiene: clean
- Notes:
  - static VSN payloads are now preserved per context key (`c_s`, `c_c`, `c_h`, `c_e`) instead of being flattened to `None`;
  - static graph-attention metadata now follows the same nested structure;
  - interpretation payloads now expose `static_feature_names`;
  - exported summaries now map static variable importance entries to schema feature names instead of anonymous indices.

## 14. Decisions

| Decision | Date | Reason | Affected tasks |
|---|---|---|---|
| Native TFT only | 2026-07-28 | Nixtla implementation is explicitly out of scope. | All |
| Fail fast for short-term in first safety release | 2026-07-28 | Honest rejection is safer than inventing markless semantics. | `TFT-C07` |
| Physics losses live outside the model | 2026-07-28 | They need dataset units/scalers and should be reusable. | `PHY-*` |
| Physical units are mandatory | 2026-07-28 | Dataset-standardized residuals are not physical laws. | `PHY-S01`, `PHY-L01` |
| No free-form expressions | 2026-07-28 | Prevents unsafe and non-reproducible configuration. | `PHY-C01` |
| Zero weight bypasses training physics | 2026-07-28 | Guarantees baseline parity and avoids `0 * NaN`. | `PHY-I01`, `PHY-T02` |
| Softmax VSN does not use L1 | 2026-07-28 | Its L1 norm is constant. | `THY-V01` |
| Synthetic proof precedes ETTh1 claims | 2026-07-28 | ETTh1 has no supplied governing law. | `PHY-V01`, `PHY-V02` |
| Planetary ephemerides are known covariates, not a NIFTY governing law | 2026-07-29 | Orbital physics determines the inputs but supplies no accepted equation for market returns. | `AST-*` |
| Raw returns precede raw OHLC levels | 2026-07-29 | Price persistence would obscure the incremental-signal test; structured OHLC is a later secondary task. | `AST-H01`, `AST-O01` |
| Real ephemerides require matched smooth nulls | 2026-07-29 | Row shuffling destroys autocorrelation and creates an unfairly weak placebo. | `AST-N01`, `AST-E01` |
| Do not enable per-feature VSN gating yet | 2026-07-29 | A direct forward reproduction currently fails at `len(self.variable_grns)` when the field is `None`. | `AST-C00` |
| Preserve old completion but open semantic v2 tasks | 2026-07-31 | Trained-checkpoint audit found new semantics outside the historical task acceptance scope. | `TFT-SR00`–`TFT-SR09` |
| No neural astrology training before `G2-SR` | 2026-07-31 | The next substantial run must not inherit known seed, neutrality, lag, compression, graph, or interaction defects. | `TFT-SR09`, `AST-*` |
| First astrology model keeps generic extensions off | 2026-07-31 | Correct implementation does not imply support from a small 1995-onward dataset. | `AST-B01`, `AST-E00` |

## 15. Change Log

| Date | Task | Old status | New status | Reason |
|---|---|---|---|---|
| 2026-07-28 | `GOV-001` | — | `DONE` | Native implementation audit completed. |
| 2026-07-28 | `GOV-002` | — | `DONE` | Physics design made implementation-ready. |
| 2026-07-28 | `BASE-001` | — | `DONE` | Focused baseline commands passed. |
| 2026-07-28 | `GOV-003` | — | `DONE` | Tracker/orchestrator system created. |
| 2026-07-28 | `TFT-H01` | `READY` | `DONE` | Discovery path fixed; pytest wrapper and deep-benchmark controls validated. |
| 2026-07-28 | `TFT-C07` | `READY` | `DONE` | Native TFT now rejects unsupported short-term construction clearly. |
| 2026-07-28 | `TFT-C09` | `READY` | `DONE` | Shared target mapping added to train/validation/test/inverse paths. |
| 2026-07-28 | `TFT-O01` | `READY` | `DONE` | Production long-term TFT now uses a gatherable structured output contract. |
| 2026-07-28 | `TFT-C08` | `READY` | `DONE` | Schema validation now resolves roles, custom-known features, and detailed frequencies consistently. |
| 2026-07-28 | `TFT-C03` | `READY` | `DONE` | Quantile training, ordered decoding, median evaluation, RevIN ordering, and calibration metrics are now aligned. |
| 2026-07-28 | `TFT-C01` | `READY` | `DONE` | Static values are now preserved across normalization and validated before embedding. |
| 2026-07-28 | `TFT-C02` | `READY` | `DONE` | Recurrent static contexts now initialize LSTM and hybrid backbones in `(h_0, c_0)` order. |
| 2026-07-28 | `TFT-C04` | `READY` | `DONE` | Top-amplitude FFT selection now uses safe rank weights and no longer depends on unrelated batch members. |
| 2026-07-28 | `TFT-C05` | `READY` | `DONE` | Higher-order interaction now uses one independent gate per real term and order-3 no longer has a gate/term mismatch. |
| 2026-07-28 | `TFT-C06` | `READY` | `DONE` | Static interpretation payloads now preserve per-context metadata and export static feature names. |
| 2026-07-29 | `GOV-004` | — | `DONE` | Planetary/NIFTY hypothesis lane audited and documented. |
| 2026-07-29 | `AST-H01` | — | `READY` | Native `G2` is passed; protocol can be frozen after the data-format discussion. |
| 2026-07-29 | `AST-D01` | — | `BLOCKED` | Representative data and generation/provenance details are not yet present in the workspace. |
| 2026-07-29 | `AST-C00` | — | `READY` | Optional per-feature VSN gating crash was directly reproduced. |
| 2026-07-31 | `TFT-SR00`–`TFT-SR09` | — | `NOT_STARTED` | Post-matrix audit opened a distinct semantic-v2 repair wave without rewriting historical `DONE` evidence. |
| 2026-07-31 | `G2-SR` | — | `OPEN` | Native semantic readiness now requires `TFT-SR09`. |
| 2026-07-31 | `AST-C00` | `READY` | `NOT_STARTED` | Repair ownership moved into native `TFT-SR06`; the AST task accepts that evidence instead of duplicating code. |
| 2026-07-31 | Native feature matrix | `13/14 ACTIVE` | `14/14 COMPLETE` | Final experimental profile wrote result/checkpoint artifacts at 23:22 IST. |
| 2026-07-31 | `TFT-SR00` | `NOT_STARTED` | `READY` | Matrix-exit dependency is satisfied; legacy-v1 inventory may begin. |
| 2026-07-31 | `AST-D01` | `BLOCKED` | `IN_PROGRESS` | Supplied CSVs made the read-only source audit executable. |
| 2026-07-31 | `AST-D01` | `IN_PROGRESS` | `WAITING_EXTERNAL` | Audit completed, but mixed market dates and absent raw OHLC/generator prevent acceptance. |
| 2026-08-01 | `TFT-SR00` | `IN_PROGRESS` | `DONE` | Immutable 14-case/65-record legacy inventory, semantic identities, bound checkpoint metadata, guarded legacy replay, and 158-test regression evidence passed. |
| 2026-08-01 | `TFT-SR01` | `NOT_STARTED` | `IN_PROGRESS` | `TFT-SR00` dependency passed; reproducibility and paired-ablation repair claimed by Codex `/root`. |
| 2026-08-01 | `TFT-SR01` | `IN_PROGRESS` | `DONE` | Independent adversarial audit found no blocker; exact paired state/order, separated seed streams, truthful manifests, and validation-only fitting passed with `EV-IMP-026`. |
| 2026-08-01 | `TFT-SR02` | `NOT_STARTED` | `IN_PROGRESS` | `TFT-SR01` dependency passed; exact-neutral adapter and physical-coordinate contract claimed by Codex `/root`. |
| 2026-08-01 | `TFT-SR02` | `IN_PROGRESS` | `DONE` | Exact-neutral adapter and physical-coordinate contracts validated in semantic regression closeout (`EV-IMP-027`). |
| 2026-08-01 | `TFT-SR03` | `NOT_STARTED` | `DONE` | FFT v2 scope/diagnostic semantics repaired and validated (`EV-IMP-027`). |
| 2026-08-01 | `TFT-SR04` | `NOT_STARTED` | `DONE` | Explicit cross-attention semantics and interpretation labeling repaired and validated (`EV-IMP-027`). |
| 2026-08-01 | `TFT-SR05` | `NOT_STARTED` | `DONE` | Lag semantics split/release, digest compatibility, and regression checks validated (`EV-IMP-027`). |
| 2026-08-01 | `TFT-SR06` | `NOT_STARTED` | `READY` | Upstream dependency `TFT-SR02` is now complete; task is available for claim. |
| 2026-08-01 | `TFT-SR06` | `READY` | `DONE` | Latent-polynomial semantic separation, named pre-VSN interaction interface, and per-feature VSN liveness validation passed (`EV-IMP-028`). |
| 2026-08-01 | `TFT-SR07` | `NOT_STARTED` | `READY` | Upstream dependency `TFT-SR02` is complete and SR06 closeout is recorded; task is available for claim. |
| 2026-08-01 | `TFT-SR07` | `READY` | `DONE` | Temporal compression mode redesign, short-window guardrails, pooled K/V coordinate diagnostics, and regression slice passed (`EV-IMP-029`). |
| 2026-08-01 | `TFT-SR08` | `NOT_STARTED` | `READY` | Upstream dependency `TFT-SR02` remains complete and SR07 closeout is recorded; task is available for claim. |
| 2026-08-01 | `TFT-SR08` | `READY` | `DONE` | Graph cross-mixing propagation, neutral no-op compatibility, and graph metadata validation passed (`EV-IMP-030`). |
| 2026-08-01 | `TFT-SR09` | `NOT_STARTED` | `READY` | Upstream dependency `TFT-SR00`–`TFT-SR08` is now complete; task is available for claim. |
| 2026-08-01 | `TFT-SR09` | `READY` | `DONE` | Post-matrix semantic release gate passed with the full SR02-SR09 regression slice and diff hygiene (`EV-IMP-031`). |
| 2026-08-01 | `G2-SR` | `OPEN` | `PASSED` | Semantic-release wave completed and evidence recorded through `EV-IMP-031`. |

## 16. Update Checklist

At the end of every implementation session:

- [ ] Update the task status, owner, timestamp, evidence, and next action.
- [ ] Add exact test commands and results to an evidence entry.
- [ ] Add any design decision that changes scope or behavior.
- [ ] Release the primary-file lock.
- [ ] Promote newly unblocked tasks from `NOT_STARTED` to `READY`.
- [ ] Recalculate weighted progress.
- [ ] Re-evaluate the active gate.
- [ ] Add one change-log row per status transition.
