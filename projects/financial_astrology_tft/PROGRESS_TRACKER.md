# Financial Astrology TFT Progress Tracker

> Canonical source for task status. Plans describe intended work; this file says
> what has actually happened.
>
> Updated: 2026-08-01 14:05 Asia/Kolkata.

## 1. Status Vocabulary

| Status | Meaning |
|---|---|
| `NOT_STARTED` | No implementation/evidence exists |
| `DISCUSSION` | Theory or requirement is being resolved |
| `READY` | Dependencies satisfied and work may be claimed |
| `IN_PROGRESS` | An agent has recorded ownership and started work |
| `WAITING_EXTERNAL` | Waiting for a process, artifact, or user input |
| `BLOCKED` | A named blocker prevents progress |
| `VERIFYING` | Implementation exists; acceptance evidence is incomplete |
| `COMPLETE` | Acceptance criteria and evidence are recorded |
| `RETIRED` | Deliberately abandoned with reason |

`COMPLETE` is evidence-based. A written plan is not evidence of implementation.

## 2. Project Dashboard

| Gate | State | Exit requirement |
|---|---|---|
| G0 Project control | `COMPLETE` | Canonical docs and resume protocol exist |
| G1 Native TFT matrix | `COMPLETE` | All 14 result/checkpoint directories exist and final metrics are recorded |
| G1R Native semantic release | `IN_PROGRESS` | `TFT-SR00`–`TFT-SR09` pass; one deterministic micro-run recorded |
| G2 Theory/data freeze | `DISCUSSION` | Convention/hypothesis frozen; corrected market data and reproducible generator pass acceptance |
| G3 Loader/baselines | `NOT_STARTED` | Leak-free loader and non-planet baselines pass |
| G4 Feature/null engine | `NOT_STARTED` | Typed features, events, and false ephemerides pass |
| G5 Cheap falsification | `NOT_STARTED` | Linear/flat-TFT development screen recorded |
| G6 Specialized model | `NOT_STARTED` | Response/graph/multiclock architecture earns advancement |
| G7 Separate extensions | `NOT_STARTED` | Anchored, outer, probabilistic/OHLC arms isolated |
| G8 Locked evaluation | `NOT_STARTED` | Frozen run and evidence classification complete |

## 3. Active and Waiting Work

| Task | Status | Owner | Started | Blocker/next action |
|---|---|---|---|---|
| `FA-TFT-001` | `COMPLETE` | — | earlier session | 14/14 complete at 2026-07-31 23:22 IST; legacy-v1 artifact inventory passes to `TFT-SR00` |
| `FA-TFT-SEM-001` | `IN_PROGRESS` | Codex `/root` | 2026-08-01 09:36 IST | `TFT-SR00`/`TFT-SR01` complete with root `EV-IMP-025`/`EV-IMP-026`; `TFT-SR02` exact-neutrality/coordinate repair active |
| `FA-THEORY-001` | `DISCUSSION` | unclaimed | 2026-07-29 | Resolve `OPEN-*`, target, and interval choices with user |
| `FA-DATA-001` | `WAITING_EXTERNAL` | unclaimed | 2026-07-31 23:21 IST | Read-only audit complete; supply raw session-dated OHLC and PySwissEph generator/convention package |

The project plan and native repair plan are authorized. The first source audit
is complete but cannot close until corrected market provenance and the
ephemeris generator package arrive. Financial-astrology neural training remains
gated.

## 4. Stable Task Registry

| Task | Status | Dependencies | Evidence required |
|---|---|---|---|
| `FA-GOV-001` | `COMPLETE` | none | Canonical project documents |
| `FA-TFT-001` | `COMPLETE` | completed matrix | `EV-TFT-001-COMPLETE`; 14 result and checkpoint directories |
| `FA-TFT-SEM-001` | `IN_PROGRESS` | `FA-TFT-001` complete | Root `TFT-SR00`–`TFT-SR09` ledger and G2-SR evidence |
| `FA-THEORY-001` | `DISCUSSION` | user choices | Frozen convention + hypothesis hashes |
| `FA-DATA-001` | `WAITING_EXTERNAL` | raw OHLC/session keys and generator/convention package | `DATA_AUDIT_REPORT.md` plus passing regenerated-data tests |
| `FA-LEAK-001` | `NOT_STARTED` | `FA-DATA-001` | Prefix-invariance/leakage tests |
| `FA-LOAD-001` | `NOT_STARTED` | theory, data, leakage, semantic release | Named known-future batch and interval tests |
| `FA-BASE-001` | `NOT_STARTED` | loader, semantic release | Reproducible B00–B05 scorecard with identical folds/seeds |
| `FA-FEAT-001` | `NOT_STARTED` | theory, data | Formula and boundary tests |
| `FA-EVENT-001` | `NOT_STARTED` | theory, data | Independent event fixtures |
| `FA-NULL-001` | `NOT_STARTED` | features/events | Null quality and determinism tests |
| `FA-SCREEN-001` | `NOT_STARTED` | baselines/features/nulls | Development-only paired results |
| `FA-MEM-001` | `NOT_STARTED` | screen gate | Irregular-time recurrence tests/results |
| `FA-ENC-001` | `NOT_STARTED` | memory/screen gate | Disabled parity, gradients, matched capacity |
| `FA-MULTI-001` | `NOT_STARTED` | encoder/memory gate | Clock ablations and matched comparison |
| `FA-ANCHOR-001` | `BLOCKED` | anchor decision + core model | Frozen anchor/time policy |
| `FA-OUTER-001` | `NOT_STARTED` | core model | Separate modern profile results |
| `FA-PROB-001` | `NOT_STARTED` | stable point model | Pinball/coverage/width/crossing |
| `FA-OHLC-001` | `NOT_STARTED` | stable primary model | Valid-bar reconstruction tests |
| `FA-LOCK-001` | `NOT_STARTED` | frozen selected model | One-time lockbox evidence |
| `FA-REPORT-001` | `NOT_STARTED` | locked evaluation | Allowed evidence classification |

## 5. Evidence Log

### `EV-TFT-SR00-001` — Native semantic-version boundary

```text
date: 2026-08-01
root_task: TFT-SR00
root_evidence: EV-IMP-025
state: COMPLETE
legacy_cases: 14
verified_records: 65
native_regression: 158 passed, 37 warnings
next_root_task: TFT-SR02 (IN_PROGRESS)
```

The legacy producer, explicit-v1 replay script, dataset, results, and
checkpoints are separately hashed. Repaired identities are versioned; pending
operators cannot emit v2 artifacts; versioned checkpoints bind the exact state
file and resolved schema/config; historical no-version checkpoints require the
explicit legacy path.

### `EV-TFT-SR01-001` — Native reproducibility and paired-ablation boundary

```text
date: 2026-08-01
root_task: TFT-SR01
root_evidence: EV-IMP-026
state: COMPLETE
combined_gate: 101 passed, 23 warnings
independent_review: no remaining blocker; 65 focused and 89 broader passed
legacy_records_reverified: 65
next_root_task: TFT-SR02 (IN_PROGRESS)
```

Production now uses independent deterministic seed streams, train-only stable
sample order, validation-only fitting, exact shared-state paired construction,
and content-/sample-addressed fold manifests. Pair reports and canonical
reference controls are tamper-evident; the repeated paired micro-run preserves
shared/checkpoint/order hashes and a changed seed changes both initialization
and sample order.

### `EV-GOV-001` — Cross-session project scaffold

```text
date: 2026-07-29
task: FA-GOV-001
state: COMPLETE
evidence:
  README.md
  CURRENT_STATUS.md
  IMPLEMENTATION_PLAN.md
  PROGRESS_TRACKER.md
  ORCHESTRATOR.md
  HYPOTHESIS_REGISTRY.md
  DATA_CONTRACT.md
  ASTROLOGY_FEATURE_SPEC.md
  DECISIONS.md
  RISKS_VALIDITY.md
  RESULTS_SCORECARD.md
  SESSION_HANDOFF.md
```

This evidence establishes planning/governance only. It is not model
implementation evidence.

### `EV-TFT-001-COMPLETE` — Legacy-v1 feature matrix

```text
date: 2026-07-31 23:22 Asia/Kolkata
task: FA-TFT-001
state: COMPLETE
matrix_process: exited
final_case: tft_ot_p24_experimental_profile
complete_cases: 14 of 14
result_directories: 14
checkpoint_directories: 14
```

Final ranking:

| Case | MSE | MAE | Notes |
|---|---:|---:|---|
| quantile-only | 0.03435775 | 0.14262331 | Leader; 71.91% coverage for nominal 80% interval |
| covariate reattention | 0.03445451 | 0.14307196 | Promising single seed; semantics/seeds still need validation |
| ALiBi | 0.03467850 | 0.14418337 | Third by MSE |
| regime MoE | 0.03547186 | 0.14576912 | Near neutral versus reference |
| baseline | 0.03568881 | 0.14699268 | Reference |
| SDPA | 0.03568881 | 0.14699268 | Exact prediction parity |
| joint quantile | 0.03587884 | 0.14622945 | 68.04% coverage for nominal 80% interval |
| learned FFT | 0.03607463 | 0.14592449 | +1.08% MSE; selector semantics require repair |
| cross-attention | 0.03616716 | 0.14684366 | +1.34% MSE; semantic/neutrality repair required |
| higher-order | 0.03687086 | 0.14817350 | +3.31% MSE; not named covariate interactions |
| lag attention | 0.03693667 | 0.14863680 | +3.50% MSE; shifted-prefix semantics |
| temporal compression | 0.03730480 | 0.15012075 | +4.53% MSE; compression/decompression defects |
| sparse cross-mixing | 0.03858219 | 0.15343992 | +8.11% MSE; graph semantics/integration defects |
| experimental profile | 0.04245717 | 0.16271803 | +18.96% MSE; stacked profile is not a clean component ablation |

Evidence locations:

```text
result_long_term_forecast.txt
results/long_term_forecast_tft_ot_p24_*/
checkpoints/long_term_forecast_tft_ot_p24_*/
```

The final case also recorded pinball `0.05300126`, nominal-80% coverage
`0.67665675`, width `0.41057798`, and zero crossing. `FA-TFT-001` is complete
as execution evidence; its semantic limitations remain the reason for
`FA-TFT-SEM-001`.

### `EV-DATA-001-BLOCKING-AUDIT` — Supplied NIFTY/planetary artifacts

```text
date: 2026-07-31
task: FA-DATA-001
state: WAITING_EXTERNAL
planet_rows_columns: 18,251 x 88
return_rows_columns: 7,109 x 6
planet_sha256: 3036dc79790fed35d3a5f9cf2b4e7726bb00eec7915b54f839aba4a99671fa6d
returns_sha256: 6765711ed03964f06fec725683e28bc1e83461c36feba09181c080d9a05485b6
report: projects/financial_astrology_tft/DATA_AUDIT_REPORT.md
```

Confirmed findings:

- all 18,251 rows are structurally complete, ordered, daily, and finite;
- every direct longitude pair has unit-circle error at most `4.44e-16` and
  finite-difference motion agrees with the stored speed;
- every rashi pair for all 12 bodies/nodes implements
  `max(floor(longitude/30)-1, 0)`, so all supplied sign columns are rejected;
- the return file has mixed session-label regimes; 412 strict four-field value
  fingerprints match the following calendar day's independently retrieved bar;
- no PySwissEph generator or convention manifest exists in the current tree or
  Git history, and two local artifacts differ by exactly 8.25 hours of angular
  motion for the same displayed date;
- Shadbala is quarantined, Ketu is derived from one node axis, and no Hilbert
  feature is identifiable;
- the first provisional planet family is 37 transparent continuous columns,
  but it is not admitted until timestamp/unit/frame provenance is resolved.

No source file was mutated and no NIFTY training was launched. Completion
requires authoritative raw OHLC/session dates plus the generator/convention
package and passing rebuilt-data tests.

Read-only invariant replay after documentation updates:

```text
planet_shape: (18251, 88)
returns_shape: (7109, 6)
max_unit_circle_error: 4.440892098500626e-16
rashi_bug_rows_per_body: 18251
weekend_return_labels: 120
provisional_feature_count: 37
SHA-256 assertions: passed
git diff --check: passed
17-document fence/link validation: passed
```

### `EV-TFT-SEM-PLAN-001` — Post-matrix semantic-repair specification

```text
date: 2026-07-31
task: FA-TFT-SEM-001 (planning evidence only)
state: READY; implementation not started
root_tasks: TFT-SR00 through TFT-SR09
required_gate: G2-SR
```

The code audit found semantic defects in reproducibility, neutral extension
behavior, coordinate handling, FFT selection, lag response, cross-attention,
named interactions/per-feature VSN, temporal compression, and graph mixing.
Detailed task cards and acceptance tests are in the root
`implementation_plan.md`. This evidence proves that the repair work is specified;
it does not prove that any repair is implemented.

Planning-document validation:

```text
git diff --check: passed
local relative-link existence check: passed
Markdown fence parity: passed for every changed document
semantic task-card uniqueness: exactly one card for each TFT-SR00..TFT-SR09
```

### `EV-THEORY-001-DRAFT` — Classical source audit

The source audit identified:

- *Brihat Samhita* Chapter 42 on price fluctuations;
- Chapter 97 on different times of fruition;
- chapters on conjunctions, nakshatras, nodes/eclipses, and transits;
- BPHS material on Navagraha, dignity, and special aspects.

The resulting candidate families are in `HYPOTHESIS_REGISTRY.md`. They remain
draft until convention and formula choices are frozen.

## 6. Blockers

| Blocker | Affects | Resolution |
|---|---|---|
| Native advanced-feature semantic gate open | All neural NIFTY work | Complete `TFT-SR00`–`TFT-SR09` and G2-SR |
| Raw OHLC/session keys absent; derived return dates are mixed | Data, loader, targets | Supply authoritative raw NIFTY OHLC or exact retrieval script/artifact; rebuild rather than relabel |
| Generator/convention provenance absent | Data, loader, features | Supply PySwissEph code, versions, ephemeris source/hash, flags, timestamp/timezone, frame, and ayanamsha |
| Supplied rashi fields invalid | Rashi/event features | Keep rejected; regenerate from audited longitude only after convention freeze and boundary tests |
| Shadbala generator absent | Shadbala arm | Keep quarantined or supply complete formula/location/time provenance |
| Ayanamsha/node/timestamp conventions unresolved | Every astro feature | Theory discussion and manifest |
| Anchor chart unresolved | Anchored mundane model | Predeclare or omit |
| Single NIFTY history has <1 full outer orbit for Uranus/Neptune/Pluto | Strong outer-cycle claim | Limit claim; add older/multi-market data |

## 7. Next-Task Rule

Current execution boundary:

```text
allowed:
    read-only TFT status inspection
    continue claimed TFT-SR02, then the remaining native semantic wave
    theory discussion
    receive/remediate source data without overwriting audited artifacts
    resume FA-DATA-001 when raw OHLC and generator package arrive
    documentation corrections

not allowed:
    financial-astrology loader/model implementation
    NIFTY neural training before FA-TFT-SEM-001 and loader/baseline gates
    killing/restarting the matrix
    choosing rules from final-test outcomes
```

The matrix has exited and `TFT-SR00`/`TFT-SR01` passed, so active `TFT-SR02` under
`FA-TFT-SEM-001` is the current native code task. `FA-DATA-001` resumes in parallel only after its missing source
package arrives. The planetary encoder is not the first astrology code task.

## 8. Change Log

| Date | Change |
|---|---|
| 2026-07-29 | Created separate cross-session project tracker |
| 2026-07-29 | Recorded live TFT matrix partial evidence |
| 2026-07-29 | Replaced 128-day total-memory assumption with multi-clock architecture |
| 2026-07-29 | Added distinct classical-text and modern-outer theory lanes |
| 2026-07-31 | Corrected matrix cardinality to 14; recorded 13 complete and final case active |
| 2026-07-31 | Added `FA-TFT-SEM-001` and mandatory `TFT-SR00`–`TFT-SR09` release gate |
| 2026-07-31 | Authorized data intake in parallel but prohibited neural NIFTY training before semantic/loader/baseline gates |
| 2026-07-31 | Closed `FA-TFT-001` at 14/14; final experimental profile recorded 0.04245717 MSE |
| 2026-07-31 | Moved `FA-TFT-SEM-001` to `READY`; `TFT-SR00` may now freeze legacy-v1 artifacts |
| 2026-07-31 | Moved `FA-DATA-001` to `WAITING_EXTERNAL` after the blocking source audit; rejected rashi fields and mixed session dates documented |
| 2026-08-01 | Closed root `TFT-SR00` with `EV-IMP-025`; 65 legacy records and 158 native tests passed; claimed `TFT-SR01` |
| 2026-08-01 | Closed root `TFT-SR01` with `EV-IMP-026` after independent review; claimed `TFT-SR02` |
