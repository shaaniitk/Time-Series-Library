# Financial Astrology TFT Progress Tracker

> Canonical source for task status. Plans describe intended work; this file says
> what has actually happened.
>
> Updated: 2026-07-29.

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
| G1 Native TFT reference | `WAITING_EXTERNAL` | Feature matrix closes and reference is frozen |
| G2 Theory/data freeze | `DISCUSSION` | Convention, hypothesis, sample, and generator audited |
| G3 Loader/baselines | `NOT_STARTED` | Leak-free loader and non-planet baselines pass |
| G4 Feature/null engine | `NOT_STARTED` | Typed features, events, and false ephemerides pass |
| G5 Cheap falsification | `NOT_STARTED` | Linear/flat-TFT development screen recorded |
| G6 Specialized model | `NOT_STARTED` | Response/graph/multiclock architecture earns advancement |
| G7 Separate extensions | `NOT_STARTED` | Anchored, outer, probabilistic/OHLC arms isolated |
| G8 Locked evaluation | `NOT_STARTED` | Frozen run and evidence classification complete |

## 3. Active and Waiting Work

| Task | Status | Owner | Started | Blocker/next action |
|---|---|---|---|---|
| `FA-TFT-001` | `WAITING_EXTERNAL` | user terminal | earlier session | Let matrix finish naturally; inspect read-only |
| `FA-THEORY-001` | `DISCUSSION` | unclaimed | 2026-07-29 | Resolve `OPEN-*` choices with user |
| `FA-DATA-001` | `WAITING_EXTERNAL` | unclaimed | — | Need representative data and generator |

No financial-astrology implementation task is currently authorized or in
progress.

## 4. Stable Task Registry

| Task | Status | Dependencies | Evidence required |
|---|---|---|---|
| `FA-GOV-001` | `COMPLETE` | none | Canonical project documents |
| `FA-TFT-001` | `WAITING_EXTERNAL` | live matrix | Complete metrics/rank/reference decision |
| `FA-THEORY-001` | `DISCUSSION` | user choices | Frozen convention + hypothesis hashes |
| `FA-DATA-001` | `WAITING_EXTERNAL` | sample/generator | Audit report + passing data tests |
| `FA-LEAK-001` | `NOT_STARTED` | `FA-DATA-001` | Prefix-invariance/leakage tests |
| `FA-LOAD-001` | `NOT_STARTED` | theory, data, leakage | Named known-future batch tests |
| `FA-BASE-001` | `NOT_STARTED` | loader, TFT reference | Reproducible B00–B05 scorecard |
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

### `EV-TFT-001-PARTIAL` — Live feature matrix

```text
date: 2026-07-29 21:07 Asia/Kolkata
task: FA-TFT-001
state: PARTIAL / WAITING_EXTERNAL
matrix_process: alive
active_case: tft_ot_p24_xattn_interp
complete_cases: 5 of 13
```

Partial ranking:

| Case | MSE | MAE | Notes |
|---|---:|---:|---|
| quantile-only | 0.03435775 | 0.14262331 | Current leader; 71.91% coverage for nominal 80% interval |
| ALiBi | 0.03467850 | 0.14418337 | Second |
| baseline | 0.03568881 | 0.14699268 | Reference |
| SDPA | 0.03568881 | 0.14699268 | Exact prediction parity |
| joint quantile | 0.03587884 | 0.14622945 | 68.04% coverage for nominal 80% interval |

Evidence locations:

```text
result_long_term_forecast.txt
results/long_term_forecast_tft_ot_p24_*/
checkpoints/long_term_forecast_tft_ot_p24_*/
```

Do not mark `FA-TFT-001` complete from this partial evidence.

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
| TFT matrix still running | Reference configuration | Wait; read-only monitoring |
| No representative data/generator | Data, loader, features | User supplies sample and generation code |
| Ayanamsha/node/timestamp conventions unresolved | Every astro feature | Theory discussion and manifest |
| Anchor chart unresolved | Anchored mundane model | Predeclare or omit |
| Single NIFTY history has <1 full outer orbit for Uranus/Neptune/Pluto | Strong outer-cycle claim | Limit claim; add older/multi-market data |

## 7. Next-Task Rule

Until the user changes the current state:

```text
allowed:
    read-only TFT status inspection
    theory discussion
    data-format review
    documentation corrections

not allowed:
    financial-astrology loader/model implementation
    killing/restarting the matrix
    choosing rules from final-test outcomes
```

Once implementation is authorized, the first code task is `FA-DATA-001`, not the
planetary encoder.

## 8. Change Log

| Date | Change |
|---|---|
| 2026-07-29 | Created separate cross-session project tracker |
| 2026-07-29 | Recorded live TFT matrix partial evidence |
| 2026-07-29 | Replaced 128-day total-memory assumption with multi-clock architecture |
| 2026-07-29 | Added distinct classical-text and modern-outer theory lanes |
