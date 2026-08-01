# Financial Astrology TFT Research Project

> Canonical project home for testing Jyotisha- and financial-astrology-derived
> hypotheses with the repository's native `TemporalFusionTransformer`.
>
> Project state: `NATIVE SEMANTIC HARDENING / DATA REMEDIATION`.
>
> Financial-astrology model state: `NOT STARTED`; native semantic gate open.
>
> Last updated: 2026-07-31 (Asia/Kolkata).

## Purpose

This project tests a narrow, falsifiable question:

> Do preregistered astronomical and Jyotisha-derived covariates add stable
> out-of-sample information about NIFTY 50 returns, volatility, gaps, ranges, or
> drawdown risk after controlling for market history, ordinary calendar effects,
> secular time, and equally smooth placebo ephemerides?

The project is deliberately willing to model the astrological theory on its own
terms. Rashi transitions, nakshatras, retrograde loops, stations, conjunctions,
Parashari aspects, eclipses, dignity, fast triggers, slow background regimes, and
delayed fruition are therefore valid candidate inductive biases.

The evaluation remains skeptical. A positive result means that a frozen feature
family improved prediction under the declared protocol. It does not, by itself,
prove causality or validate astrology in general.

## Important Architecture Decision

`seq_len` is the high-resolution market-history window. It is **not** the
planetary orbital-memory horizon.

The project will not pretend that a 128-day tensor represents Saturn, nor will it
feed thirty years of daily rows into one LSTM. It will use separate clocks:

| Clock | Initial representation | Purpose |
|---|---|---|
| Market clock | 252 trading sessions; sensitivity at 64, 128, and 504 | Recent price dynamics |
| Fast astronomy clock | Daily calendar grid around the forecast date | Moon/Mercury and exact event timing |
| Medium astronomy clock | Weekly/event history over about five years | Retrograde loops, ingresses, conjunction episodes |
| Slow astronomy clock | Monthly state over available history plus recursive response summaries | Jupiter/Saturn/nodal background |
| Secular outer-planet clock | Current phase, relative phase, event clocks, and optional low-capacity coarse grid | Uranus/Neptune/Pluto exploratory regime state |

Orbital phase is represented directly by circular state. Alleged persistence is
represented by calendar-time response kernels and event clocks. These are
different concepts and must not be conflated.

## Theory Families

The following families must remain separate in code, configuration, results, and
claim language:

1. `CLASSICAL_TRANSIT`: Navagraha, rashi, nakshatra, retrograde/station,
   conjunction, graha drishti, dignity, combustion, nodes/eclipses, and
   panchanga without a market natal chart.
2. `CLASSICAL_PRICE_TEXT`: literal, preregistered hypotheses derived from
   *Brihat Samhita* Chapters 42 and 97.
3. `ANCHORED_MUNDANE`: transits to a frozen NIFTY/NSE/India event chart. This
   is disabled until the anchor event, time, location, and uncertainty policy
   are agreed.
4. `MODERN_OUTER`: Uranus, Neptune, Pluto, modern aspects, and modern
   financial-cycle rules. These are not to be described as classical
   Navagraha doctrine.

See [HYPOTHESIS_REGISTRY.md](HYPOTHESIS_REGISTRY.md) for the candidate rules and
their current freeze state.

## Canonical Documents

Read these in order when resuming the project:

1. [CURRENT_STATUS.md](CURRENT_STATUS.md) — live state and immediate next action.
2. [DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md) — audited source artifacts,
   confirmed defects, admitted features, and required remediation inputs.
3. [ORCHESTRATOR.md](ORCHESTRATOR.md) — mandatory agent start/finish protocol.
4. [PROGRESS_TRACKER.md](PROGRESS_TRACKER.md) — task state, blockers, and evidence.
5. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — architecture, phases, and
   acceptance gates.
6. [HYPOTHESIS_REGISTRY.md](HYPOTHESIS_REGISTRY.md) — theory definitions and
   preregistration state.
7. [DATA_CONTRACT.md](DATA_CONTRACT.md) — required input schema and anti-leakage
   rules.
8. [ASTROLOGY_FEATURE_SPEC.md](ASTROLOGY_FEATURE_SPEC.md) — exact feature
   semantics; currently a design draft.
9. [DECISIONS.md](DECISIONS.md) — append-only material decisions.
10. [RISKS_VALIDITY.md](RISKS_VALIDITY.md) — identifiability and validity risks.
11. [RESULTS_SCORECARD.md](RESULTS_SCORECARD.md) — frozen-baseline comparisons;
    the native prerequisite matrix is populated, while astrology results remain
    empty.
12. [SESSION_HANDOFF.md](SESSION_HANDOFF.md) — concise last-session state and
    exact resume boundary.

The earlier repository-level
[`Vedic_Astrology_TFT_Implementation_Plan.md`](../../Vedic_Astrology_TFT_Implementation_Plan.md)
remains a technical audit and historical design document. This directory is the
canonical cross-session project control plane.

## Current Execution Boundary

The 14-case ETTh1 native-TFT advanced-feature matrix completed at 23:22 on
2026-07-31. The final `experimental_full` case produced its checkpoint and
result artifacts; it was 18.96% worse than the baseline by MSE. `TFT-SR00` is
now ready to inventory and freeze all legacy-v1 artifacts before code semantics
change.

Its results revealed semantic defects in several advanced switches. The root
[`implementation_plan.md`](../../implementation_plan.md#14-post-matrix-native-tft-semantic-repair-wave)
therefore defines the mandatory `TFT-SR00`–`TFT-SR09` repair wave. The umbrella
project task is `FA-TFT-SEM-001`.

The supplied data has been audited. Continuous coordinates are coherent, but
the stored rashi pairs are wrong on every row, market dates use mixed session
labels, and the source generator is absent. See
[DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md). A financial-astrology neural
training run is **not** authorized
until all of the following gates close:

1. freeze the now-complete TFT matrix artifacts under `TFT-SR00`;
2. complete `TFT-SR00`–`TFT-SR09`, including neutral/no-op, semantic, gradient,
   mask, and reproducibility tests;
3. rebuild market sessions from authoritative raw OHLC and audit the PySwissEph
   generator/convention manifest;
4. define decision-time-to-target-time interval features and freeze the
   astronomical convention manifest;
5. freeze the first confirmatory hypothesis family;
6. prove the production known-future loader does not leak future market values;
7. build market-only and calendar-only baselines before enabling planets.

The next substantial training data will be NIFTY, not another exhaustive ETTh1
matrix. Native repairs use focused unit tests, synthetic known-answer fixtures,
gradient tests, and one deterministic micro-run.

## Naming

The recommended technical name is:

> Astronomy-constrained, Jyotisha-theory-informed TFT

It is not a physics-informed neural network in the strict sense. Ephemerides and
circular geometry supply physical constraints; Jyotisha supplies theory-derived
features, interactions, and lag priors. There is no established governing
equation mapping planetary state to market price.
