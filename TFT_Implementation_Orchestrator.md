# TFT, Planetary-Covariate, and Physics Implementation Orchestrator

> Version: 2
>
> Created: 2026-07-28; planetary lane added 2026-07-29
>
> Audited base: `564cffbc712f`
>
> Scope: repository-native TemporalFusionTransformer only.
>
> Dynamic state: [`TFT_Implementation_Progress.md`](TFT_Implementation_Progress.md)
>
> Implementation recipes: [`implementation_plan.md`](implementation_plan.md)
>
> Technical rationale: [`TFT_Deep_Analysis_Report.md`](TFT_Deep_Analysis_Report.md)
>
> Planetary/NIFTY implementation recipes: [`Vedic_Astrology_TFT_Implementation_Plan.md`](Vedic_Astrology_TFT_Implementation_Plan.md)
>
> Canonical financial-astrology orchestrator:
> [`projects/financial_astrology_tft/ORCHESTRATOR.md`](projects/financial_astrology_tft/ORCHESTRATOR.md)

## 1. What This File Does

This is the control document for the implementation. It answers:

1. Which task IDs exist?
2. What depends on what?
3. Which phase gate is active?
4. Which tasks may run in parallel?
5. When is a task allowed to become `DONE`?
6. Which evidence must be written to the progress tracker?

This file is not a background daemon. It is an explicit operating protocol for a human or coding agent. The tracker records live state; this orchestrator determines whether that state is legal.

Current audited implementation state through 2026-07-29:

- `TFT-H01` is complete.
- `TFT-C07` is complete.
- `TFT-C09` is complete.
- `TFT-O01` is complete.
- `TFT-C08` is complete.
- `TFT-C03` is complete.
- `TFT-C01` is complete.
- `TFT-C02` is complete.
- `TFT-C04` is complete.
- `TFT-C05` is complete.
- `TFT-C06` is complete.
- `TFT-T01` is complete.
- `G1` is now passed because the exact `TFT-T01` release command was recorded green.
- `TFT-P01` is complete.
- `TFT-A03` is complete.
- `TFT-A02` is complete.
- `TFT-E01` is complete.
- `TFT-A01` is complete.
- `TFT-A04` is complete.
- `TFT-A05` is complete.
- `TFT-A07` is complete.
- `TFT-A09` is complete.
- `TFT-A08` is complete.
- `TFT-A10` is complete.
- `TFT-A06` is complete.
- The planetary/NIFTY branch is explicitly open at protocol/data-audit stage.
- Financial-astrology implementation is currently paused while the native TFT
  feature matrix runs and the theory/data choices are discussed.
- Generic output-physics work remains separate; it is not a prerequisite for treating deterministic ephemerides as known-future covariates.

## 1.1 If you are resuming work mid-stream

Use this exact branch logic:

| Situation | Required action |
|---|---|
| You think `TFT-C03` is still open | Re-check the tracker first. `TFT-C03` is already `DONE`; do not reopen it without a reproduced regression. |
| You need the next legal task | Read the canonical financial-astrology `CURRENT_STATUS.md`. At present only discussion/read-only matrix inspection is legal; after authorization use `FA-DATA-001` first. |
| You are verifying whether `TFT-T01` ever closed | Check evidence `EV-IMP-009`; it records the exact green release command. |
| You are verifying whether `TFT-P01` ever closed | Check evidence `EV-IMP-010`; it records the shared profile/digest implementation and regression pass. |
| You are verifying whether `TFT-A03` ever closed | Check evidence `EV-IMP-011`; it records the canonical embedding/static-encoder implementation and regression pass. |
| You are verifying whether `TFT-A02` ever closed | Check evidence `EV-IMP-012`; it records the dead-branch pruning and liveness/state-dict regression pass. |
| You are verifying whether `TFT-E01` ever closed | Check evidence `EV-IMP-013`; it records the reproducible canonical reference benchmark and saved artifacts. |
| You are verifying whether `TFT-A01` ever closed | Check evidence `EV-IMP-014`; it records the learned FFT selector repair, runtime-length policy, interpretation summaries, and regression pass. |
| You are verifying whether `TFT-A04` ever closed | Check evidence `EV-IMP-015`; it records the sparse-support-preserving temporal graph repair and regression pass. |
| You are verifying whether `TFT-A05` ever closed | Check evidence `EV-IMP-016`; it records the MoE capacity/fallback/load-balancing repair and regression pass. |
| You are verifying whether `TFT-A07` ever closed | Check evidence `EV-IMP-017`; it records lag masking, shifted physical positions, and temporal-compression coordinate preservation. |
| You are verifying whether `TFT-A09` ever closed | Check evidence `EV-IMP-018`; it records the native TFT config/digest cleanup and ignored-knob handling. |
| You are verifying whether `TFT-A08` ever closed | Check evidence `EV-IMP-019`; it records debug-gated finite checks and the removal of import-time environment mutation. |
| You are verifying whether `TFT-A10` ever closed | Check evidence `EV-IMP-020`; it records attention-probability dropout across exact/SDPA paths. |
| You are verifying whether `TFT-A06` ever closed | Check evidence `EV-IMP-021`; it records schema-aware interpretation exports and caveat flags. |
| You reproduce a new native regression | Open a new task or reopen the specific failing contract, not `TFT-T01` by default. |
| You want to test planetary covariates | Follow `AST-*`; first prove the loader/time alignment and nested baselines. Do not route directly into generic `PHY-L01`. |
| You want generic physics integration | Keep it separate unless a domain supplies a quantitative governing law and physical units for the target relationship. |

## 2. Source-of-Truth Hierarchy

When documents disagree, use this order:

1. This orchestrator owns stable IDs, dependencies, task weights, gates, and status-transition rules.
2. [`implementation_plan.md`](implementation_plan.md) owns exact implementation steps, APIs, tests, and task-specific acceptance criteria.
3. [`TFT_Implementation_Progress.md`](TFT_Implementation_Progress.md) owns current status, owner, blockers, evidence, and percentages.
4. [`TFT_Deep_Analysis_Report.md`](TFT_Deep_Analysis_Report.md) owns the audited diagnosis and rationale.
5. [`Vedic_Astrology_TFT_Implementation_Plan.md`](Vedic_Astrology_TFT_Implementation_Plan.md) owns exact planetary-market protocol, task recipes, controls, and claim gates.

Never rename or reuse an ID. If a task must split, keep the parent and create suffixes such as `TFT-C03-1` and `TFT-C03-2`.

## 3. Frozen Scope and Design Decisions

These rules require an explicit recorded decision before they may change:

- Do not edit `models/TFT_Nixtla.py`.
- The first safety release supports native TFT long-term forecasting only.
- `short_term_forecast` must fail early until a separate markless contract is designed.
- The experiment, target loss, inverse metrics, and physics engine must share one `tft_target_pos` mapping.
- Model auxiliary outputs must be tensors in a dictionary or `NamedTuple`; production code must not read mutable `last_*` attributes.
- Output physics constraints live in the experiment/loss layer, not inside the model.
- Physics operands are qualified by namespace and feature name.
- Physical equations run in physical units through differentiable Torch transforms.
- Configuration is declarative JSON. Never use `eval`, dynamic imports, or arbitrary callbacks.
- Missing or zero physics weight bypasses the training physics path.
- L1 is invalid for softmax VSN weights.
- A synthetic known-law result is required before ETTh1 theory claims.
- Hard projection or simulator coupling stays deferred until a defensible domain law exists.
- Planetary ephemerides are named known-future inputs. Orbital consistency does not imply a governing equation for NIFTY returns.
- The primary planetary test is incremental predictive value over market history plus flexible calendar controls.
- Raw next-day return precedes raw OHLC-level forecasting; structurally valid OHLC is a later secondary head.
- Real and placebo planet arms must be capacity-matched and paired on the same future dates.
- Independently row-shuffled ephemerides are not an adequate smooth-process null.
- VSN/attention/graph weights are diagnostics, not causal evidence.
- The final temporal holdout is frozen and inspected once per confirmatory protocol version.

## 4. Status State Machine

```text
NOT_STARTED --dependencies done--> READY
READY -------claimed-------------> IN_PROGRESS
IN_PROGRESS --implementation-----> IN_REVIEW
IN_REVIEW ---acceptance passes---> DONE

IN_PROGRESS/IN_REVIEW --real impasse--> BLOCKED
BLOCKED ------------impasse removed--> READY or IN_PROGRESS
any non-DONE state --scope decision---> DEFERRED
```

Rules:

- Only one owner may hold an `IN_PROGRESS` task.
- A task cannot be `READY` while a declared dependency is incomplete.
- A failing red test is expected and leaves a task `IN_PROGRESS`.
- `DONE` requires evidence, not confidence.
- A reviewer must not accept their own unverified claim when an independent reviewer is available.
- A gate passes only when every required task is `DONE`.

## 5. Stable Task Registry

### 5.1 Governance

| ID | Work item |
|---|---|
| `GOV-001` | Native TFT deep audit |
| `GOV-002` | Corrected physics/theory-guided design |
| `BASE-001` | Focused baseline verification |
| `GOV-003` | Master guide, progress tracker, and orchestrator |

Governance tasks do not count toward implementation progress.

### 5.2 Native safety tasks

| ID | Weight | Dependencies | Work item |
|---|---:|---|---|
| `TFT-C01` | 3 | `G0` | Preserve static covariates through normalization. |
| `TFT-C02` | 1 | `G0` | Pass LSTM state as `(c_h, c_c)`. |
| `TFT-O01` | 3 | `G0` | Add tensor-only forecast/quantile/MoE output contract. |
| `TFT-C09` | 2 | `G0` | Honor `tft_target_pos` in train, validation, test, inverse metrics, and plots. |
| `TFT-C03` | 3 | `TFT-O01`, `TFT-C09` | Repair quantile training, ordering, point selection, and evaluation. |
| `TFT-C04` | 2 | `G0` | Repair FFT top-amplitude indexing and batch invariance. |
| `TFT-C05` | 2 | `G0` | Repair order-2/order-3 interaction gating. |
| `TFT-C06` | 1 | `TFT-C01` | Preserve static interpretation payload. |
| `TFT-C07` | 1 | `G0` | Reject unsupported short-term use early. |
| `TFT-C08` | 2 | `G0` | Validate feature roles, ETT-S schemas, mark lengths, known channels, and frequency aliases. |
| `TFT-H01` | 1 | `G0` | Repair test discovery and deep benchmark controls. |
| `TFT-T01` | 3 | All native safety tasks | Run the full native semantic release gate. |

Native safety weight: `24`.

### 5.3 Canonicalization and extension tasks

| ID | Weight | Dependencies | Work item |
|---|---:|---|---|
| `TFT-P01` | 2 | `G1` | Add canonical, extended-safe, and experimental-full profiles. |
| `TFT-A02` | 2 | `TFT-P01` | Remove structurally dead profile parameters. |
| `TFT-A03` | 5 | `TFT-P01` | Add typed pointwise embeddings and canonical static VSN/context design. |
| `TFT-E01` | 3 | `TFT-A02`, `TFT-A03` | Benchmark canonical profile and parameter budget. |
| `TFT-A01` | 3 | `TFT-C04` | Implement true frequency-selective learned FFT mode. |
| `TFT-A04` | 5 | `G1` | Preserve graph sparsity and replace dense/C⁴ temporal evolution. |
| `TFT-A05` | 5 | `G1` | Fix MoE capacity/balancing and sparse-dispatch claim. |
| `TFT-A06` | 2 | `TFT-P01` | Formalize interpretation contracts and caveats. |
| `TFT-A07` | 3 | `G1` | Add lag masks and physical-position semantics. |
| `TFT-A08` | 2 | `G1` | Debug-gate finite checks and remove import-time hardware mutation. |
| `TFT-A09` | 2 | `TFT-P01` | Consolidate configuration and reject ignored knobs. |
| `TFT-A10` | 1 | `G1` | Implement attention-probability dropout. |

Canonicalization/extension weight: `35`.

### 5.4 Physics and theory-guided tasks

| ID | Weight | Dependencies | Work item |
|---|---:|---|---|
| `PHY-C01` | 3 | `G0` | Frozen dataclasses, safe JSON parser, and Pint/unit validation. |
| `PHY-D01` | 3 | `TFT-C08`, `TFT-C09`, `PHY-C01` | Dataset namespace names, cadence, scaler, and known-future metadata. |
| `PHY-S01` | 3 | `PHY-C01`, `PHY-D01` | Qualified-name resolver and differentiable namespace transforms. |
| `PHY-L01` | 5 | `PHY-S01` | Equality, inequality, polynomial, value, and rate constraint engine. |
| `PHY-I02` | 2 | `PHY-S01` | CLI validation, run digest, and resolved manifest lifecycle. |
| `PHY-I01` | 5 | `G1`, `PHY-L01`, `PHY-I02` | Production loss composition and zero-weight bypass. |
| `PHY-M01` | 2 | `PHY-I01` | Per-term metrics and checkpoint-selection policies. |
| `PHY-T01` | 3 | `PHY-L01` | Complete physics unit-test matrix. |
| `PHY-T02` | 3 | `PHY-I01`, `PHY-M01` | Production, AMP, quantile, target-map, standalone, and parity integration tests. |
| `PHY-V01` | 3 | `G4` | Synthetic known-law end-to-end proof. |
| `PHY-V02` | 2 | `PHY-V01` | Controlled ETTh1 theory-guided ablation. |
| `THY-A01` | 3 | `G1`, `TFT-O01` | Selective differentiable VSN auxiliary output. |
| `THY-V01` | 3 | `THY-A01` | Entropy/KL/forbidden-mass and valid sigmoid priors. |
| `THY-M01` | 5 | `G5` | Physical-unit counterfactual monotonicity. |
| `ADV-P01` | 5 | `G5`, validated law | Hard projection, simulator, or physics-residual architectures. |

Active physics/theory weight, excluding deferred `ADV-P01`: `45`.

### 5.5 Planetary-covariate research tasks

| ID | Weight | Dependencies | Work item |
|---|---:|---|---|
| `AST-H01` | 2 | `G2` | Freeze the falsifiable hypothesis, target, cutoff, folds, metrics, seeds, and locked holdout. |
| `AST-D01` | 3 | supplied sample/generator | Audit provenance, coordinates, timestamps, joins, units, redundancy, missingness, and Hilbert boundaries. |
| `AST-C00` | 1 | `G2` | Repair or explicitly reject the reproduced per-feature VSN gating crash. |
| `AST-K01` | 5 | `AST-D01` | Build a production loader with separate market, target, calendar, and planetary namespaces. |
| `AST-B01` | 4 | `AST-H01`, `AST-K01` | Add zero/Ridge/TFT market and market+calendar baselines plus paired metrics. |
| `AST-F01` | 3 | `AST-D01`, `AST-K01` | Build frozen raw-circular, relative-harmonic, and declared-aspect feature families. |
| `AST-N01` | 4 | `AST-D01`, `AST-F01` | Build coherent shifted, spectrum-preserving, and smooth pseudo-planet nulls. |
| `AST-E00` | 2 | `AST-H01`, `AST-B01`, `AST-F01`, `AST-N01` | Test raw incremental predictive value and issue a stop/proceed decision. |
| `AST-M01` | 5 | `AST-E00` proceed decision | Add grouped body tokens and low-rank relative-geometry messages. |
| `AST-L01` | 4 | `AST-M01` | Add calendar-time-aware fast/intermediate/slow response kernels. |
| `AST-O01` | 3 | stable point model | Add a structurally valid gap/body/upper/lower OHLC head. |
| `AST-Q01` | 1 | stable point model | Add quantile calibration ablations. |
| `AST-E01` | 5 | selected frozen architecture/nulls | Run the complete walk-forward/seed/null study and inspect the locked test once. |
| `AST-R01` | 2 | `AST-E01` | Classify the result without claim inflation. |

Planetary-covariate weight: `44`.

Legacy native plus generic physics/theory weight remains `104`. The planetary lane is reported separately rather than rewriting historical completion percentages.

## 6. Dependency Graph

```text
G0
├─ TFT-C01 ─> TFT-C06 ───────────┐
├─ TFT-C02                       │
├─ TFT-O01 ─┐                    │
├─ TFT-C09 ─┴> TFT-C03           │
├─ TFT-C04                       ├─> TFT-T01 ─> G1
├─ TFT-C05                       │
├─ TFT-C07                       │
├─ TFT-C08                       │
└─ TFT-H01 ──────────────────────┘

G0 ─> PHY-C01
TFT-C08 + TFT-C09 + PHY-C01 ─> PHY-D01
PHY-C01 + PHY-D01 ─> PHY-S01 ─> PHY-L01 ─> PHY-T01
PHY-S01 ─> PHY-I02
G1 + PHY-L01 + PHY-I02 ─> PHY-I01
PHY-I01 ─> PHY-M01 ─> PHY-T02 ─> G4
G4 ─> PHY-V01 ─> G5 ─> PHY-V02 / THY-M01 / ADV-P01

G1 ─> TFT-P01 ─> TFT-A02 + TFT-A03 ─> TFT-E01 ─> G2
TFT-C04 ─> TFT-A01
G1 ─> TFT-A04 / TFT-A05 / TFT-A07 / TFT-A08
TFT-P01 ─> TFT-A06 / TFT-A09
G1 ─> TFT-A10

G1 + TFT-O01 ─> THY-A01 ─> THY-V01 ─> G6

G2
├─> AST-H01 ──────────────┐
├─> AST-C00               │
└─> AST-D01 ─> AST-K01 ─> AST-B01
             └> AST-F01 ─> AST-N01

AST-H01 + AST-B01 + AST-F01 + AST-N01
                    └─> AST-E00
                           ├─ stop and report a null/invalid representation, or
                           └─ proceed -> AST-M01 -> AST-L01 -> AST-E01 -> AST-R01

stable point model -> AST-O01 / AST-Q01
```

## 6.1 Practical execution order

The dependency graph is authoritative. This section translates it into simple “what do I do next?” instructions:

1. `TFT-T01` is already closed and `G1` is already passed.
2. `TFT-P01` is already closed.
3. `TFT-A03` is already closed.
4. `TFT-A02` is already closed.
5. `TFT-E01` is already closed.
6. `TFT-A01` is already closed.
7. `TFT-A04` is already closed.
8. `TFT-A05` is already closed.
9. `TFT-A07`, `TFT-A09`, `TFT-A08`, `TFT-A10`, and `TFT-A06` are also already closed.
10. Do not reopen TFT unless a fresh regression reproduces against the recorded evidence.
11. For the planetary project, claim `AST-H01` for protocol work, `AST-D01` when the sample/generator is present, or `AST-C00` for the reproduced optional-gating regression.
12. Do not implement `AST-K01` until the dataset/provenance contract is frozen.
13. Do not implement `AST-M01`/`AST-L01` until `AST-E00` records a justified proceed decision.
14. Keep generic output-physics work separate; it is not a dependency of `AST-*`.

## 7. Phase Gates

### G0 — Planning baseline

Required:

- audit exists;
- master implementation plan exists;
- tracker and orchestrator exist;
- audited SHA is recorded;
- baseline commands/results are recorded.

State at creation: `PASSED`.

### G1 — Native safety

Required tasks:

- `TFT-C01`, `TFT-C02`, `TFT-O01`, `TFT-C09`, `TFT-C03`;
- `TFT-C04`, `TFT-C05`, `TFT-C06`, `TFT-C07`, `TFT-C08`;
- `TFT-H01`, `TFT-T01`.

Required behavior:

- semantic red/green tests exist for every defect;
- focused tests pass;
- the existing native suite passes;
- affected unsupported combinations fail before training;
- no production loss depends on mutable replica attributes.

Gate-closure note:

- `G1` is already closed by a recorded green run of the exact `TFT-T01` command;
- a smoke test or subset pass would not have been sufficient.

### G2 — Canonical baseline

Required:

- `TFT-P01`, `TFT-A02`, `TFT-A03`, `TFT-E01`;
- canonical profile has no unexpected dead parameters;
- reference results include accuracy, calibration, memory, latency, and parameter count.

### G3 — Physics foundation

Required:

- `PHY-C01`, `PHY-D01`, `PHY-S01`, `PHY-L01`, `PHY-T01`;
- no arbitrary expression evaluation;
- no Pint calls inside the batch loop;
- every constraint has numerical and gradient tests;
- irregular cadence and unavailable sources fail early.

### G4 — Production physics integration

Required:

- `PHY-I02`, `PHY-I01`, `PHY-M01`, `PHY-T02`;
- exact deterministic-CPU zero-weight parity;
- AMP and non-AMP paths share semantics;
- target mappings are correct;
- resolved manifests support standalone test mode;
- held-out physical compliance is persisted separately from accuracy.

### G5 — Physical validity

Required:

- `PHY-V01`;
- known-law held-out violation improves by a predeclared amount;
- forecasting stays inside a predeclared non-inferiority budget;
- baseline and physics runs share initialization and data order.

### G6 — VSN theory priors

Required:

- `THY-A01`, `THY-V01`;
- requested auxiliaries remain differentiable;
- no full-attention materialization is forced;
- softmax-L1 configuration is rejected.

### G7 — Counterfactual monotonicity

Required:

- `THY-M01`;
- deterministic/shared stochastic treatment;
- perturbation survives normalization;
- module modes restore even after exceptions;
- gradients, AMP, boundaries, and cost are tested.

### G8 — Experimental hardening

Required:

- `TFT-A01`, `TFT-A04`, `TFT-A05`, `TFT-A06`;
- `TFT-A07`, `TFT-A08`, `TFT-A09`, `TFT-A10`;
- each extension is benchmarked separately against the canonical profile.

### AST-G0 — Hypothesis and data contract

Required:

- `AST-H01`, `AST-D01`, and `AST-C00`;
- forecast cutoff, target, primary metric, folds, seeds, and locked holdout are versioned;
- ephemeris frame, ayanamsha, timestamp, units, body list, node policy, and generator are reproducible;
- Hilbert transforms have an explicit boundary/availability policy;
- the per-feature VSN flag is either repaired or rejected clearly.

### AST-G1 — Production loader and baselines

Required:

- `AST-K01`, `AST-B01`;
- `x_enc` contains causal market history only;
- `x_mark_*` contains named calendar plus real past/future planetary values;
- no future market-derived value enters a known namespace;
- scalers are training-fold local;
- test loss is not consulted during epoch-by-epoch selection;
- zero/Ridge/TFT market and market+calendar baselines save paired per-date predictions.

### AST-G2 — Circular features and nulls

Required:

- `AST-F01`, `AST-N01`;
- circular fields, relative harmonics, Rahu/Ketu policy, and any aspect/orb basis are frozen;
- every real ephemeris block has coherent, autocorrelation/spectrum-aware null counterparts;
- generated names/formulas/null seeds are persisted.

### AST-G3 — Raw test and multiscale architecture

Required:

- `AST-E00` records a stop/proceed decision using development folds only;
- if proceeding, `AST-M01` and `AST-L01` are complete;
- real and null branches are capacity matched;
- planet-pair interactions occur on named planet tokens before TFT VSN collapse;
- duration paths use elapsed-calendar-time-aware features;
- large generic TFT extensions remain isolated ablations.

### AST-G4 — Locked evaluation and claim gate

Required:

- `AST-E01`, `AST-R01`;
- all frozen folds/seeds/nulls are complete;
- paired block-bootstrap/HAC-aware uncertainty and multiplicity controls are reported;
- the locked test is inspected once;
- the result is classified as stable association, null, unstable/exploratory, or invalid;
- no interpretation weight is presented as causal evidence.

## 8. Task Selection Algorithm

At the start of a session:

1. Read the current gate in the tracker.
2. List tasks whose status is `READY`.
3. Remove tasks whose primary files are locked by active work.
4. Prefer tasks required by the active gate.
5. Break ties by:
   1. most downstream work unlocked;
   2. highest severity;
   3. smallest task weight;
   4. lexical task ID.
6. Claim exactly one task per owner.
7. Update the tracker before editing code.

Do not start a `NOT_STARTED` task merely because it looks easy.

## 9. File-Lock and Parallelism Rules

Only one active writer may own a primary file.

### Model lock

These tasks touch `models/TemporalFusionTransformer.py` and must normally be serialized:

- `TFT-C01`, `TFT-C02`, `TFT-O01`, `TFT-C06`, `TFT-C07`;
- `TFT-C03`, `TFT-A03`, `TFT-A06`, `TFT-A08`;
- `THY-A01`, `AST-C00`, `AST-M01`.

### Experiment lock

Serialize:

- `TFT-C09`, `TFT-O01`, `TFT-C03`;
- `PHY-I01`, `PHY-M01`, `PHY-T02`;
- `AST-B01`, `AST-E00`, `AST-E01`.

### Temporal-fusion layer lock

Serialize:

- `TFT-C04`, `TFT-C05`, `TFT-A01`, `TFT-A05`, `TFT-A07`, `AST-L01`.

### Data/CLI lock

Serialize overlapping edits among:

- `TFT-C08`, `PHY-D01`, `PHY-I02`, `TFT-A09`, `AST-D01`, `AST-K01`.

### Safe initial parallel wave

With separate owners:

- model/API lane: `TFT-O01`;
- experiment lane: `TFT-C09`;
- layer lane: `TFT-C04`, then `TFT-C05`;
- new-file physics lane: `PHY-C01`;
- harness/reviewer lane: `TFT-H01`.

For the planetary lane, `AST-H01` protocol drafting and read-only `AST-D01` inspection may overlap, but do not implement `AST-K01` until the data manifest is frozen.

Do not parallelize merely because multiple agents are available.

## 10. Required Task Workflow

### Before editing

1. Read the task card in [`implementation_plan.md`](implementation_plan.md), or in [`Vedic_Astrology_TFT_Implementation_Plan.md`](Vedic_Astrology_TFT_Implementation_Plan.md) for `AST-*`.
2. Confirm every dependency is `DONE`.
3. Set the task to `IN_PROGRESS`.
4. Record owner, start date, primary-file lock, and next action.
5. Record the current repository state and unrelated dirty files.

### Test-first step

1. Add the smallest semantic regression that fails for the audited reason.
2. Run only that test.
3. Record the red command and failure in a draft evidence entry.
4. Confirm the failure is not caused by a typo in the test.

Never merge an intentionally failing test. The red result is evidence during implementation, not the final state.

### Implementation step

1. Change only files declared by the task, unless the tracker records a scope decision.
2. Keep public tensor shapes and compatibility behavior described in the task card.
3. Add validation before silent fallback.
4. Do not repair unrelated findings opportunistically.
5. Add comments for non-obvious tensor axes and unit conversions.

### Verification step

1. Run the focused red/green test.
2. Run the task's neighboring test file.
3. Run the gate regression command.
4. Run `git diff --check`.
5. Inspect changed files and tensor-shape assumptions.
6. Record exact commands, results, and warnings.

Special rule for `TFT-T01`: the “gate regression command” means the exact release command from the `TFT-T01` task card in [`implementation_plan.md`](implementation_plan.md), not an approximate subset.

### Review and completion

1. Set status to `IN_REVIEW`.
2. Reviewer checks the acceptance list and reruns evidence.
3. If any criterion fails, return to `IN_PROGRESS`.
4. Set `DONE` only after every criterion passes.
5. Release the file lock.
6. Promote newly unblocked children to `READY`.
7. Recalculate progress and gate state.

## 11. Evidence Requirements

Every `DONE` task needs one evidence entry containing:

- evidence ID;
- task ID;
- base/result revision or patch reference;
- owner and reviewer;
- changed files;
- red test command and expected failure;
- focused green command/result;
- regression command/result;
- acceptance checklist;
- performance/memory measurements when required;
- warnings or known limitations;
- children unlocked.

Statements such as “looks good,” “tests passed,” or “implemented” are insufficient without exact commands and results.

## 12. Failure and Rollback Rules

| Situation | Required action |
|---|---|
| New test fails for the wrong reason | Fix the test before implementation. |
| Existing unrelated test fails before the patch | Record it as baseline evidence; do not claim it was introduced. |
| Focused test passes but gate regression fails | Keep task `IN_PROGRESS`; identify the contract regression. |
| Target mapping is ambiguous | Fail before training; never fall back to the last channel. |
| Physics unit/source cannot be resolved | Fail configuration; never guess. |
| Zero-weight mode calls physics | Reject the patch. |
| Quantile/auxiliary data is read from mutable `last_*` state | Reject the patch. |
| Counterfactual pass changes model mode permanently | Reject the patch. |
| Hard constraints are infeasible | Stop `ADV-P01`; require a domain decision. |
| Planetary fields appear only in `x_enc` | Reject the run; future-known semantics are not being tested. |
| Calendar control disappears in a planet arm | Reject the comparison; custom-known composition is incomplete. |
| Ephemeris timestamp/frame/ayanamsha is unresolved | Stop `AST-D01`; never guess. |
| Future market-derived data appears in `x_mark_*` | Treat as leakage and invalidate the run. |
| Real planet arm has more capacity than its null comparator | Reject the causal/incremental interpretation; capacity-match and rerun. |
| Locked-test performance prompts a configuration change | Version the experiment as exploratory and require a new holdout. |
| Task grows materially beyond its card | Record a decision and split the task. |

Do not use destructive Git recovery commands. Preserve unrelated user changes.

## 13. Standard Verification Commands

### Native focused baseline

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_comprehensive.py \
  tests/test_tft_interpretation_and_exp.py
```

### Native diagnostics and scripts

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_bugfix_diagnostics.py \
  tests/test_tft_scripts_smoke.py \
  tests/test_revin_ablation.py
```

### Physics foundation

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_physics_losses.py
```

### Physics integration

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_physics_integration.py
```

### Planetary data and known-future contract

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_planetary_schema.py \
  tests/test_planetary_market_loader.py
```

These files are planned, not present yet. Once created, the gate must also inspect a saved batch manifest containing forecast-origin dates, target dates, ordered known names, and train-fitted scaler metadata.

### Diff hygiene

```bash
git diff --check
```

Add narrower task-specific commands from the master guide. Do not treat a smoke test as a substitute for a semantic test.

## 14. Session Handoff Template

Copy this into the tracker/evidence log:

```text
Task:
Owner:
Status:
Primary-file lock:
Base revision:
Files changed:
Red test:
Current behavior:
Green tests:
Regression tests:
Acceptance items remaining:
Known limitations:
Next exact action:
Children potentially unlocked:
```

## 15. Change-Control Rules

Record a decision before:

- changing a task dependency or weight;
- renaming a tensor axis or output key;
- changing default profile behavior;
- allowing physics on another model/task;
- adding a constraint type;
- allowing irregular cadence;
- changing checkpoint selection;
- calling an ETTh1 prior “physics”;
- enabling a deferred task.

The decision entry must include date, reason, alternatives, affected tasks, migration impact, and required new tests.

## 16. Orchestrator Self-Check

Before closing a planning or implementation session:

- [ ] Every task ID in the tracker exists here.
- [ ] Every implementation task here has a card in the master guide.
- [ ] No `READY` task has an incomplete dependency.
- [ ] No two active tasks own the same primary file.
- [ ] Every `DONE` task has evidence.
- [ ] Gate states match task states.
- [ ] Progress excludes governance and deferred work.
- [ ] Newly unlocked tasks were promoted.
- [ ] Links and Markdown fences are valid.
