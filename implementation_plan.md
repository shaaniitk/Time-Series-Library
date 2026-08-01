# Master Implementation Plan: Native TFT, Planetary-Covariate Research, and Physics-/Theory-Guided Learning

> Status: implementation-active plan. The current worktree contains the full native TFT safety/hardening set. The planetary-covariate research lane was opened for protocol and data-contract work on 2026-07-29; generic output-physics constraints remain unimplemented.
>
> Scope: the repository-native `models/TemporalFusionTransformer.py` only. `models/TFT_Nixtla.py` is out of scope. The planetary lane tests deterministic ephemerides as known-future covariates; it is not a claim that an orbital governing equation constrains NIFTY returns.
>
> Revision basis: repository commit `564cffbc712f`, audited 2026-07-28.
>
> Live status: [`TFT_Implementation_Progress.md`](TFT_Implementation_Progress.md)
>
> Task dependencies and execution rules: [`TFT_Implementation_Orchestrator.md`](TFT_Implementation_Orchestrator.md)
>
> Full diagnosis: [`TFT_Deep_Analysis_Report.md`](TFT_Deep_Analysis_Report.md)
>
> Planetary/NIFTY research plan: [`Vedic_Astrology_TFT_Implementation_Plan.md`](Vedic_Astrology_TFT_Implementation_Plan.md)
>
> Canonical cross-session financial-astrology project:
> [`projects/financial_astrology_tft/README.md`](projects/financial_astrology_tft/README.md)

## 0. How to Use This Plan

This document is deliberately explicit. A new contributor should follow it without inventing architecture or silently changing scope.

Current execution snapshot on 2026-07-31:

| Area | State | What that means |
|---|---|---|
| Native safety repairs | Complete | The audited correctness defects in static handling, recurrent state order, quantile flow, FFT indexing, higher-order interaction, schema validation, target mapping, interpretation payloads, and short-term rejection have code changes and recorded evidence in the worktree. |
| Native release gate | Closed | `TFT-T01` passed on Tuesday, July 28, 2026 with the exact required regression command, so `G1` is now passed. |
| Physics/theory plan | Designed, not implemented | Schema/parser, namespace metadata, differentiable transforms, constraint loss, and production integration remain future work. |
| Post-G1 TFT upgrades | Legacy implementation/matrix complete; semantic-v2 open | `TFT-P01`, `TFT-A01`–`TFT-A10`, and `TFT-E01` are recorded `DONE`; the 14-case matrix completed, and `TFT-SR00`–`TFT-SR09` now own the discovered semantic defects. |
| Planetary/NIFTY lane | Data remediation / theory discussion | Source audit found invalid rashi encodings, mixed session dates, and absent generator provenance; no loader, planetary encoder, market experiment, or claim has been implemented. |

If you are picking work up cold, use this exact branch logic:

1. Check [`TFT_Implementation_Progress.md`](TFT_Implementation_Progress.md).
2. Confirm `TFT-T01` is recorded `DONE` in the tracker with evidence `EV-IMP-009`.
3. Confirm `TFT-A06` through `TFT-A10` are all recorded `DONE` in the tracker.
4. Only reopen a TFT task if a fresh regression is reproduced.

For every task:

1. Find its status in the progress tracker.
2. Start it only when the orchestrator says every dependency is `DONE`.
3. Set the tracker row to `IN_PROGRESS` and claim its primary files.
4. Write the listed failing semantic test first.
5. Make only the listed implementation change.
6. Run the focused command and the required regression command.
7. Record exact evidence.
8. Set the task to `IN_REVIEW`.
9. Mark it `DONE` only after every acceptance item passes.

Do not:

- edit `TFT_Nixtla.py`;
- build physics losses on the current legacy `f_dim` target selection;
- read production quantile or MoE values from mutable `last_*` attributes;
- evaluate physical equations in dataset-standardized coordinates;
- add free-form expression evaluation;
- call ETTh1 smoothness a governing law;
- combine unrelated fixes in one task.

### 0.1 Zero-confusion operator workflow

Use this when the repo changes hands between developers or agents.

| Step | Action | Allowed outcome |
|---|---|---|
| 1 | Open the progress tracker and locate the highest-priority `READY` task. | One task selected. |
| 2 | Read the matching task card in this file completely. | You understand exact files, tests, and “done” criteria. |
| 3 | Add or confirm the failing semantic test first. | Task becomes `IN_PROGRESS`. |
| 4 | Patch only the files listed by the task unless a scope decision is recorded. | No accidental side-quest work. |
| 5 | Run the focused command, then the neighboring suite, then the gate/regression command. | Evidence is concrete. |
| 6 | Update the tracker, evidence log, and change log on the same day. | Another implementer can continue without guessing. |
| 7 | Promote only dependency-unblocked children to `READY`. | The orchestrator remains truthful. |

### 0.2 Master task map

Native safety:

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
```

Physics:

```text
G0 ─> PHY-C01
TFT-C08 + TFT-C09 + PHY-C01 ─> PHY-D01
PHY-C01 + PHY-D01 ─> PHY-S01 ─> PHY-L01 ─> PHY-T01
PHY-S01 ─> PHY-I02
G1 + PHY-L01 + PHY-I02 ─> PHY-I01
PHY-I01 ─> PHY-M01 ─> PHY-T02 ─> G4
G4 ─> PHY-V01 ─> G5 ─> PHY-V02 / THY-M01 / ADV-P01
G1 + TFT-O01 ─> THY-A01 ─> THY-V01
```

Planetary-covariate hypothesis lane:

```text
G2 ─> AST-H01
raw OHLC + generator package ─> AST-D01
TFT-SR06 ─> AST-C00
G2-SR + AST-D01 ─> AST-K01
G2-SR + AST-H01 + AST-K01 ─> AST-B01
AST-D01 + AST-K01 ─> AST-F01 ─> AST-N01

G2-SR + AST-H01 + AST-B01 + AST-F01 + AST-N01
                    └─> AST-E00 ─> AST-M01 ─> AST-L01 ─> AST-E01 ─> AST-R01
```

This lane does not depend on `PHY-L01`: planetary ephemerides are inference-available covariates, not a supplied governing law for the market target.

Canonicalization and extension hardening:

```text
G1 ─> TFT-P01 ─> TFT-A02 + TFT-A03 ─> TFT-E01
TFT-C04 ─> TFT-A01
G1 ─> TFT-A04 / TFT-A05 / TFT-A07 / TFT-A08 / TFT-A10
TFT-P01 ─> TFT-A06 / TFT-A09
```

### 0.3 Implementation phases

| Phase | Work | Exit gate |
|---|---|---|
| 0 | Audit, baseline, tracker, orchestration | `G0` |
| 1 | Native TFT correctness and honest contracts | `G1` |
| 2 | Canonical reference profile | `G2` |
| 3 | Physics schema, metadata, transforms, loss engine | `G3` |
| 4 | Production physics integration and metrics | `G4` |
| 5 | Synthetic physical validation | `G5` |
| 6 | VSN theory priors | `G6` |
| 7 | Counterfactual monotonicity | `G7` |
| 8 | Experimental extension hardening | `G8` |
| AST-0 | Hypothesis/data contract | `AST-G0` |
| AST-1 | Production known-future loader and nested baselines | `AST-G1` |
| AST-2 | Circular features and matched nulls | `AST-G2` |
| AST-3 | Grouped multiscale planetary architecture | `AST-G3` |
| AST-4 | Locked evaluation and evidence classification | `AST-G4` |

### 0.4 Recommended execution order after this update

This is the intended path unless a failing test forces a narrower fix-first branch.

| Order | Task | Why now | Stop condition |
|---:|---|---|---|
| 1 | `TFT-SR00` -> `TFT-SR01` -> `TFT-SR02` | The 14-case matrix is complete; freeze legacy-v1, then repair paired randomness and neutral integration. | Contract tests pass and the common baseline state is identical. |
| 2 | `TFT-SR03`–`TFT-SR08` -> `TFT-SR09` | Repair each named advanced operator and close one semantic release gate. | Focused semantic suite and deterministic micro-run pass. |
| 3 | `AST-H01`; resume `AST-D01` when inputs arrive | Freeze the falsifiable question now; rebuild/audit sources when raw OHLC and the generator package arrive. `AST-C00` accepts `TFT-SR06` evidence. | Dataset roles, provenance, cutoff, target, and holdout are frozen. |
| 4 | `AST-K01` -> `AST-B01` + `AST-F01` -> `AST-N01` | Make planetary values real production known-future inputs and create fair baselines/nulls. | One audited batch and every baseline/null test pass. |
| 5 | `AST-E00` | Test raw incremental value before building a bespoke large architecture. | Record stop/proceed decision from development folds only. |
| 6 | `AST-M01` -> `AST-L01` -> `AST-E01` -> `AST-R01` | Add explicit group/aspect and duration priors only if justified, then run locked evaluation. | Evidence is classified without claim inflation. |
| 7 | `PHY-C01` onward | Generic output-law work remains a separate reusable branch for domains with defensible equations. | Do not use it to invent a planet-to-price law. |

## 1. Decision Summary

The original proposal is directionally useful but is not implementation-ready. The revised design makes these decisions:

1. Put output-level physics losses in the experiment/loss layer, not inside the TFT model.
2. Integrate the feature into the production path in `exp/exp_long_term_forecasting.py`; keep `tests/test_tft_deep_ett.py` as an optional benchmark only.
3. Evaluate physical equations, rates, and bounds in physical units through a differentiable Torch inverse transform.
4. Resolve variables by names and source namespaces, not by unqualified integer indices.
5. Start with output-level equality, inequality, rate, and bound constraints.
6. Add VSN priors only after a lightweight differentiable auxiliary-output API exists.
7. Add feature-effect monotonicity later because it requires counterfactual forwards or higher-order input gradients.
8. Treat ETTh1 experiments as theory-guided ablations unless a domain expert supplies defensible equations, units, and topology.
9. Preserve exact baseline training behavior when the feature is disabled or its global weight is zero by not constructing or evaluating the physics loss path during train/validation. A separate post-hoc test evaluator may still measure baseline compliance.

The optimization objective will be:

```text
L_total =
    L_task
  + lambda_moe * L_moe
  + lambda_physics(epoch) * sum_k(weight_k * L_constraint_k)
```

Every `L_constraint_k` must be dimensionless, normally by dividing its residual by a declared physical tolerance or characteristic scale before applying the configured penalty.

## 2. Why the Original Plan Needs Revision

### 2.1 It targets the wrong integration layer

`tests/test_tft_deep_ett.py` is a standalone training/ablation script. The normal training and validation loops are in `exp/exp_long_term_forecasting.py`. A loss implemented only in the benchmark would not be available through `run.py`.

Output constraints do not require a change to `models/TemporalFusionTransformer.py`. Coupling a generic loss to the model would also make reuse, testing, and non-TFT comparisons harder.

### 2.2 Model outputs are not in physical units

ETT and custom datasets use a dataset-level `StandardScaler` before batches reach the model. TFT then applies and reverses a second, per-window normalization internally. Consequently, its returned forecast is back in dataset-standardized coordinates, not raw engineering units.

A maximum rate such as `5 °C/hour` or a balance equation with physical coefficients is invalid in standardized coordinates unless the coefficients and thresholds are transformed consistently. The implementation therefore needs a differentiable Torch inverse scaler based on the training dataset's `mean_` and `scale_`.

### 2.3 Future physical covariates are not generally available

The native TFT uses:

- `x_enc`: historically observed variables;
- `x_mark_enc` and the prediction part of `x_mark_dec`: known calendar/custom features;
- no values from `x_dec` in the embedding or forecast computation.

An equation such as `y = f(x_future)` is implementable only when every operand is either:

- another predicted output channel;
- a genuinely known-future feature carried through a defined interface; or
- a future label explicitly marked as training-only forcing.

The last option must never be presented as an inference-time guarantee.

Planetary ephemerides are an important concrete exception to the availability problem: their future values are deterministically computable. They belong in the genuine known-future category. However, the current `Dataset_Custom` loader still emits only calendar fields through `x_mark_*`, so the availability is not yet wired into production. This is handled by the separate `AST-K01` task rather than by inventing a `PHY-L01` planet-to-price residual.

### 2.4 The proposed VSN L1 term is ineffective

The normal VSN uses softmax selection weights. They are non-negative and sum to one, so:

```text
||w||_1 = sum_i w_i = 1
```

Its L1 penalty is constant and supplies no useful sparsity gradient. Suitable alternatives are:

- entropy minimization for a sparse simplex distribution;
- KL divergence or cross-entropy to a declared theory prior;
- mass penalties on forbidden variables;
- entmax/sparsemax as an architectural experiment;
- L1 only when the independent sigmoid-gating VSN is active.

### 2.5 Monotonicity needs a causal definition

Comparing ordinary time differences in a covariate and target measures correlation along a trajectory, not the target's partial response to that covariate.

Feature-effect monotonicity should use paired counterfactuals:

```text
violation = relu(
    -direction * (forecast(x + delta_i) - forecast(x)) / delta_i
)
```

The specification must also state the perturbation, valid input range, affected history or known-future timesteps, target channels, forecast horizons, and reduction.

Input-Jacobian penalties are an optional later alternative. They require `create_graph=True`, second-order gradients, more memory, and special AMP/checkpointing tests.

### 2.6 Production target selection bypasses `tft_target_pos`

The model uses `tft_target_pos` for its output de-normalization, but the long-term experiment still selects labels with the legacy `f_dim` convention and tiles reduced outputs before dataset inverse scaling. A non-last MS target or non-contiguous multi-output mapping can therefore train and score against the wrong channels.

Before physics integration, add one schema-aware target helper used by point loss, quantile loss, validation, test metrics, inverse scaling, and physics. Predictions must be inverse-transformed with the training scaler entries at the resolved target positions, never tiled to `enc_in`.

## 3. Domain and Data Contract

### 3.1 Dataset metadata

Expose immutable, namespace-specific metadata on supported forecasting datasets:

```python
input_feature_names: list[str]             # x_enc / batch_y order
known_future_feature_names: list[str]      # x_mark_* order
static_feature_names: list[str]            # subset of input features
timestamps: np.ndarray | None
sample_interval: float | None
sample_interval_unit: str | None
observed_scaler_mean: np.ndarray | None    # full input feature order
observed_scaler_scale: np.ndarray | None
known_future_transform: TransformMetadata
```

For every loader, capture `input_feature_names` from the actual `df_data.columns` after effective `M`/`MS`/`S` selection and any target-last reorder. Do not copy raw ETT CSV order when `features='S'` has reduced the tensor to one channel. Resolve TFT outputs through `tft_target_pos`, rather than assuming output channel `j` maps to input channel `j`.

The loader must expose the actual generated calendar-feature order for standard `x_mark` tensors. Custom known-future physical drivers require an explicit dataset interface and matching transform metadata; they must not be inferred from calendar marks.

CSV files do not establish engineering units. Load units and optional conversions from the physics specification or a separately versioned dataset sidecar. Do not guess them in `data_loader.py`.

Do not add metadata to every batch tuple; it is dataset-level state and should be resolved once when the experiment is built. Preserve the full `batch_y` tensor until physics namespaces have been gathered; the current target-only slicing happens too early for arbitrary `future_label` operands.

### 3.2 Constraint source namespaces

Every operand must use one of:

- `prediction`: a TFT output channel;
- `history`: an observed encoder channel;
- `known_future`: an inference-available future driver;
- `future_label`: training/validation supervision only, rejected unless explicitly allowed;
- `static`: a declared time-invariant field.

Names are resolved to indices once at startup. Ambiguous unqualified references, missing units, duplicates within a namespace, unavailable future sources, or target-mapping conflicts must fail before training. The same spelling in two different qualified namespaces is valid.

Temporal alignment must be explicit:

- `prediction`, `known_future`, and `future_label` are horizon-aligned;
- `history` requires `last`, `lag:k`, or a declared reduction;
- `static` requires an explicit horizon-broadcast rule;
- operands with incompatible time axes are rejected rather than implicitly broadcast.

### 3.3 Versioned declarative schema

Add JSON specifications under `configs/physics/`. Do not use `eval`, arbitrary expressions, or Python callbacks loaded from CLI strings.

Example:

```json
{
  "version": 1,
  "name": "tank_balance_v1",
  "space": "physical",
  "dt": {"value": 1.0, "unit": "hour"},
  "constraints": [
    {
      "name": "flow_balance",
      "type": "linear_equality",
      "residual_unit": "meter**3/hour",
      "terms": [
        {"source": "prediction", "feature": "inflow", "coefficient": 1.0},
        {"source": "prediction", "feature": "outflow", "coefficient": -1.0}
      ],
      "rhs": 0.0,
      "tolerance": 0.5,
      "penalty": "smooth_l1",
      "beta": 1.0,
      "reduction": "valid_mean",
      "quantile_policy": "point",
      "feasibility_metric": "violation_rate",
      "feasibility_max": 0.05,
      "weight": 1.0
    },
    {
      "name": "temperature_rate",
      "type": "absolute_rate_bound",
      "source": "prediction",
      "feature": "temperature",
      "history_boundary": {"source": "history", "time_selector": "last"},
      "residual_unit": "delta_degC/hour",
      "max_abs_rate": 5.0,
      "tolerance": 0.25,
      "penalty": "smooth_l1",
      "beta": 1.0,
      "reduction": "valid_mean",
      "quantile_policy": "point",
      "feasibility_metric": "violation_rate",
      "feasibility_max": 0.05,
      "weight": 0.25
    }
  ]
}
```

Polynomial equations should be explicit monomial lists. For example, `2*x*y - z^2 = 0` is represented by named factors and integer exponents, not a free-form expression.

Each constraint declares one residual unit. Its coefficients, right-hand side, bounds, and tolerance must already be expressed so that the computed residual has that unit, unless an explicit conversion is declared. The resolver validates dimensional compatibility where metadata permits and otherwise rejects missing unit declarations; it never silently infers coefficient units.

Dataset timestamp cadence is authoritative by default. A specification `dt` may override it only when unit conversion shows agreement within a configured tolerance. Stage 1 rejects irregular timestamps; per-step deltas can be added later.

Use Pint during configuration resolution, with its accepted unit grammar documented in `configs/physics/README.md` and the dependency pinned after compatibility testing. Pint is not used in the batch loop: the resolver converts every compatible quantity into precomputed affine factors and rejects undefined or dimensionally incompatible units.

`TransformMetadata` contains:

```python
feature_names: tuple[str, ...]
coordinate_space: Literal["dataset_standardized", "native"]
mean: np.ndarray | None
scale: np.ndarray | None
storage_units: tuple[str, ...]
physical_units: tuple[str, ...]
```

The resolver composes dataset scaling and unit conversion into Torch affine transforms. It must explicitly test offset units such as Celsius as well as multiplicative units.

### 3.4 Complete constraint semantics

Every constraint has these common fields:

- `name`, `type`, `residual_unit`, positive `tolerance`, non-negative `weight`;
- `penalty`: `absolute`, `squared`, or `smooth_l1` with explicit `beta`;
- `reduction`: `valid_mean` by default, or `valid_max`;
- `quantile_policy`: `point`, `median`, or `all_quantiles`;
- for `all_quantiles`, `quantile_reduction`: `mean` or `max`;
- `feasibility_metric`: `violation_rate`, `mean_normalized_violation`, or `max_normalized_violation`;
- non-negative `feasibility_max`, distinct from the loss-normalization tolerance.

Type-specific fields are:

- `linear_equality`: terms and `rhs`; every term has a coefficient and optional `coefficient_unit` (default dimensionless);
- `linear_inequality`: terms, `rhs`, and an explicit `operator` of `<=` or `>=`;
- `polynomial_equality`: `monomials`, where each monomial has a coefficient, `coefficient_unit`, and a non-empty list of `{source, feature, exponent, time_selector}` factors;
- `value_bound`: at least one of `lower` or `upper`;
- `absolute_rate_bound`: `feature`, positive `max_abs_rate`, and the history-boundary policy.

For example, a polynomial monomial is encoded as:

```json
{
  "coefficient": 2.0,
  "coefficient_unit": "dimensionless",
  "factors": [
    {"source": "prediction", "feature": "x", "exponent": 1, "time_selector": "horizon"},
    {"source": "known_future", "feature": "u", "exponent": 1, "time_selector": "horizon"}
  ]
}
```

Each operand declares its temporal selector and any static broadcast. Prediction/future operands default to horizon alignment; history and static never receive an implicit default broadcast.

Optional validity masks are dataset-owned tensors: observed/history masks use `[B,T,enc_in]`, future-label masks use `[B,H,enc_in]`, known-future masks use `[B,H,K]`, prediction masks use `[B,H,C_out]`, and static masks use `[B,S]`. Missing masks mean “all valid.” A constraint combines all operand masks before reduction and reports both valid count and violation count.

For equality, normalized violation is `abs(residual) / tolerance`. For an inequality, it is the infeasible-side residual (`relu(lhs-rhs)` for `<=`, reversed for `>=`) divided by tolerance. `violation_rate` is the valid fraction whose normalized violation exceeds `1`.

## 4. Proposed APIs

### 4.1 Configuration loader

Create `utils/physics_config.py`:

```python
@dataclass(frozen=True)
class PhysicsSpec:
    version: int
    name: str
    space: Literal["physical"]
    dt: TimeStep | None
    constraints: tuple[ConstraintSpec, ...]


def load_physics_spec(path: str | Path) -> PhysicsSpec:
    ...


def resolve_physics_spec(
    spec: PhysicsSpec,
    dataset_metadata: DatasetPhysicsMetadata,
    target_pos: Sequence[int],
) -> ResolvedPhysicsSpec:
    ...
```

Validation belongs here so the inner training loop receives only resolved indices and tensors.

### 4.2 Differentiable scaler

Create a small `nn.Module` or helper in `utils/physics_losses.py` with transforms for every enabled namespace:

```python
class TorchNamespaceTransforms(nn.Module):
    def __init__(self, observed, known_future, target_pos, static_pos):
        super().__init__()
        ...

    def inverse_prediction(self, x_standardized):
        ...  # select full observed scaler by target_pos

    def inverse_history(self, x_standardized):
        ...

    def inverse_future_labels(self, x_standardized):
        ...

    def to_physical_known_future(self, x):
        ...

    def to_physical_static(self, x):
        ...  # native static fields select static_pos from observed transform
```

The observed transform stores the full feature scaler, then selects `target_pos` only for predictions and `static_pos` for native static fields. Known-future inputs use their own declared affine transform or an explicit identity transform. Unscaled datasets also use an explicit identity transform. Categorical identifiers cannot participate in arithmetic constraints unless a physical numeric mapping is declared. An external-static namespace is explicitly deferred beyond Stage 1.

Every operation must retain gradients. Do not call the datasets' NumPy/sklearn `inverse_transform` from the loss.

### 4.3 Loss result and module

Create `utils/physics_losses.py`:

```python
@dataclass
class PhysicsLossResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    violation_rates: dict[str, torch.Tensor]
    valid_counts: dict[str, torch.Tensor]
    violation_counts: dict[str, torch.Tensor]


class PhysicsInformedLoss(nn.Module):
    def __init__(self, resolved_spec, transforms):
        ...

    def forward(
        self,
        forecast,               # [B, H, C_out], forecast only
        history,                # [B, T, enc_in]
        known_future=None,
        future_labels=None,     # [B, H, enc_in]
        static=None,            # [B, S]
        valid_masks=None,
    ) -> PhysicsLossResult:
        ...
```

Channels are selected from the resolved namespaces inside the loss. A quantile adapter must first reduce or reshape probabilistic output into a supported forecast form.

The name is retained for discoverability, but documentation should describe the initial feature as physics-/theory-guided regularization rather than a classical PINN.

### 4.4 Supported Stage-1 constraints

Implement only:

1. `linear_equality`
2. `linear_inequality`
3. declarative `polynomial_equality`
4. `absolute_rate_bound`
5. `value_bound`

Required behavior:

- slice the final `pred_len` forecast before applying constraints;
- include the last observed target to first forecast transition in rate loss;
- divide rates by the declared `dt`;
- support per-target and per-horizon masks;
- expose both differentiable loss and detached compliance metrics;
- return `forecast.sum() * 0` when all entries are masked so the zero remains connected to autograd.

Do not label generic squared temporal smoothness as physics unless a domain-specific derivative law justifies it.

## 5. Training Integration

### 5.1 Production experiment

Modify `exp/exp_long_term_forecasting.py` to:

1. reject non-native-TFT and non-long-term tasks whenever a physics config is supplied;
2. replace all `f_dim`/tiling target selection with one `tft_target_pos`-aware helper for point/quantile labels, inverse transforms, metrics, and physics;
3. build the physics module after a dataset carrying training-fitted scaler metadata is available;
4. in standalone `--is_training 0` mode, load and verify the saved resolved manifest or lazily resolve from equivalent training-fitted metadata—never refit transforms on the test range;
5. skip physics-loss construction and computation during train/validation when the config is absent or the global weight is zero; allow a separate no-gradient held-out compliance evaluator for a configured zero-weight baseline;
6. preserve full `batch_y` until namespace tensors and masks have been gathered;
7. centralize primary, MoE, and physics loss composition in one helper used by AMP and non-AMP paths;
8. slice the model output to `[B, pred_len, C_out]` before calling the physics loss;
9. log primary loss, the legacy validation objective, total loss, every constraint term, valid/violation counts, and every violation rate separately;
10. compute validation compliance without accidentally requiring gradients for output-only constraints;
11. compute and persist held-out raw-unit constraint metrics in `test()` before NumPy conversion.

Default early stopping must preserve the repository's current validation objective: primary loss plus the configured MoE auxiliary term. The learning-rate schedule remains the existing epoch-based schedule. Treat any change to those semantics as a separate migration. Optional checkpoint policies may select primary loss, a fixed-weight total loss, or a feasibility-aware primary loss.

### 5.2 CLI

Add near the existing TFT arguments in `run.py`:

```text
--tft_physics_config PATH
--tft_physics_weight FLOAT
--tft_physics_warmup_epochs INT
--tft_physics_schedule {constant,linear,cosine}
--tft_physics_selection_metric {legacy,primary,total,feasible_primary}
--tft_allow_future_label_constraints
--tft_physics_posthoc_eval
```

Defaults are empty config, weight `0.0`, warm-up `0`, `constant` schedule, `legacy` selection, future-label constraints disabled, and post-hoc evaluation disabled.

Enablement rules are:

- empty config: physics is entirely disabled; non-zero weight, non-legacy selection, or post-hoc evaluation is rejected;
- config + zero weight: train/validation take the untouched baseline path; `--tft_physics_posthoc_eval` may resolve the spec and evaluate only held-out forecasts;
- config + positive weight: training is active and held-out compliance is always persisted.

At setting construction time, hash the canonical raw config content plus relevant CLI/TFT flags. Resolution needs dataset metadata and therefore happens later; after resolution, write a manifest beside checkpoints/results containing the resolved spec, feature mappings, transform metadata, and its digest. The current setting omits TFT-specific configuration and can otherwise reuse a directory for materially different runs.

### 5.3 Quantile policy

Before enabling physics with `loss=Quantile`, fix the native TFT's quantile output contract described in `TFT_Deep_Analysis_Report.md`. Return point, quantile, and MoE values in a tensor-only dictionary or `NamedTuple` that `DataParallel` gathers; do not read mutable replica attributes. Define sample-weighted/global MoE-statistic reduction. Any monotone quantile parameterization must remain ordered after RevIN de-normalization, which requires a positive affine scale or a post-de-normalization ordering guarantee.

Then declare policy per constraint:

- `point`: apply to the trained point forecast;
- `median`: require and apply the quantile whose level is exactly `0.5`;
- `all_quantiles`: adapt the native `[B,H,Q,C]` output by applying a separable constraint over `Q`.

`all_quantiles` is allowed only for pointwise per-channel value/support bounds. Reject cross-channel equations and cross-horizon rate/dynamics constraints in this mode: separately estimated marginal quantiles do not form a coherent joint channel state or sample path. Do not apply physics to the current unrelated point head during quantile-only training.

### 5.4 Weight schedule

Let `e` be the one-based epoch index and `W` the configured warm-up epochs:

```text
constant: lambda(e) = lambda_max
linear:   lambda(e) = lambda_max * min(1, e / W)
cosine:   lambda(e) = lambda_max * 0.5 * (1 - cos(pi * min(1, e / W)))
```

For `W=0`, both ramp schedules return `lambda_max` from the first epoch. Reject negative weights or warm-up lengths.

When checkpointing by `total`, compute validation totals with fixed `lambda_max`, not the changing training weight. For `feasible_primary`, each constraint supplies a validation threshold; choose the lowest primary loss among feasible epochs. If none are feasible, choose the smallest maximum normalized violation, breaking ties by primary loss.

Start with fixed, dimensionless per-term weights. Adaptive Lagrange multipliers are a later feature because they add optimizer/checkpoint state and change reproducibility.

## 6. TFT-Internal Theory Priors

This is a separate stage because it needs differentiable internal diagnostics.

Add a lightweight `return_auxiliary=True` output mode to the native TFT that returns only requested tensors, such as history/future VSN weights or logits. It must not force full decoder-attention materialization or the exact-attention fallback.

Rules:

- softmax VSN: entropy, KL-to-prior, ranking, or forbidden-mass penalties;
- sigmoid VSN: L1 or target-cardinality penalties are allowed only on pre-dropout gates/logits;
- history namespace: observed variables followed by known variables;
- future namespace: known variables only;
- account for graph pre-mixing and residual bypass, which weaken a literal feature-importance interpretation;
- return differentiable values for training and detached copies for reporting.

Use a tensor-only dictionary or `NamedTuple`, not a plain dataclass or mutable `last_*` attributes, so `DataParallel` can gather the result.

## 7. Counterfactual Monotonicity Stage

Implement in the experiment/training wrapper, not in `PhysicsInformedLoss.forward`, because it requires model execution.

Each monotonicity rule must specify:

```text
source namespace
feature name
target name(s)
direction (+1 or -1)
physical perturbation delta
valid min/max
affected input timesteps
affected forecast horizons
reduction and tolerance
```

Use a deterministic, differentiable counterfactual context that disables dropout and stochastic depth for both base and perturbed forwards, or implement a tested shared-randomness mechanism. Merely concatenating the pairs does not guarantee shared masks. The context manager must snapshot every module's prior training flag and restore it in `finally`, including when either forward raises.

Perturb in physical coordinates, transform back to the model input space, and re-run the model. Avoid whole-window constant shifts that TFT's per-window normalization can cancel; each rule must identify the affected timesteps, and tests must confirm a non-zero post-normalization perturbation. Track the added forward-pass cost.

Defer Jacobian-based monotonicity until paired counterfactual tests pass. Jacobian mode needs tests for second-order gradients, checkpointing, AMP, and normalization semantics.

## 8. File-by-File Change Set

### New files

- `utils/physics_config.py`
- `utils/physics_losses.py`
- `configs/physics/README.md`
- `configs/physics/example_linear_balance.json`
- `tests/test_physics_losses.py`
- `tests/test_tft_physics_integration.py`

### Modified files

- `requirements.txt`
  - add the unit-registry dependency after compatibility pinning;
- `data_provider/data_loader.py`
  - expose ordered namespace names, timestamps/cadence, and scaler/transform statistics; do not infer units;
- `exp/exp_long_term_forecasting.py`
  - build, compute, log, and schedule physics terms; print resolved mappings after dataset resolution;
- `run.py`
  - add CLI/config validation and raw-spec/flag digest;
- `utils/print_args.py`
  - display raw physics controls and raw-config digest only;
- `models/TemporalFusionTransformer.py`
  - Stage 2 only: lightweight differentiable VSN auxiliary output;
- `tests/test_tft_deep_ett.py`
  - benchmark only: fix undefined `use_amp`, add epoch override, and report compliance.

No `layers/PhysicsLoss.py` file is recommended.

## 9. Verification Matrix

### 9.1 Unit tests: `tests/test_physics_losses.py`

- satisfying equality returns exact/near-zero loss;
- violated equality returns positive loss and finite, non-zero forecast gradients;
- inequality penalizes only the infeasible side;
- polynomial monomials match hand calculations;
- rate loss includes history-to-first-forecast boundary;
- rate uses physical `dt`;
- value/rate constraints select the correct target mapping;
- standardized and raw representations produce equivalent physical residuals;
- masks exclude invalid samples/horizons;
- valid/violation count dictionaries match masks and reductions;
- invalid names, units, sources, shapes, and schema versions fail early;
- inequality direction, value-bound sides, polynomial factors, and temporal selectors are validated;
- affine unit conversion covers offset and multiplicative units;
- quantile `mean` and `max` reductions match hand calculations;
- `all_quantiles` rejects cross-channel and cross-horizon constraints;
- mask shapes, valid counts, and all-masked behavior are correct;
- future labels are rejected unless explicitly enabled;
- CPU/device/dtype behavior is correct;
- autocast remains finite;
- disabled/no-constraint result is a same-device scalar zero;
- softmax-VSN L1 is rejected by config validation.

### 9.2 Integration tests: `tests/test_tft_physics_integration.py`

- on deterministic CPU, `physics_weight=0` matches baseline loss and parameter updates exactly; accelerator comparisons use tolerances;
- the normal long-term experiment trains with physics enabled;
- non-last MS and non-contiguous multi-output M target mappings work through loss, inverse scaling, and metrics;
- registered ETT `features='S'` resolves a one-channel schema without stale `0..6` indices;
- primary + MoE + physics composition is correct;
- quantile policy uses the trained output;
- exact-median and post-RevIN quantile-order rules are enforced;
- parallel output gathers quantile and MoE tensors with the defined reduction;
- validation logs primary and compliance separately;
- checkpoint selection follows the configured policy;
- held-out `test()` persists physical-unit compliance metrics;
- standalone test mode loads/verifies the resolved manifest without fitting on test data;
- a tiny synthetic system with a known conservation/dynamics law reduces violation versus a matched baseline;
- unsupported task/model/extension combinations fail before training.

Stage-2 VSN tests belong in a separate auxiliary-output test module and are not Stage-1 acceptance criteria.

Counterfactual-stage tests must additionally verify identical stochastic treatment, non-cancelled perturbations after normalization, and exception-safe restoration of every module's prior train/eval state.

### 9.3 Regression tests

Run:

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_physics_losses.py \
  tests/test_tft_physics_integration.py \
  tests/test_tft_comprehensive.py \
  tests/test_tft_interpretation_and_exp.py
```

Then run the wider TFT suite. `run_tests.py` should also be fixed to discover lowercase `tests`, or be retired in favor of pytest.

### 9.4 ETTh1 ablation

Only after unit and synthetic-law tests pass:

- fix `tests/test_tft_deep_ett.py`'s undefined `use_amp`;
- add `--epochs` and `--max-train-batches`;
- compare identical seeds, data splits, optimizer settings, and model initialization;
- run `lambda=0` and multiple non-zero weights;
- report forecast MSE/MAE and raw-unit violation metrics;
- do not call a smoothness/rate prior an ETTh1 governing law without supporting domain metadata.

The original “2-epoch” command was not a 2-epoch test: `baseline_safe` configures 150 epochs and the script currently has no epoch override.

## 10. Acceptance Gates

The feature is ready to merge only when:

1. disabled/zero-weight mode never evaluates physics and is exactly identical in deterministic CPU tests; accelerator comparisons use documented tolerances;
2. all residuals are evaluated in a documented coordinate system;
3. every operand is available in its declared source namespace;
4. synthetic satisfying cases have zero loss and violating cases have correct gradients;
5. the synthetic-law end-to-end test improves compliance while staying inside a predeclared forecast non-inferiority budget;
6. primary and constraint metrics are reported separately;
7. no arbitrary expression evaluation is used;
8. target mapping, quantile, DataParallel, static, FFT, and interaction defects are either fixed or the corresponding combinations fail fast;
9. ETTh1 results are described as theory-guided unless an actual law is supplied.

## 11. Recommended Delivery Order

1. Add fail-fast guards for unsafe combinations; first make production honor `tft_target_pos` and return DataParallel-gatherable quantile/MoE outputs. Fix static handling before static constraints and FFT/higher-order defects before enabling those extensions. Fix benchmark `use_amp` before ETTh1 ablation.
2. Add dataset metadata and strict physics schema.
3. Implement differentiable scaling and output-level constraints.
4. Integrate and test the production experiment path.
5. Prove behavior on a synthetic physical system.
6. Run ETTh1 theory-guided ablations.
7. Add lightweight VSN priors.
8. Add counterfactual monotonicity.
9. Consider hard projection, simulator coupling, or physics-baseline-plus-residual architectures only after domain validation.

## 12. Native TFT Implementation Task Cards

### `TFT-H01` — Freeze the baseline and repair test discovery

**Depends on:** `G0`

**Why this comes first**

The main native test class can be skipped because an unrelated Nixtla dependency is missing, `run_tests.py` searches the wrong directory case, and the deep benchmark has an undefined `use_amp` variable and misleading epoch behavior. Fixing code before fixing discovery can create false confidence.

**Primary files**

- `tests/test_tft_comprehensive.py`
- `tests/test_tft_bugfix_diagnostics.py`
- `tests/test_tft_interpretation_and_exp.py`
- `tests/test_tft_deep_ett.py`
- `run_tests.py`

**Create**

- `tests/test_tft_core_contracts.py`
- `tests/test_tft_experiment_contracts.py`
- `tests/test_tft_extension_contracts.py`
- `tests/test_tft_profiles.py`

**Implementation steps**

1. Copy the two baseline commands from the progress tracker into the evidence entry.
2. Remove any class-level skip that makes native tests depend on Nixtla availability.
3. Apply Nixtla skips only to tests that import or execute Nixtla code.
4. Change `run_tests.py` discovery from `Tests` to lowercase `tests`, or replace it with a documented pytest wrapper.
5. In `tests/test_tft_deep_ett.py`, define `use_amp` from parsed configuration before it is read.
6. Add explicit `--epochs` and `--max-train-batches` arguments.
7. Ensure a requested two-epoch run cannot inherit `baseline_safe`'s 150 epochs.
8. Create the four empty-but-valid focused test modules above. Each later task adds tests to the correct module.

**Focused checks**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_comprehensive.py \
  tests/test_tft_interpretation_and_exp.py

PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_bugfix_diagnostics.py \
  tests/test_tft_scripts_smoke.py \
  tests/test_revin_ablation.py
```

**Done when**

- native tests execute even if Nixtla is unavailable;
- both baseline commands remain green;
- the benchmark dry run prints the requested epoch/batch limits;
- `run_tests.py` finds lowercase `tests`;
- evidence is recorded under `TFT-H01`.

**Do not**

- modify `TFT_Nixtla.py`;
- use an expensive training run to test a shape or contract issue;
- mark known semantic defects fixed merely because old tests pass.

**Unlocks:** reliable red/green evidence for all native tasks.

---

### `TFT-C08` — Introduce a resolved feature schema and strict validation

**Depends on:** `G0`

**Problem**

Feature roles are resolved from raw indices before the effective `M`/`MS`/`S` schema is known. Registered ETT indices `0..6` fail for `features='S'`, encoder mark time length is unchecked, custom-known CLI controls are incomplete, and detailed frequency aliases can fail.

**Primary files**

- new `utils/tft_schema.py`
- `models/TemporalFusionTransformer.py`
  - `get_typepos()`
  - `get_target_pos()`
  - `get_known_len()`
  - `Model.__init__()`
  - `_validate_inputs()`
- `data_provider/data_loader.py`
- `run.py`

**Target API**

```python
@dataclass(frozen=True)
class ResolvedTFTSchema:
    feature_names: tuple[str, ...]
    observed_positions: tuple[int, ...]
    static_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    known_feature_names: tuple[str, ...]
    features_mode: Literal["M", "MS", "S"]
    enc_in: int
    c_out: int
```

**Implementation steps**

1. Make each supported dataset expose feature names from the actual `df_data.columns` after feature-mode selection and any target-last reorder.
2. Resolve schema indices against that effective order.
3. Give explicit user-supplied roles precedence over registry defaults.
4. For registered ETT with `features='S'`, resolve observed and target positions to `[0]`.
5. Validate:
   - every index is an integer;
   - every index is within `[0, enc_in)`;
   - no list contains duplicates;
   - static and observed roles are disjoint;
   - each target is historically observed;
   - target count equals `c_out`.
6. Validate `x_mark_enc.shape[1] == seq_len`.
7. Keep the decoder mark check at `label_len + pred_len`.
8. Add CLI arguments for `tft_known_len`, `tft_known_max_channels`, and optional custom-known names.
9. Resolve detailed frequencies such as `15min` and `3h` through the repository time-feature utility or `pandas.to_offset()`, not a short-code-only dictionary.
10. Store the resolved schema on the model and reuse it instead of resolving roles independently in multiple functions.

**Tests to write first**

- `test_registered_ett_single_feature_schema`
- `test_explicit_schema_overrides_registry`
- `test_static_observed_overlap_rejected`
- `test_target_must_be_observed`
- `test_encoder_mark_length_must_equal_sequence_length`
- `test_detailed_frequency_known_length`
- `test_custom_known_requires_names_and_length`

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_core_contracts.py -k schema
```

**Done when**

- M, MS, and S schemas resolve before the first forward;
- all invalid mappings fail with the role name and invalid value;
- explicit roles are never silently overwritten by a registry;
- custom-known tensor width and names match;
- detailed frequencies no longer fail exact-key lookup.

**Common mistakes**

- using raw CSV indices after the target has been moved;
- treating the target as encoder position `-1`;
- “repairing” an invalid explicit mapping silently;
- inferring categorical/static meaning from numeric values.

**Unlocks:** `PHY-D01`.

---

### `TFT-C09` — Make every production path honor `tft_target_pos`

**Depends on:** `G0`

**Problem**

The model uses `tft_target_pos`, but the long-term experiment still selects truth with `f_dim`. It also tiles reduced predictions before inverse scaling. Non-last MS and non-contiguous multi-output targets are trained or scored against the wrong channels.

**Primary files**

- new or shared target helpers in `utils/tft_schema.py`
- `exp/exp_long_term_forecasting.py`
- related experiment tests

**Target helpers**

```python
def resolve_target_positions(args) -> tuple[int, ...]:
    ...


def select_tft_truth(batch_y, pred_len, target_positions):
    # batch_y: [B, label_len + pred_len, enc_in]
    return batch_y[:, -pred_len:, :].index_select(-1, target_index_tensor)


def inverse_transform_selected(values, scaler, target_positions):
    # values: [..., c_out]
    # use scaler.mean_[target_positions] and scaler.scale_[target_positions]
    ...
```

**Implementation steps**

1. Resolve target positions once and keep their declared order.
2. Preserve full `batch_y`; never overwrite it with the target-only slice.
3. Use `index_select()` on full truth for point loss, quantile loss, validation, test, and plots.
4. Apply target indices to truth only. Model forecast channels already use `c_out` order.
5. Keep legacy `f_dim` behavior only for non-TFT models.
6. Remove TFT prediction tiling before inverse transformation.
7. Inverse-transform each predicted channel with the corresponding target mean and scale.
8. Assert forecast and selected-truth shapes match immediately.
9. Expose the same helper to later physics code.

**Tests to write first**

- `test_nonlast_ms_target_mapping_end_to_end`
- `test_noncontiguous_multioutput_target_mapping`
- `test_selected_target_inverse_transform`
- `test_tft_path_never_uses_f_dim`

Use deliberately different scaler means/scales so a wrong channel is obvious.

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_experiment_contracts.py -k target
```

**Done when**

- MS target position 1 is not replaced with the last channel;
- target mapping `[4, 1]` preserves that order through loss and metrics;
- no TFT path tiles predictions to `enc_in`;
- plots use the selected target;
- point, quantile, metrics, and future physics share one mapping.

**Unlocks:** `TFT-C03`, `PHY-D01`.

---

### `TFT-O01` — Replace mutable side channels with gatherable output

**Depends on:** `G0`

**Problem**

Production reads `last_quantile_predictions` and `last_moe_aux_loss` from the base module. Replica attributes are not a valid `DataParallel` output, so quantiles can disappear and configured MoE loss can be silently omitted.

**Primary files**

- `models/TemporalFusionTransformer.py`
- `exp/exp_long_term_forecasting.py`
- `utils/tools.py`
- focused model/experiment tests

**Target API**

Use a tensor-only dictionary or `NamedTuple`; do not use a plain dataclass as the direct `DataParallel` return.

```python
class TFTForecastOutput(NamedTuple):
    point_forecast: torch.Tensor          # [B, pred_len, c_out]
    point_full: torch.Tensor              # [B, seq_len + pred_len, c_out]
    quantile_forecast: torch.Tensor | None  # [B, pred_len, Q, c_out]
    moe_importance_sum: torch.Tensor | None
    moe_token_count: torch.Tensor | None
```

**Implementation steps**

1. Add a structured-output request such as `return_auxiliary=True`.
2. Keep the legacy tensor return temporarily when the request is false.
3. Make the production experiment request structured output.
4. Return forecast-only and compatibility full tensors explicitly.
5. Return quantiles directly.
6. Return MoE sufficient statistics that can be summed after replica gathering; do not average replica-local scalar means blindly.
7. Compute the global MoE auxiliary term from gathered sums/counts.
8. Stop using `get_auxiliary_loss()` for production training.
9. Retain mutable `last_*` values only as deprecated diagnostics until callers migrate.
10. Ensure all replicas return identical keys/field structure even when an optional value is absent.

**Tests to write first**

- `test_structured_output_shapes`
- `test_parallel_output_gathers_quantiles_and_moe`
- `test_experiment_does_not_read_last_attributes`
- `test_global_moe_reduction_uses_counts`

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_core_contracts.py \
  tests/test_tft_experiment_contracts.py \
  -k "structured or parallel or moe"
```

**Done when**

- single-device and DataParallel output the same semantic fields;
- quantile and MoE loss reach the experiment without mutable attributes;
- a non-zero configured MoE coefficient cannot be silently ignored;
- legacy tensor callers still work for one documented migration release.

**Common mistakes**

- returning a plain dataclass from `DataParallel`;
- gathering a replica mean instead of numerator/count;
- returning different dictionary keys on different replicas;
- keeping production coupled to `model.module.last_*`.

**Unlocks:** `TFT-C03`, `THY-A01`.

---

### `TFT-C01` — Preserve raw static values through normalization

**Depends on:** `G0`

**Problem**

Manual normalization and RevIN center constant static channels to zero before static embedding.

**Primary files**

- `models/TemporalFusionTransformer.py`
  - `Model.forecast()`
  - `TFTEmbedding.forward()`
- `layers/StandardNorm.py` only if an explicit normalization mask is introduced

**Tensor contract**

```text
raw x_enc:          [B, seq_len, enc_in]
raw static values:  [B, static_count]
normalized observed:[B, seq_len, observed_count]
```

**Implementation steps**

1. Keep a reference to raw encoder input before internal normalization.
2. Select declared static channels from raw input.
3. Validate each static channel is constant across encoder time within a documented tolerance.
4. Take one timestep only after validation.
5. Pass `static_values` explicitly to `TFTEmbedding`.
6. Continue to use normalized dynamic observed inputs.
7. Reject a target/static overlap.
8. In this safety task, preserve continuous static values in dataset coordinates. Typed categorical handling belongs to `TFT-A03`.

**Tests to write first**

- `test_static_values_survive_manual_normalization`
- `test_static_values_survive_revin`
- `test_static_context_changes_with_entity_value`
- `test_nonconstant_static_feature_rejected`
- `test_static_target_overlap_rejected`

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_core_contracts.py -k static
```

**Done when**

- static values 5 and 50 remain distinguishable;
- identical dynamics with different static values produce different static contexts in evaluation mode;
- both manual normalization and RevIN pass;
- nonconstant declared-static input fails clearly.

**Do not**

- write static values back into normalized dynamic input as an undocumented exception;
- infer static roles from one constant window;
- standardize categorical IDs per window.

**Unlocks:** `TFT-C06` and static-operand physics after its other dependencies.

---

### `TFT-C02` — Correct LSTM hidden/cell order

**Depends on:** `G0`

**Primary file**

- `models/TemporalFusionTransformer.py`, `TemporalFusionDecoderLayer.forward()`

**Implementation steps**

1. Find both LSTM and hybrid initialization paths.
2. Rename local variables to `initial_hidden` and `initial_cell`.
3. Pass:

   ```python
   initial_state = (
       c_h.unsqueeze(0),
       c_c.unsqueeze(0),
   )
   ```

4. Pass the same order into `HybridTemporalBackbone`.
5. Add a short comment that PyTorch expects `(h_0, c_0)`.

**Tests to write first**

- `test_lstm_receives_hidden_then_cell_context`
- `test_hybrid_receives_hidden_then_cell_context`

Use a recording/mock recurrent module. Shape-only tests do not prove identity.

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_core_contracts.py -k state_order
```

**Done when**

- the first state is exactly `c_h`;
- the second state is exactly `c_c`;
- both recurrent modes pass forward/backward tests.

---

### `TFT-C03` — Make point and quantile objectives coherent

**Depends on:** `TFT-O01`, `TFT-C09`

**Primary files**

- `models/TemporalFusionTransformer.py`
- `utils/losses.py`
- `exp/exp_long_term_forecasting.py`
- `layers/StandardNorm.py`
- `run.py`

**Frozen output modes**

```text
point:    train/evaluate point head only
quantile: train quantiles; require 0.5; point metric is trained median
joint:    train both with explicit positive coefficients
```

**Implementation steps**

1. Add `tft_output_mode = point | quantile | joint`.
2. Canonicalize quantile levels once at startup.
3. Reject duplicates and levels outside `(0, 1)`.
4. Pass the same ordered tuple to the head, `QuantileLoss`, metrics, manifest, and physics adapter.
5. In point mode, do not instantiate or expose an untrained quantile head.
6. In quantile mode:
   - require exact level `0.5`;
   - derive point forecast from that trained slice;
   - do not score the unrelated point projection.
7. In joint mode, require explicit positive point/quantile coefficients.
8. Implement an ordered quantile head: an unconstrained base plus cumulative `softplus` increments.
9. Make the RevIN effective affine scale positive, for example `softplus(raw_scale) + eps`.
10. Provide checkpoint migration for legacy RevIN affine weights.
11. Verify ordering after final de-normalization.
12. Add pinball, coverage, interval width, and crossing-rate metrics.

**Tests to write first**

- `test_quantile_only_evaluates_trained_output`
- `test_point_mode_has_no_untrained_quantile_head`
- `test_unsorted_quantiles_canonicalized_once`
- `test_duplicate_quantiles_rejected`
- `test_quantile_outputs_do_not_cross`
- `test_quantile_order_survives_revin_denormalization`
- `test_revin_effective_scale_is_positive`
- gradient tests for every evaluated head

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_core_contracts.py \
  tests/test_tft_experiment_contracts.py \
  -k quantile
```

**Done when**

- every reported prediction comes from a trained head;
- model and loss share one quantile order;
- quantiles cannot cross before or after RevIN;
- DataParallel and single-device metrics agree;
- point/MSE mode has no misleading untrained probabilistic output.

**Common mistakes**

- sorting predicted values after the fact, which changes quantile identity;
- using the nearest-to-0.5 level and calling it a median;
- fixing the model order without fixing `QuantileLoss`;
- allowing negative RevIN scale to reverse an ordered head.

---

### `TFT-C04` — Repair FFT top-amplitude selection

**Depends on:** `G0`

**Primary file**

- `layers/TemporalFusion_layers.py`, `SpectralBranch`

**Frozen safety design**

- choose top bins per sample and latent channel;
- physical FFT-bin indices select data;
- rank positions select the `modes`-wide weight table;
- unrelated batch members must not change a sample's output.

**Implementation steps**

1. Compute `x_ft` as `[B, D, F]`.
2. Compute amplitudes without reducing batch or channel dimensions.
3. Use `topk(..., dim=-1)` to obtain `[B, D, k]` physical-bin indices.
4. Gather selected complex coefficients with those indices.
5. Use `weight_real[:, :k]` and `weight_imag[:, :k]` as rank weights. Do not index these tensors with physical bins.
6. Scatter transformed coefficients back to their physical bins.
7. Keep the output spectrum shape `[B, D, F]`.
8. Document whether DC may be selected.
9. Record in interpretation metadata that FFT sees the full history-plus-known-future latent sequence.

**Tests to write first**

- `test_fft_top_amplitude_high_bin`
- dominant bin below/equal/above `modes`
- `test_fft_selection_batch_permutation_invariant`
- `test_fft_selection_batch_composition_invariant`
- forward/backward finite-gradient test

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_extension_contracts.py -k fft
```

**Done when**

- no valid high bin can index outside the weight table;
- adding an unrelated sample does not change an existing sample;
- permuting batch order only permutes output;
- gradients reach rank weights.

**Unlocks:** `TFT-A01`.

---

### `TFT-A01` — Implement a real learned frequency selector

**Depends on:** `TFT-C04`

**Status on Tuesday, July 28, 2026**

Done in the current worktree and recorded in `EV-IMP-014`.

**Problem at the audited base commit**

The current learned mask has shape `[1, D, 1]`, so it cannot vary by frequency.

**Implemented design**

1. Keep `low`, `top_amplitude`, and `learned` as separate execution paths.
2. Parameterize learned logits over spectral anchor bins with shape `[1, D, modes]`.
3. Interpolate learned logits and complex weights onto the runtime `rfft` grid.
4. Preserve the existing `modes` meaning as the learned spectral-anchor budget.
5. Export learned-mask summaries through interpretation metadata.

**Regression tests**

- two frequency logits can learn different values;
- runtime lengths slice correctly;
- mask gradients are non-zero;
- saved/loaded checkpoints reproduce the mask.

**Done when**

The option selects frequencies rather than applying one scalar per latent channel.

---

### `TFT-C05` — Repair higher-order interaction gating

**Depends on:** `G0`

**Primary file**

- `layers/TemporalFusion_layers.py`, `HigherOrderInteractionBlock`

**Frozen design**

Use independent sigmoid gates:

```text
order 2: terms=[pair], gates=[pair_gate]
order 3: terms=[pair,triple], gates=[pair_gate,triple_gate]
```

The residual input is the implicit “no interaction” path.

**Implementation steps**

1. Set `num_terms = interaction_order - 1`.
2. Make `gate_projection` output `num_terms`.
3. Build exactly one pair term and, for order 3, one triple term.
4. Apply `sigmoid`, not a softmax whose sum must be one.
5. Stack gates/terms on matching axes.
6. Return interpretation gates with documented shape `[B, T, num_terms]`.
7. Preserve pair/triple normalization already used by the block.

**Tests to write first**

- `test_higher_order_two_forward_backward`
- `test_higher_order_two_has_gate_gradient`
- `test_higher_order_three_forward_backward`
- `test_higher_order_terms_match_gate_count`
- perturbing gate logits changes output

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_tft_extension_contracts.py -k higher_order
```

**Done when**

- order 2 is gate-dependent;
- order 3 does not crash;
- every gate receives a meaningful gradient;
- a gate near zero can suppress its term.

---

### `TFT-C06` — Preserve static interpretation payload

**Depends on:** `TFT-C01`

**Primary files**

- `models/TemporalFusionTransformer.py`
- `utils/tft_interpretation.py`

**Implementation steps**

1. Keep the nested context structure returned by `StaticCovariateEncoder`:

   ```text
   c_s, c_c, c_h, c_e
   ```

2. Do not pass it through a helper that expects only top-level `selection` and `graph_attention`.
3. Return context-keyed selection and graph payloads.
4. Detach only for reporting; training auxiliaries remain differentiable when explicitly requested later.
5. Map the variable axis to schema names.
6. After `TFT-A03`, migrate to one canonical static-VSN payload and document compatibility.

**Tests**

- `test_static_interpretation_payload_preserved`
- `test_static_interpretation_uses_feature_names`
- each context key has expected shape

**Done when**

Configured static input never yields `static_vsn_weights=None`, and exported indices are feature names.

---

### `TFT-C07` — Close the short-term contract honestly

**Depends on:** `G0`

**Chosen first-release behavior**

Reject short-term forecasting at construction. Do not implement an unreviewed markless time-feature strategy in a correctness patch.

**Primary files**

- `models/TemporalFusionTransformer.py`
- `run.py`
- short-term/native contract tests

**Implementation steps**

1. At the start of model construction, check `task_name`.
2. Accept `long_term_forecast`.
3. Raise a clear `NotImplementedError` for `short_term_forecast` before `get_typepos()` or tensor validation.
4. Explain that M4 marks/output semantics are not implemented.
5. Reject all other unsupported task names with the supported set.
6. Remove short-term from the normal forward success branch.

**Tests**

- `test_short_term_rejected_before_m4_schema_lookup`
- `test_unsupported_task_rejected_at_construction`
- `test_long_term_still_constructs`

**Done when**

There is no path where normal short-term construction succeeds and later crashes on `None` marks or output length.

**Future support**

Create a new task only after defining relative-time known features, prediction-only output, M4 schema, and experiment-level train/validation/test tests.

---

### `TFT-T01` — Native semantic release gate

**Depends on:** every native safety task

**Steps**

1. Run all focused native contract files.
2. Run both recorded baseline suites.
3. Run deterministic CPU forward/backward for manual norm and RevIN.
4. Run AMP smoke where available.
5. Run the DataParallel gather test.
6. Cover M, non-last MS, non-contiguous outputs, and S.
7. Cover point, quantile, and joint output modes.
8. Run `git diff --check`.
9. Review migration notes for structured output and RevIN parameters.

**Required command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_core_contracts.py \
  tests/test_tft_experiment_contracts.py \
  tests/test_tft_extension_contracts.py \
  tests/test_tft_comprehensive.py \
  tests/test_tft_interpretation_and_exp.py \
  tests/test_tft_bugfix_diagnostics.py \
  tests/test_revin_ablation.py
```

**Done when**

- every native blocker is `DONE`;
- every listed command is recorded and green;
- no test is hidden behind Nixtla availability;
- unsupported combinations fail before the first batch;
- `G1` is marked passed.

---

### `TFT-P01` — Add centralized TFT profiles

**Depends on:** `G1`

**Primary files**

- new `utils/tft_config.py`
- `run.py`
- `models/TemporalFusionTransformer.py`

**Profiles**

| Profile | Purpose | Required defaults |
|---|---|---|
| `canonical` | Trustworthy reference | LSTM, softmax VSN, interpretable attention, quantile output; experimental branches off |
| `extended_safe` | Repaired optional features | All additions explicit and individually tested |
| `experimental_full` | Research combinations | Allows all hardened experimental branches with warnings |

**Implementation steps**

1. Add `--tft_profile`.
2. Resolve profile defaults in one function used by CLI and direct construction.
3. Validate overrides after applying the profile.
4. Reject incompatible combinations instead of silently ignoring flags.
5. Add a stable TFT configuration digest to setting/checkpoint paths.
6. Store the resolved profile in result metadata.

**Tests**

- `test_cli_and_model_defaults_match`
- `test_canonical_profile_flags`
- `test_profile_rejects_incompatible_override`
- `test_config_digest_changes_for_material_flag`

**Done when**

The same profile produces identical resolved configuration through `run.py` and direct tests.

---

### `TFT-A02` — Remove structurally dead profile parameters

**Depends on:** `TFT-P01`

**Implementation steps**

1. Instantiate static GRNs only when static input exists.
2. Instantiate residual projection/gate only when residual bypass is active.
3. Instantiate only the selected VSN gating path.
4. Instantiate point/quantile heads only for the selected output mode.
5. Instantiate extension branches only when enabled.
6. Add a parameter-liveness audit that runs one forward/backward and lists unexpected `grad is None`.

**Tests**

- `test_selected_profile_has_no_unexpected_dead_parameters`
- parameter-count regression per profile
- state-dict keys do not include disabled branches

**Done when**

Every expected trainable parameter in the selected profile receives a gradient.

---

### `TFT-A03` — Restore canonical embeddings and static encoder

**Depends on:** `TFT-P01`

**Primary files**

- `models/TemporalFusionTransformer.py`
- optionally new `layers/TFTEmbedding.py`

**Implementation steps**

1. Replace observed continuous `DataEmbedding` with per-variable `nn.Linear(1, d_model)`.
2. Add categorical `nn.Embedding` only when schema supplies type and cardinality.
3. Never dataset-standardize categorical IDs.
4. Remove per-variable circular Conv1d and duplicated positional buffer in canonical mode.
5. Use one static VSN.
6. Feed its selected static representation through four context GRNs.
7. Keep a shared positional representation only where the temporal architecture needs it.
8. Document or migrate incompatible legacy checkpoints.

**Tests**

- pointwise embedding does not use neighboring timesteps;
- categorical range validation;
- exactly one static VSN invocation;
- no duplicated position buffers;
- parameter/checkpoint-size comparison;
- liveness test from `TFT-A02`.

**Done when**

Canonical mode matches the original TFT component contract and contains no unexpected dead parameters.

---

### `TFT-E01` — Benchmark the canonical reference

**Depends on:** `TFT-A02`, `TFT-A03`

**Steps**

1. Freeze seeds, data split, optimizer, initialization policy, and parameter budget.
2. Compare canonical with `extended_safe`.
3. Report:
   - MSE/MAE or pinball risk;
   - coverage and interval width;
   - latency and peak memory;
   - parameter count and checkpoint size;
   - interpretation stability across seeds.
4. Do not enable multiple new extensions and attribute aggregate improvement to one.

**Done when**

A reproducible canonical reference table is saved and `G2` passes.

---

### `TFT-A04` — Repair advanced graph sparsity and scaling

**Depends on:** `G1`

**Status on Tuesday, July 28, 2026**

Done in the current worktree and recorded in `EV-IMP-015`.

**Primary file**

- `layers/AdvancedDynamicGraph.py`

**Implemented design**

1. Define `top_k=0` as truly dense.
2. Evolve graph logits, not already normalized probabilities.
3. Reapply the structural/top-k mask before final softmax.
4. Mask removed edges to negative infinity.
5. Initialize effective evolution strength to `0.1` through a logit parameterization.
6. Replace the `C²` hidden GRU with low-rank source/destination factors.
7. Preserve the current one-shared-graph behavior honestly.
8. Guard edge-feature mode by node/memory limit.

**Tests**

- `test_temporal_sparse_graph_stays_top_k`
- `test_graph_top_k_zero_dense_or_rejected`
- row sums/finite values
- initial alpha
- temporal variation without support expansion
- parameter-growth regression

**Done when**

Temporal evolution cannot recreate masked edges, and memory scaling is documented/tested.

---

### `TFT-A05` — Repair MoE capacity, reduction, and naming

**Depends on:** `G1`

**Status on Tuesday, July 28, 2026**

Done in the current worktree and recorded in `EV-IMP-016`.

**Primary files**

- `layers/TemporalFusion_layers.py`
- model structured-output plumbing
- experiment loss composition

**Implemented design**

1. Compute capacity with `ceil()`.
2. Enforce minimum capacity one when tokens exist.
3. After pruning, restore each token's strongest route if its sum is zero.
4. Remove the prior `.item()`-driven per-expert capacity logic from the pruning path.
5. Return global-reducible importance and load statistics.
6. Include both importance and load-balancing terms.
7. Aggregate all decoder layers explicitly.
8. Call the current implementation `dense_compute_topk_mixing` unless real token dispatch is implemented.
9. Treat sparse dispatch as a later performance subtask.

**Tests**

- `test_moe_small_batch_has_nonzero_route`
- `test_moe_every_token_has_route`
- heavy imbalance/capacity case
- DataParallel global reduction
- selected experts receive gradients

**Done when**

`B=1,T=1` cannot produce an all-zero mixture and configured aux loss cannot disappear.

---

### `TFT-A06` — Make interpretation schema- and profile-aware

**Depends on:** `TFT-P01`

**Primary files**

- `utils/tft_interpretation.py`
- model interpretation payload

**Implementation steps**

1. Export named variables instead of flattened tensor indices.
2. Preserve time, variable, and selection-head axes.
3. Add flags:
   - `is_canonical_vsn_attribution`;
   - `uses_graph_pre_mixing`;
   - `uses_vsn_bypass`;
   - `uses_noninterpretable_attention_branch`;
   - `uses_global_spectral_mixing`;
   - `routing_is_detached`.
4. Refuse the label “canonical TFT importance” when flags invalidate it.
5. Define whether multihead selection is averaged, retained, or summarized.

**Done when**

Every exported value has a feature name, axis definition, and caveat metadata.

Status: Done in the current worktree; see `EV-IMP-021`.

---

### `TFT-A07` — Repair lag masks and physical positions

**Depends on:** `G1`

**Primary files**

- `layers/TemporalFusion_layers.py`
- decoder position plumbing

**Implementation steps**

1. Add key-padding-mask support to attention.
2. Mask the first `lag` positions introduced by shifting.
3. Pass physical key position `index - lag` for RoPE/ALiBi.
4. Reject `lag >= active_sequence_length`.
5. Decide whether the branch is exact-lag or shifted-history attention and name it honestly.
6. Carry original position indices through temporal compression.
7. Reject lag/compression combinations until position plumbing exists.

**Tests**

- padded lag keys receive zero probability;
- shifted physical positions are used;
- excessive lag fails;
- compressed positions remain monotonic original coordinates.

Status: Done in the current worktree; see `EV-IMP-017`.

---

### `TFT-A08` — Move runtime diagnostics out of the hot path

**Depends on:** `G1`

**Primary files**

- `models/TemporalFusionTransformer.py`
- launcher/run configuration

**Implementation steps**

1. Remove MIOpen/HSA environment mutation from model import.
2. Put documented hardware workarounds in the launcher before Torch initialization.
3. Add `tft_debug_checks`.
4. Keep cheap schema/shape checks always enabled.
5. Gate repeated deep `isfinite()` reductions and warning synchronizations behind debug mode.
6. Benchmark debug on/off.

**Tests**

- importing the model does not mutate tracked environment variables;
- debug on/off produce equal finite-input output in evaluation;
- invalid shape/schema still fails with debug off.

Status: Done in the current worktree; see `EV-IMP-019`.

---

### `TFT-A09` — Consolidate configuration and remove ignored knobs

**Depends on:** `TFT-P01`

**Implementation steps**

1. Move every native TFT default into `utils/tft_config.py`.
2. Make `run.py`, direct construction, scripts, and printed args use it.
3. Reject or clearly warn that native TFT ignores `d_ff`.
4. Clarify `tft_temporal_backbone_layers` scope.
5. Include all material flags in the configuration digest.
6. Update help strings to match actual semantics.

**Tests**

- CLI/direct default parity;
- ignored knobs fail/warn;
- material flag changes digest;
- identical resolved config reuses digest.

Status: Done in the current worktree; see `EV-IMP-018`.

---

### `TFT-A10` — Add attention-probability dropout

**Depends on:** `G1`

**Primary files**

- exact/SDPA self-attention and cross-attention implementations
- TFT config

**Implementation steps**

1. Separate probability dropout from output dropout.
2. Exact path: apply dropout to probabilities only in training.
3. SDPA path: pass configured probability dropout in training and zero in evaluation.
4. Return pre-dropout probabilities for interpretation.
5. Document the returned tensor.

**Tests**

- training dropout changes stochastic output;
- evaluation is deterministic;
- exact and SDPA evaluation agree under shared weights/tolerances;
- output dropout remains independently configurable.

Status: Done in the current worktree; see `EV-IMP-020`.

## 13. Physics and Theory-Guided Task Cards

The detailed schema, API, unit, training, and verification contracts in Sections 3–10 remain normative. The task cards below tell the implementer how to execute them in dependency order.

### `PHY-C01` — Build the safe configuration layer

**Depends on:** `G0`

**Create**

- `utils/physics_config.py`
- `configs/physics/schema_v1.json`
- `configs/physics/README.md`
- `configs/physics/example_linear_balance.json`

**Modify**

- `requirements.txt` to add a compatibility-pinned Pint dependency

**Implementation steps**

1. Implement frozen dataclasses for the root spec, operands, constraints, `dt`, penalties, feasibility, and quantile policy.
2. Load JSON only.
3. Reject unknown root and constraint fields.
4. Validate the exact version.
5. Reject duplicate constraint names.
6. Validate positive tolerance, non-negative weight, valid operator, bounds, reduction, and penalty parameters.
7. Restrict unit-string size/characters and reject newlines, `=`, registry directives, user unit definitions, contexts, and callbacks.
8. Instantiate Pint only during load/resolve.
9. Validate dimensional compatibility for linear/polynomial terms.
10. Compute affine unit conversion from converted zero and one.
11. Keep `future_label` disabled by default.
12. Hash canonical sorted JSON content.

**Tests to write first**

- invalid version/unknown field;
- duplicate names;
- missing or zero tolerance;
- invalid inequality operator;
- malformed monomial;
- incompatible dimensions;
- unsafe unit string;
- offset and multiplicative conversions;
- no `eval` or callable configuration path.

**Focused command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_physics_losses.py -k config
```

**Done when**

Invalid specifications cannot reach dataset loading or the batch loop.

**Unlocks:** `PHY-D01`; `PHY-I02` unlocks later through `PHY-S01`.

---

### `PHY-D01` — Expose dataset namespace metadata

**Depends on:** `TFT-C08`, `TFT-C09`, `PHY-C01`

**Create**

- `data_provider/physics_metadata.py`

**Modify**

- `data_provider/data_loader.py`
- time-feature utilities if names are not exposed

**Target API**

```python
@dataclass(frozen=True)
class TransformMetadata:
    feature_names: tuple[str, ...]
    coordinate_space: Literal["dataset_standardized", "native_encoded"]
    mean: tuple[float, ...] | None
    scale: tuple[float, ...] | None


@dataclass(frozen=True)
class DatasetPhysicsMetadata:
    input_feature_names: tuple[str, ...]
    known_future_feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    static_positions: tuple[int, ...]
    cadence_seconds: float | None
    regular_cadence: bool
    observed_transform: TransformMetadata
    known_future_transform: TransformMetadata
```

**Implementation steps**

1. Capture input names after actual `M`/`MS`/`S` selection/reordering.
2. Store the training-fitted scaler mean and scale for the full effective input order.
3. Expose generated calendar feature names in tensor order for `timeenc=0` and `timeenc=1`.
4. Treat the last `pred_len` of `batch_y_mark` as standard known future.
5. Require explicit names and transforms for custom-known values.
6. Derive cadence from timestamps.
7. Mark irregular cadence explicitly.
8. Validate native static channels are time-invariant.
9. Keep the standard four-tensor batch unchanged; if masks are later supported, use one shared optional-batch unpack helper.

**Tests**

- ETT hourly/minute names;
- custom M/MS/S order;
- scaler width equals effective feature count;
- time-feature names match tensor width/order;
- cadence detection;
- val/test metadata equals training metadata.

**Done when**

A printed metadata record identifies every channel and transform without opening the CSV.

**Unlocks:** `PHY-S01`.

---

### `PHY-S01` — Resolve names and build differentiable transforms

**Depends on:** `PHY-C01`, `PHY-D01`

**Primary files**

- `utils/physics_config.py`
- `utils/physics_losses.py`

**Implementation steps**

1. Resolve qualified names once at startup.
2. Resolve prediction names through the exact TFT target map.
3. Require history selectors `last`, `lag:k`, or explicit reduction.
4. Treat prediction, known future, and future label as horizon-aligned.
5. Require explicit static broadcasting.
6. Reject inference-unavailable operands.
7. Reject future labels unless enabled and mark them post-hoc/training-only.
8. Compare spec `dt` with measured cadence after unit conversion.
9. Reject irregular cadence in Stage 1.
10. Register scaler mean/scale and unit multiplier/offset as Torch buffers.
11. Apply:

   ```text
   standardized -> raw storage -> declared physical unit
   raw = standardized * scale + mean
   physical = raw * multiplier + offset
   ```

12. Keep Pint out of `forward()`.

**Tests**

- raw/standardized equivalence;
- target positions choose correct scaler entries;
- Celsius/Kelvin and delta-temperature behavior;
- ambiguous/missing names fail;
- gradients, device, and dtype are preserved.

**Done when**

Every runtime conversion is pure Torch and every operand has one unambiguous source.

**Unlocks:** `PHY-L01`.

---

### `PHY-L01` — Implement the Stage-1 physics loss engine

**Depends on:** `PHY-S01`

**Primary file**

- `utils/physics_losses.py`

**Coding order**

1. Namespace selection and mask combination.
2. Linear equality.
3. Linear inequality.
4. Value bounds.
5. Absolute rate bound.
6. Polynomial monomials.
7. Reductions, counts, and metrics.

**Required formulas**

```text
equality normalized residual = residual / tolerance
<= violation = relu(lhs - rhs) / tolerance
>= violation = relu(rhs - lhs) / tolerance
```

Rate path:

```python
path = torch.cat([history_last.unsqueeze(1), forecast], dim=1)
rate = torch.diff(path, dim=1) / dt
violation = torch.relu(torch.abs(rate) - max_abs_rate)
```

**Implementation rules**

- combine every operand mask before reduction;
- report valid and violating counts;
- apply each term weight only after its own reduction;
- define violation as normalized magnitude greater than one;
- use `forecast.sum() * 0` for an all-masked autograd-connected zero;
- reject non-finite inputs with the constraint name;
- never apply generic smoothness and call it physics.

**Tests**

- hand-calculated satisfying/violating cases;
- correct non-zero gradients;
- inequality directions;
- rate history boundary and `dt`;
- polynomial powers;
- mask counts and all-masked case;
- CPU/device/dtype/autocast;
- NaN/Inf messages.

**Done when**

Every supported constraint has an exact numerical and gradient test.

**Unlocks:** `PHY-T01`, `PHY-I01` after its other dependencies.

---

### `PHY-I02` — Add CLI, digest, and resolved-manifest lifecycle

**Depends on:** `PHY-S01`

**Primary files**

- `run.py`
- `utils/print_args.py`
- `utils/physics_config.py`
- experiment setup/test path

**Implementation steps**

1. Add all CLI controls from Section 5.2.
2. Implement enablement validation:
   - no config requires all physics defaults;
   - config plus zero weight keeps training bypassed;
   - positive weight requires config;
   - physics requires native TFT long-term task.
3. Hash canonical raw config and material TFT flags in the setting.
4. After dataset resolution, write `physics_manifest.json` beside the checkpoint.
5. Include raw/resolved digests, repository revision, target map, namespaces, cadence, transforms, units, quantiles, future-label flag, and selection policy.
6. Copy/verify the manifest beside test results.
7. In standalone `--is_training 0`, load and verify it.
8. Never refit transforms on the test interval.
9. Use `args.checkpoints`, not a hard-coded path.
10. Print raw controls before dataset loading and resolved mappings afterward.

**Tests**

- enablement combinations;
- digest changes for a material field;
- canonical JSON stability;
- manifest mismatch for changed target/unit/scaler;
- standalone load;
- no test-range fitting.

**Done when**

A materially incompatible run cannot silently reuse a checkpoint.

---

### `PHY-I01` — Integrate physics into production training

**Depends on:** `G1`, `PHY-L01`, `PHY-I02`

**Primary files**

- `exp/exp_long_term_forecasting.py`
- `utils/tools.py`

**Target loss container**

```python
@dataclass
class ForecastLosses:
    primary: torch.Tensor
    moe: torch.Tensor
    physics: torch.Tensor
    total: torch.Tensor
    physics_result: PhysicsLossResult | None
```

**Implementation steps**

1. Build runtime only after training-fitted metadata is available.
2. Preserve full `batch_y`.
3. Gather history from `batch_x`.
4. Gather known future from the final `pred_len` `batch_y_mark`.
5. Gather future labels only when explicitly allowed.
6. Gather static from validated channels.
7. Use structured model output and forecast-only slice.
8. Share one composition helper between AMP and non-AMP paths.
9. Compute:

   ```text
   legacy = primary + moe_coefficient * moe
   total = legacy + scheduled_physics_weight * physics
   ```

10. Preserve legacy early-stopping objective by default.
11. Short-circuit before physics construction/computation when absent or zero-weight.
12. Aggregate by sample/valid counts rather than unweighted batch means.

**Mandatory zero-weight test**

With identical deterministic CPU model, batch, RNG, and optimizer:

1. run one baseline step;
2. restore state;
3. run config-plus-zero-weight step;
4. assert equal outputs, loss, gradients, optimizer state, and parameters;
5. assert a spy physics module was never constructed/called.

**Done when**

AMP and non-AMP semantics match and zero weight is a real bypass.

**Unlocks:** `PHY-M01`.

---

### `PHY-M01` — Add compliance metrics and checkpoint policies

**Depends on:** `PHY-I01`

**Implementation steps**

1. Log primary, legacy, and total losses separately.
2. Log every term, valid count, violation count/rate, mean normalized violation, and maximum normalized violation.
3. Persist `physics_metrics.json` for held-out test.
4. Mark any metric using future labels as post-hoc only.
5. Implement:
   - `legacy`;
   - `primary`;
   - fixed-`lambda_max` `total`;
   - `feasible_primary`.
6. For feasible-primary, choose lowest primary among feasible epochs.
7. If none are feasible, choose smallest maximum normalized violation, then primary loss.
8. Do not compare total validation values computed with changing training lambda.

**Done when**

Accuracy and physical compliance are separately visible and machine-readable.

**Unlocks:** `PHY-T02`.

---

### `PHY-T01` — Complete the physics unit gate

**Depends on:** `PHY-L01`

Run and record every unit test from Sections 9.1 and the `PHY-C01`–`PHY-L01` cards.

**Required command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q tests/test_physics_losses.py
```

**Done when**

- schema, units, transforms, constraints, masks, gradients, and autocast pass;
- no test imports the model unnecessarily;
- `G3` requirements are satisfied when the other foundation tasks are done.

---

### `PHY-T02` — Production, quantile, mapping, and parity gate

**Depends on:** `PHY-I01`, `PHY-M01`

**Additional prerequisite**

Quantile physics cases require `TFT-C03`; until then they must fail early.

**Quantile rules**

- `point`: use a trained point;
- `median`: require exact level 0.5;
- `all_quantiles`: only pointwise per-channel value/support bounds;
- reject cross-channel equations and cross-horizon rate/dynamics for marginal quantiles.

**Tests**

- normal long-term train/validation/test;
- M, non-last MS, non-contiguous multi-output;
- AMP/non-AMP;
- DataParallel structured output;
- exact zero-weight parity;
- manifest/standalone test;
- held-out metrics persistence;
- quantile physics reaches the trained head;
- unsupported combinations fail before training.

**Required command**

```bash
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_physics_integration.py
```

**Done when**

All production integration criteria pass and `G4` is marked passed.

---

### `PHY-V01` — Prove behavior on a synthetic governing law

**Depends on:** `G4`

**Create**

- `tests/support/synthetic_conservation.py`
- slow end-to-end case in `tests/test_tft_physics_integration.py`

**Recommended system**

```text
storage + release = total_capacity
```

Use slightly law-violating noisy training labels and clean validation/test truth.

**Experimental controls**

- identical initial state;
- identical batches/order;
- identical optimizer and epochs;
- baseline, zero-weight, and non-zero physics runs;
- predeclared compliance improvement;
- predeclared forecast non-inferiority budget.

**Acceptance**

- satisfying tensors have near-zero loss;
- violating tensors have finite non-zero gradient;
- one direct gradient step reduces violation;
- held-out physics violation improves by the declared margin;
- forecast MSE stays within the declared budget;
- zero-weight matches baseline.

**Done when**

Physics value is demonstrated on a known law and `G5` passes.

---

### `PHY-V02` — Run controlled ETTh1 theory-guided ablations

**Depends on:** `PHY-V01`

**Steps**

1. Use the repaired epoch/batch controls.
2. Use identical seeds, initialization, split, optimizer, and data order.
3. Compare zero and multiple non-zero weights.
4. Report MSE/MAE plus raw-unit compliance.
5. Report variance across seeds.
6. Call the result theory-guided unless a sourced governing law, units, and topology are supplied.

**Done when**

The ablation is reproducible and makes no unsupported physics claim.

---

### `THY-A01` — Add selective differentiable VSN output

**Depends on:** `G1`, `TFT-O01`

**Primary files**

- `models/TemporalFusionTransformer.py`
- new theory-prior tests

**Target API**

```python
model(
    ...,
    auxiliary_names=(
        "history_vsn_weights",
        "future_vsn_weights",
    ),
)
```

**Implementation steps**

1. Return only requested auxiliaries in the structured tensor output.
2. Do not force full decoder-attention materialization.
3. Keep training tensors differentiable.
4. Provide detached reporting copies only at the reporting boundary.
5. Document axes:
   - single head `[B,T,C]`;
   - multihead `[B,T,K,C]`.
6. Use schema names and preserve static context identity.

**Done when**

Requested weights receive gradients and unrelated interpretation tensors are not created.

**Unlocks:** `THY-V01`.

---

### `THY-V01` — Add valid VSN theory priors

**Depends on:** `THY-A01`

**Supported priors**

- softmax: positive entropy minimization, KL/cross-entropy to a normalized prior, forbidden-variable mass;
- sigmoid: L1/cardinality only on pre-dropout gates or logits.

**Implementation steps**

1. Add declarative prior rules with namespace and feature names.
2. Resolve names through the schema.
3. Reject L1 on softmax.
4. Define time/head reductions.
5. Include graph-pre-mixing and residual-bypass caveats in metrics.
6. Test gradient direction using a small controlled weight vector.

**Done when**

Every prior has a meaningful non-zero gradient and invalid combinations fail at config resolution.

---

### `THY-M01` — Add counterfactual monotonicity

**Depends on:** `G5`

**Rule fields**

- source namespace and feature;
- target names;
- direction;
- physical perturbation/unit;
- valid min/max;
- affected timesteps/horizons;
- penalty, tolerance, reduction, weight.

**Implementation steps**

1. Perturb in physical coordinates.
2. Convert back to model coordinates.
3. Reject future-label perturbations.
4. Avoid a whole-window constant shift that internal normalization cancels.
5. Run base and perturbed forwards with identical stochastic behavior.
6. Use an exception-safe context manager that snapshots every module's train/eval flag and restores it in `finally`.
7. Penalize:

   ```text
   relu(-direction * (forecast_perturbed - forecast_base) / delta)
   ```

8. Track extra forward-pass latency and memory.

**Tests**

- both directions;
- clipping/range behavior;
- perturbation survives normalization;
- gradient reaches model;
- identical stochastic treatment;
- state restoration after a forced exception;
- AMP.

**Done when**

The penalty measures a controlled partial response, not ordinary temporal correlation, and `G7` passes.

---

### `ADV-P01` — Explore hard physics architectures

**Status:** `DEFERRED`

**Depends on:** `G5` and a validated domain law

Candidate order:

1. exact linear projection in physical space;
2. physics-baseline-plus-learned-residual;
3. differentiable simulator coupling;
4. augmented Lagrangian/adaptive multipliers.

Do not start until the tracker records:

- the governing law and source;
- units and topology;
- feasibility of simultaneous equations/bounds;
- conditioning/rank tests;
- expected inference behavior;
- a soft-loss baseline.

## 14. Post-Matrix Native TFT Semantic-Repair Wave

> Added after the July 2026 ETTh1 advanced-feature matrix exposed a gap between
> structural correctness and intended feature semantics.
>
> These tasks do **not** reopen or erase the historical `DONE` evidence for
> `TFT-A01` through `TFT-A10`. Those tasks repaired the contracts declared at
> that time. `TFT-SR00` through `TFT-SR09` are a new, immutable repair wave for
> defects and experiment confounds discovered by trained-checkpoint audit.

### 14.1 Release rule and execution order

No financial-astrology neural training may start until `TFT-SR09` is `DONE`.
The initial data/provenance audit is complete. Source remediation and re-audit
may resume in parallel when raw OHLC and the generator/convention package
arrive; it does not depend on model semantics.

```text
TFT-SR00 freeze legacy evidence and semantic version
       |
       v
TFT-SR01 reproducibility and paired-ablation contract
       |
       v
TFT-SR02 exact baseline-neutral extension contract
       |
       +--> TFT-SR03 FFT semantics -------------------+
       +--> TFT-SR04 explicit cross-attention --------+
       +--> TFT-SR05 lag/time semantics --------------+
       +--> TFT-SR06 interaction semantics -----------+--> TFT-SR09
       +--> TFT-SR07 temporal compression ------------+    release gate
       +--> TFT-SR08 sparse graph semantics ----------+
```

This wave does not require another full ETTh1 matrix. Acceptance uses:

1. focused semantic tests;
2. synthetic tasks with a known feature relationship;
3. exact initialization/batch-order hashes;
4. all-parameter gradient-liveness audits;
5. checkpoint/config migration tests;
6. one short deterministic micro-training replay;
7. the existing native TFT regression suites.

The adverse one-seed ETTh1 deltas remain diagnostic evidence only. They are not
acceptance thresholds for these tasks.

---

### `TFT-SR00` — Freeze legacy extension semantics and evidence

**Depends on:** completed/closed ETTh1 matrix artifact inventory

**Primary files**

- `utils/tft_config.py`
- checkpoint/config loading utilities
- experiment/result metadata
- migration tests and documentation

**Implementation steps**

1. Add `tft_extension_semantics_version`; new repaired configurations resolve
   to version `2`.
2. Label the July 2026 ETTh1 matrix as legacy semantics version `1` without
   modifying its prediction, checkpoint, or metric artifacts.
3. Include the resolved semantic version in the TFT digest and experiment ID.
4. Treat a checkpoint without version metadata as legacy.
5. Require an explicit compatibility flag to load legacy extension weights.
6. Never silently map a v1 lag, compression, graph, FFT, or interaction weight
   into a v2 module whose computation changed.
7. Save a capability/migration table stating whether each v1 branch can be
   reproduced, partially migrated, or must be retrained.

**Done when**

Version-1 evidence remains reproducible, v1 and v2 digests differ, and an old
checkpoint either enters an explicit legacy path or fails with a precise
migration message.

**Status:** `DONE` on 2026-08-01 with `EV-IMP-025`. The frozen producer and
explicit-v1 replay launchers are separately hashed; 65 legacy records and 158
native regression tests passed. Pending operators are prohibited from emitting
v2 artifacts until their repair task releases them.

---

### `TFT-SR01` — Make seeds, initialization, and data order reproducible

**Depends on:** `TFT-SR00`

**Primary files**

- `run.py`
- `data_provider/data_factory.py`
- `exp/exp_long_term_forecasting.py`
- a new shared reproducibility utility under `utils/`
- new paired-ablation tests

**Defects being repaired**

1. `run.py` parses `--seed` but hardcodes `2021` for model training.
2. Optional modules consume RNG before common downstream modules are created,
   so a one-seed feature comparison does not share baseline initialization.
3. Model construction advances the global RNG before shuffled loaders are
   created, changing batch order between architecture variants.
4. The production loop evaluates the test set every epoch.

**Required configuration**

```text
seed                  overall experiment seed
model_init_seed       shared/base parameter seed
extension_init_seed   optional-branch seed
data_order_seed       sampler order seed
worker_seed           data-loader worker seed base
deterministic_mode    off | warn | strict
evaluation_policy     validation_only | legacy_val_and_test
```

`legacy_val_and_test` may remain only for backward-compatible exploratory
scripts. Financial-astrology runs must use `validation_only`.

**Implementation steps**

1. Replace the hardcoded seed with the resolved CLI/config seed.
2. Add one `set_experiment_seed()` entry point for Python, NumPy, Torch CPU,
   accelerator RNG, and deterministic-backend policy.
3. Give each shuffled `DataLoader` an explicit `torch.Generator` seeded from
   `data_order_seed`; add a deterministic worker seeding function.
4. Shuffle the training loader only. Validation, forward test, and lockbox
   loaders must preserve manifest order.
5. Add a paired-initialization helper that:
   - creates the reference model;
   - creates the variant with its independent extension seed;
   - copies every matching shared tensor by fully qualified name and shape;
   - records unmatched reference/variant tensors;
   - fails if a tensor expected to be shared is missing or shape-changed.
6. Persist all derived seeds, shared-state hash, first-batch index hash, and
   fold manifest hash with each experiment.
7. Remove epoch-level test evaluation from the confirmatory experiment path.
8. Ensure `itr>1` has an explicit seed schedule rather than silently advancing
   process-global RNG state.

**Tests written first**

- two identical runs produce identical shared-state and batch-order hashes;
- changing `--seed` changes both initialization and sampler order;
- baseline and variant have bitwise-identical matching initial tensors;
- baseline and variant see identical ordered sample IDs;
- a validation epoch never iterates the locked test loader;
- worker counts `0` and `>0` preserve sample order for the same manifest.

**Done when**

The run manifest proves which randomness was shared and which was independent,
and a paired ablation no longer changes common initialization or sample order
merely because an optional module exists.

**Status:** `DONE` on 2026-08-01 with `EV-IMP-026`. Independent adversarial
review found no remaining blocker. Production-path tests prove isolated seed
streams, bitwise paired shared state, identical two-arm sample IDs, repeated
non-null pair/checkpoint/order hashes, changed-seed sensitivity, validation-only
fitting, truthful augmented/unaugmented folds, and tamper-evident manifests.

---

### `TFT-SR02` — Add an exact baseline-neutral extension protocol

**Depends on:** `TFT-SR01`

**Primary files**

- `models/TemporalFusionTransformer.py`
- `layers/TemporalFusion_layers.py`
- `utils/tft_config.py`
- extension-neutrality tests

**Target contract**

Every optional extension must expose the same conceptual interface:

```text
base_output
delta_output
residual_strength
combined = base_output + residual_strength * delta_output
```

At `residual_strength == 0`, enabling the extension must be an exact numerical
no-op. No additional LayerNorm may alter `base_output` after the zero residual.

All temporal extensions must also receive an explicit coordinate contract:

```text
positions: [T] or [B,T]
position_unit: steps | trading_sessions | calendar_days
validity/padding mask
monotonicity assertion
```

Astronomy batches use real elapsed calendar-day coordinates. Row indices are a
backward-compatible default only for ordinary regularly sampled datasets.

**Required modes**

```text
off              module not constructed
neutral          module constructed; exact zero residual at initialization
small_residual   explicitly experimental nonzero initialization
legacy           checkpoint-compatibility only
```

**Implementation steps**

1. Add a reusable scalar or channel-wise residual adapter.
2. Initialize confirmatory/ablation variants at exactly zero.
3. Keep the old branch behavior behind an explicit legacy semantic version.
4. Export residual strength, base RMS, delta RMS, and combined-minus-base RMS.
5. Provide a context-manager/helper that temporarily forces an extension to
   exact zero for counterfactual evaluation without rebuilding the model.
6. Add `tft_semantic_version` to the config digest and checkpoint metadata.
7. Document the expected learning behavior at zero: the residual-strength
   parameter learns first; branch gradients become active once it moves away
   from zero. If a branch requires immediate learning, use a declared two-stage
   warm-up or a tiny nonzero exploratory mode, never an undocumented 25–50%
   initial mixture.

**Tests written first**

- each extension in neutral mode matches the reference prediction exactly;
- the reference loss and shared gradients match exactly at zero strength;
- changing strength away from zero changes predictions and reaches the branch;
- disabling a trained branch restores its logged base path;
- save/load preserves semantic version and residual strength.

**Done when**

An extension can be enabled for a fair capacity-matched experiment without
destroying a good baseline before it earns a contribution.

**Status:** `IN_PROGRESS`, claimed by Codex `/root` on 2026-08-01 14:05 IST.

---

### `TFT-SR03` — Make FFT names, mode selection, and fusion truthful

**Depends on:** `TFT-SR02`

**Defect summary**

The matrix used `modes=16, mode_select=learned`, but the implementation
interpolated 16 spectral control points across every runtime FFT bin. The
trained 61-bin mask remained almost uniform around `0.5`; it was a smooth
all-frequency filter, not a 16-mode selector.

**Required public semantics**

```text
low_k             retain the first k bins
top_amplitude_k   retain k sample/channel-specific bins
learned_filter    interpolate learned spectral control points over all bins
learned_sparse_k  optional future mode; explicitly selects/sparsifies k bins
```

The legacy token `learned` must resolve to `learned_filter` with a deprecation
warning and a digest that records the resolved meaning. CLI help must call
`tft_fft_modes` “retained bins” only for the hard-selection modes and “spectral
control points” for `learned_filter`.

**Implementation steps**

1. Rename/resolve modes without silently changing old checkpoints.
2. Use the `TFT-SR02` residual adapter; the temporal/LSTM path has weight 1 at
   neutral initialization.
3. Export the actual FFT contribution rather than ambiguously calling the
   temporal-path weight an “FFT gate.”
4. Apply FFT separately to history and known-future streams. Do not transform
   across their concatenation boundary.
5. Export mask entropy, effective active-bin count, selected/peak bins,
   normalized frequencies, equivalent token periods, and per-channel filter
   norm.
6. Record the configured scope and that FFT periods are token periods unless
   timestamps are regular in a declared physical unit.
7. Keep the first astrology model's FFT switch off. It becomes an isolated
   later ablation only after raw known-covariate signal survives.

**Semantic tests**

- single-frequency sine fixtures recover the expected bin;
- `low_k` and `top_amplitude_k` have exactly `k` active bins;
- `learned_filter` reports all-bin filtering rather than selection;
- neutral fusion is exact baseline parity;
- mask/filter parameters all receive finite gradients;
- runtime-length interpolation and checkpoint reload remain stable.

**Done when**

Configuration, implementation, diagnostics, and claims all describe the same
frequency operation.

---

### `TFT-SR04` — Formalize explicit cross-attention as optional enrichment

**Depends on:** `TFT-SR02`

**Decision**

The current interpretable cross-attention wiring—future queries attending to
historical keys/values—is structurally sound. The repair is integration and
metadata hardening, not a rewrite.

**Implementation steps**

1. Name the role `future_query_to_history_enrichment` in config and payloads.
2. Apply it through the exact neutral residual protocol.
3. Correct interpretation metadata so an interpretable cross-attention module
   is not automatically labeled non-interpretable.
4. Export attention entropy, head disagreement, residual strength, and branch
   knockout delta.
5. Prove that keys/values contain history only and that future market values
   can never enter the branch through the known-future tensor.
6. Keep this branch off in the first astrology run because native causal
   self-attention and the decoder LSTM already carry history into the future.

**Semantic tests**

- perturbing future labels/market placeholders does not alter the branch;
- perturbing historical context can alter future enrichment;
- attention rows sum to one and contain only history columns;
- neutral mode is exact baseline parity;
- reporting distinguishes interpretable and full variants correctly.

**Done when**

The branch is safe to ablate and its one-seed ETTh1 result is no longer
confounded by a random, always-active extra GateAddNorm.

---

### `TFT-SR05` — Separate shifted-history, exact-token, and calendar-time lag semantics

**Depends on:** `TFT-SR02`

**Defect summary**

The existing branch called “lag attention” attends to the full prefix ending at
`t-L`; it does not directly retrieve only `x[t-L]`. When positions are
irregular or compressed, it also reports `positions[j] - L` instead of the
actual source coordinate `positions[j-L]`.

**Required modes**

```text
shifted_prefix_attention   existing behavior, honestly named
exact_token_lag            direct gather/mix of declared row lags
elapsed_time_response      not generic attention; delegated to a calendar-time
                           response-bank interface
```

**Implementation steps**

1. Preserve the old computation under `shifted_prefix_attention`.
2. Add an exact-token mode whose query at `t` consumes only declared source
   tokens such as `t-1`, `t-5`, or `t-20`, with explicit missing masks.
3. Shift positions by index with the values: valid key position at shifted
   index `j` is `positions[j-L]`.
4. Compose position maps correctly through compression; reject combinations
   whose coordinate mapping is unavailable.
5. Add a separate outer residual strength initialized at zero.
6. Use distinct history/future masks and export learned scale weights.
7. For financial astrology, do not encode Saturn/Jupiter persistence with row
   lags. Use actual calendar-day event clocks and response kernels.

**Semantic tests**

- impulse at one source token appears only at the declared exact lag;
- shifted-prefix mode sees the prefix and labels itself accordingly;
- Friday-to-Monday/holiday positions preserve real elapsed coordinates;
- compressed positions point to the actual shifted source;
- padded lag tokens have zero probability and contribution;
- impossible lags fail during configuration.

**Done when**

“Lag” always has an explicit unit and retrieval meaning, and astrology effect
duration is never inferred from trading-row distance.

---

### `TFT-SR06` — Distinguish latent polynomial terms from named covariate interactions

**Depends on:** `TFT-SR02`

**Defect summary**

The current higher-order block runs after the VSN has collapsed named variables.
It multiplies projections of one `[B,T,D]` latent token. It is not evidence for
Mercury–Moon, Jupiter–Saturn, or any other named input interaction.

**Required changes**

1. Rename its resolved semantic role to `latent_polynomial_block`; retain a
   checkpoint alias for the old class/config name.
2. Return the actual residual added after dropout/projection, plus its strength,
   instead of describing the pre-projection tensor as the model contribution.
3. Apply the baseline-neutral residual protocol.
4. Fix the independent per-feature VSN crash: validate the variable dimension
   without calling `len(None)` when per-feature gating is selected.
5. Define a separate pre-VSN `NamedCovariateInteractionEncoder` interface:

   ```text
   input:  [B,T,C,D] plus ordered covariate/group names
   edges:  declared directed pairs or group-pair masks
   output: named low-rank interaction channels plus provenance metadata
   ```

6. Do not instantiate unrestricted `C x C` interactions by default.
7. The astrology-specific implementation later uses body identity, relative
   phase, aspect activation, applying/separating state, and slow×fast group
   masks through this pre-VSN contract.

**Semantic tests**

- a synthetic product target is learnable by the named pair block;
- disabling the named pair leaves the market/calendar baseline exact;
- permuting covariate order with matching names preserves output;
- undeclared pairs cannot contribute;
- latent-polynomial interpretation is never labeled original-variable effect;
- per-feature gating completes forward/backward without dead parameters.

**Done when**

The repository has two honestly distinct concepts: generic latent nonlinearity
and explicit, auditable original-covariate interaction.

---

### `TFT-SR07` — Repair temporal compression liveness and long-sequence semantics

**Depends on:** `TFT-SR02`

**Defect summary**

The ETTh1 matrix forced stride-2 compression onto a 96-step history even though
the feature is intended for long sequences. With one decoder layer, every
decompressor parameter has zero gradient because restored history is never used
by the future-only loss.

**Required design**

```text
off                default
kv_pool            preferred: compress historical attention K/V while retaining
                   the full-resolution residual/query path
legacy_codec       experimental compatibility mode
```

**Implementation steps**

1. Prefer anti-aliased historical K/V pooling over a learnable compress-then-
   reconstruct codec when only future outputs are consumed.
2. Initialize pooling as a fixed low-pass/average operation before allowing a
   small learned residual.
3. Remove final-layer decompressor parameters when their outputs cannot affect
   the forecast, or add an explicit reconstruction objective if a codec is
   intentionally trained.
4. For multi-layer codec mode, prove restored history feeds the next layer and
   every declared trainable codec parameter receives gradients.
5. Preserve both the content-center coordinate and the availability coordinate
   (latest contributing source time) for each pooled token. Causal masks use
   availability time; interpretation may show the content center. Preserve
   padding masks and original endpoints.
6. Reject or loudly require an experimental override for histories below a
   conservative long-sequence threshold; the default first-use threshold is
   at least 512 tokens and must be benchmarked for actual memory/latency gain.
7. Keep compression off in the first astrology run (`seq_len=252`).

**Semantic tests**

- every enabled trainable parameter has finite nonzero gradient;
- an impulse/step/sinusoid survives the declared pooling bandwidth;
- coordinate centers and padding masks are exact;
- `off` and inactive-threshold paths are exact baseline parity;
- short-window activation is rejected unless explicitly experimental;
- memory/latency decreases on a genuinely long synthetic sequence.

**Done when**

Compression provides a measured long-sequence benefit without dead trainable
parameters or an unnecessary lossy round trip.

---

### `TFT-SR08` — Make sparse graph mixing identity-safe, typed, and truthful

**Depends on:** `TFT-SR02`

**Defect summary**

The graph currently replaces VSN inputs with a strong stack of residual
LayerNorm transforms. It learns one adjacency and broadcasts it across heads.
The ETTh1 configuration simultaneously tested an 11-node history graph and a
4-node future graph with `top_k=3`, corresponding to very different densities.

**Required configuration**

```text
history_top_k or history_density
future_top_k or future_density
self_edge_policy: required | allowed | excluded
head_mode: single | true_multihead
temperature
entropy_regularization
support_stability_regularization
residual_strength_init
graph_scope: observed | known | observed_and_known | typed_planetary
```

**Implementation steps**

1. Wrap graph output in an exact zero-initialized residual; no unconditional
   final LayerNorm may change the identity path.
2. Stop broadcasting one adjacency as if it were multi-head. Either implement
   independent head projections/adjacencies or report `single` honestly.
3. Separate history and future sparsity settings and validate them against node
   counts.
4. Make self-edge policy explicit and test it. When residual self-information
   is present, `top_k` counts non-self neighbors and rejects `k > C-1` rather
   than silently treating an almost-complete graph as sparse.
5. Export adjacency entropy, selected support frequency, support turnover,
   self-edge mass, and residual contribution.
6. Add optional temperature and declared regularizers; default them off.
7. Keep generic graph mixing off in the first astrology model. The later
   planetary graph is restricted to typed body/pair geometry before VSN and
   must not indiscriminately mix market, Gregorian calendar, and planet fields.

**Semantic tests**

- zero residual is exact identity;
- true multi-head mode learns distinct adjacency tensors;
- single-head mode reports one head only;
- top-k/density and self-edge policies hold exactly;
- a synthetic known graph is recoverable above a parameter-matched MLP control;
- real and placebo planetary graphs have identical capacity.

**Done when**

Graph metadata is truthful, topology choices are typed and reproducible, and
merely enabling the graph no longer rescales or replaces every VSN input.

---

### `TFT-SR09` — Post-matrix semantic release gate

**Depends on:** `TFT-SR00` through `TFT-SR08`

**Primary outputs**

- one semantic-version migration note;
- one machine-readable extension capability table;
- focused and full regression evidence;
- a short reproducibility micro-run report;
- updated interpretation caveats and config help.

**Required gate checks**

1. `--seed` changes runs and reproduces when repeated.
2. Paired reference/variant common tensors and batch IDs match.
3. Every extension has an exact no-op mode.
4. Every enabled trainable parameter is either gradient-live or explicitly
   documented as frozen/non-trainable.
5. FFT, lag, interaction, compression, and graph names match their mathematics.
6. Irregular/compressed coordinates pass source-position tests.
7. Interpretation payloads report actual post-projection contributions and
   true head counts.
8. Synthetic sine, exact-lag impulse, product interaction, known graph, and
   long-sequence compression fixtures pass.
9. Existing native TFT suites pass.
10. One small paired micro-training replay is reproducible; no full ETT feature
    matrix is required.
11. Old checkpoints either migrate explicitly or fail with a clear semantic-
    version error.
12. `git diff --check` passes.

**Astrology handoff condition**

After this gate passes, the first NIFTY training still starts with FFT,
cross-attention, generic lag, latent-polynomial, generic graph, MoE, compression,
and covariate reattention **off**. The first experiment tests the data contract
and incremental value of a small flat known-future planetary block. Specialized
planetary interactions and calendar-time memory advance only after that cheap
screen survives matched nulls.

**Done when**

`TFT-SR09` has recorded evidence in the tracker and the financial-astrology
umbrella task `FA-TFT-SEM-001` may be marked complete.

## 15. Global Definition of Done

The recommended program is complete only when:

1. Every active task in the tracker is `DONE`.
2. Every `DONE` task has exact evidence and an independent review.
3. Native train/validation/test use one target map.
4. Static values and contexts remain meaningful.
5. Point, quantile, MoE, and auxiliary outputs use gatherable tensor contracts.
6. Canonical TFT works independently of every extension.
7. Physics residuals use named sources and documented physical units.
8. No arbitrary expression execution exists.
9. Zero-weight physics is an exact deterministic-CPU bypass.
10. Train, validation, held-out test, and standalone checkpoint paths are covered.
11. Synthetic known-law compliance improves within a declared forecast budget.
12. ETTh1 claims remain theory-guided unless a real law is supplied.
13. VSN and counterfactual features are independently switchable.
14. Deferred hard-physics work remains off unless its entry gate is satisfied.
15. All required tests and `git diff --check` pass.
