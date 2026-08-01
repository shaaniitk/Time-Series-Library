# TemporalFusionTransformer Deep Analysis Report

> Audited implementation: repository-native `models/TemporalFusionTransformer.py` at commit `564cffbc712f`
>
> Excluded: `models/TFT_Nixtla.py`
>
> Audit date: 2026-07-28
>
> Physics-plan companion: [`implementation_plan.md`](implementation_plan.md)
>
> Progress tracker: [`TFT_Implementation_Progress.md`](TFT_Implementation_Progress.md)
>
> Plan orchestrator: [`TFT_Implementation_Orchestrator.md`](TFT_Implementation_Orchestrator.md)
>
> Planetary/NIFTY research plan: [`Vedic_Astrology_TFT_Implementation_Plan.md`](Vedic_Astrology_TFT_Implementation_Plan.md)

## 1. Executive Verdict

The native implementation is an ambitious experimental superset of TFT, not a faithful drop-in implementation of the architecture in [Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*](https://arxiv.org/abs/1912.09363). It contains useful work—an explicit model-local target map, optional RevIN, interpretable and full attention variants, custom known inputs, graph mixing, lag attention, MoE, spectral processing, compression, rich diagnostics, and substantial tests—but several optional modes are currently incorrect or misleading.

Status note on Tuesday, July 28, 2026:

- the major native safety defects identified in this report have already been patched in the current worktree;
- `TFT-C03` is no longer an open defect and should be treated as implemented unless a new regression is reproduced;
- `TFT-T01` and `G2` are now closed in the current worktree;
- the upgrade recommendations below now primarily serve as the post-`G1` roadmap.

Post-matrix qualification on Friday, July 31, 2026: trained-checkpoint audit
reproduced additional intended-semantic and experiment-control defects outside
the original task acceptance scope. Section 14 and tasks `TFT-SR00`–`TFT-SR09`
supersede any reading that all optional extensions are semantically ready for a
new scientific dataset.

Implementation update on Saturday, August 1, 2026: `TFT-SR00` and `TFT-SR01`
are complete. The 14-case legacy matrix is frozen by a 65-record hash inventory,
v1/v2 identities and checkpoint policies are explicit, and unrepaired operators
cannot be published as v2 artifacts. Production experiments now have isolated
seed streams, deterministic data order, exact paired common initialization,
tamper-evident run/fold manifests, and validation-only fitting. `TFT-SR02`
exact-neutrality/coordinate work is active.

At the audited base commit, the default ETT point-forecast path could train and its existing tests could pass, but that alone did not clear the feature set for production. The key audited issues were:

- static covariates are erased by per-window normalization before the static encoder sees them;
- static LSTM hidden and cell contexts are passed in the wrong order;
- quantile-only training optimizes a head different from the point head later evaluated;
- `top_amplitude` FFT selection can raise an index error;
- third-order interaction mode crashes, while the second-order interaction gate is effectively inert;
- static interpretation weights are discarded by a payload-shape mismatch;
- the “learned” FFT selector was not frequency-selective at the audited base commit;
- temporal graph evolution destroys the promised top-k sparsity;
- several configurations allocate large sets of trainable parameters that can never receive gradients.

In the current worktree, the native safety items in the first seven bullets above have dedicated repairs and tests, and the learned FFT selector has now also been upgraded to a real per-frequency/interpolated spectral mask. The remaining bullets mainly describe post-`G1` upgrade work rather than open release blockers.

The proposed physics-informed plan is feasible only after its contracts are tightened. The main issues are integration at the wrong layer, no treatment of physical units, no named feature/source schema, an ineffective L1 penalty on softmax VSN weights, an underspecified definition of monotonicity, and verification commands that do not perform the advertised test.

### Priority snapshot

| Priority | Finding | Affected mode |
|---|---|---|
| P0 | Quantile objective and evaluated output head are disconnected | `loss=Quantile` |
| P0 | Static values become zero before encoding | any non-empty `tft_static_pos` |
| P0 | Static hidden/cell contexts are reversed | LSTM/hybrid with static inputs |
| P0 | FFT top-amplitude mode can index out of bounds | `tft_fft_mode_select=top_amplitude` |
| P0 | Higher-order order 3 crashes; order 2 gate has no effect | `tft_use_higher_order` |
| P0 | Advertised short-term task fails on its normal `None` time marks | `short_term_forecast` |
| P0 | Production train/test slicing ignores `tft_target_pos` | mapped MS or reduced/multi-output M |
| P1 | Static VSN interpretation payload is always lost | static + interpretation |
| P1 | Missing mark/schema validation can silently misalign known inputs | custom schemas |
| P1 | Physics losses would run in standardized, not physical, coordinates | proposed physics plan |
| P1 | Softmax-VSN L1 is constant and cannot create sparsity | proposed VSN penalty |
| P1 | “Temporal sparse” graph becomes dense after evolution | `temporal_sparse` graph |
| P1 | Dead parameter sets materially inflate models | no-static and sigmoid-VSN modes |
| P1 | Current extension combinations weaken TFT interpretation guarantees | graph/bypass/full/dual/reattention |
| P2 | Benchmark and configuration controls contain dead/broken knobs | deep ETT harness and CLI |

## 2. Scope and Method

The review covered:

- [`models/TemporalFusionTransformer.py`](models/TemporalFusionTransformer.py)
- [`layers/TemporalFusion_layers.py`](layers/TemporalFusion_layers.py)
- [`layers/DynamicGraph.py`](layers/DynamicGraph.py)
- [`layers/AdvancedDynamicGraph.py`](layers/AdvancedDynamicGraph.py)
- [`layers/StandardNorm.py`](layers/StandardNorm.py)
- [`layers/Embed.py`](layers/Embed.py)
- [`exp/exp_long_term_forecasting.py`](exp/exp_long_term_forecasting.py)
- [`data_provider/data_loader.py`](data_provider/data_loader.py)
- [`run.py`](run.py)
- the native TFT tests and the proposed [`implementation_plan.md`](implementation_plan.md)

The audit used source inspection, comparison with the original TFT component contract, the current test suite, and focused runtime probes for paths that the tests do not cover.

### Verification performed

```text
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_comprehensive.py \
  tests/test_tft_interpretation_and_exp.py

55 passed, 1 warning
```

An additional regression/smoke run covered diagnostics, script entry points, and RevIN:

```text
PYTHONPATH=. ./ai_env/bin/pytest -q \
  tests/test_tft_bugfix_diagnostics.py \
  tests/test_tft_scripts_smoke.py \
  tests/test_revin_ablation.py

10 passed
```

Focused probes additionally reproduced:

- different static sample values producing identical normalized static embeddings;
- `static_vsn_weights=None` despite configured static inputs;
- no point-head gradient under quantile-only loss;
- `IndexError` in top-amplitude FFT selection when a high FFT bin dominates;
- near-zero gate gradient for order-2 higher-order interaction;
- a shape error for order-3 higher-order interaction;
- constant L1 norm for softmax VSN weights;
- large sets of unused parameters in the sigmoid-gating VSN path.

Passing shape/smoke tests therefore should not be interpreted as coverage of these semantic contracts.

## 3. Actual Data and Model Flow

The most important fact for both TFT correctness and physics loss design is that there are two normalization layers:

```text
raw physical data
    |
    | dataset StandardScaler (fit on training split)
    v
dataset-standardized batch: x_enc, batch_y
    |
    | TFT per-window normalization or RevIN
    v
TFT embeddings -> VSNs -> temporal decoder -> point/quantile heads
    |
    | TFT reverses only its own per-window normalization
    v
forecast in dataset-standardized coordinates
    |
    | required differentiable dataset inverse scaling for physics
    v
forecast in physical coordinates
```

The dataset scaling is performed in [`data_provider/data_loader.py` lines 51-79](data_provider/data_loader.py#L51) and equivalent loader sections. TFT's internal normalization and de-normalization occur in [`models/TemporalFusionTransformer.py` lines 1214-1228](models/TemporalFusionTransformer.py#L1214) and [1313-1355](models/TemporalFusionTransformer.py#L1313).

The model then prepends a zero history block and returns `[B, seq_len + pred_len, c_out]` at [`models/TemporalFusionTransformer.py` lines 1443-1462](models/TemporalFusionTransformer.py#L1443). All task or physics losses must explicitly select the final `pred_len` steps.

### Core architecture

```text
x_enc --------------------> observed/static embeddings
x_mark_enc + x_mark_dec --> known past/future embeddings
                                  |
                  static VSN/context encoders
                    c_s, c_c, c_h, c_e
                                  |
            history VSN              future VSN
          observed + known              known
                   \                    /
                    temporal backbone
                 LSTM / TCN / hybrid
                          |
             optional FFT / reattention
                          |
                  static enrichment
                          |
       causal self-attention + optional branches
                          |
          GRN or regime-aware MoE + gating
                          |
              point and optional quantile heads
```

## 4. Fidelity to the Original TFT

The original paper defines heterogeneous static, observed, and known-future inputs; per-variable categorical embeddings or continuous linear transforms; one VSN for each input family; four static context GRNs; sequence-to-sequence recurrent locality; interpretable self-attention; gating; and direct quantile forecasts.

| TFT component contract | Repository implementation | Assessment |
|---|---|---|
| Continuous variables use pointwise linear transforms | Each observed/static scalar uses a separate `DataEmbedding`, including circular Conv1d and positional embedding | Material deviation |
| Categorical variables use entity embeddings | Only calendar categorical embeddings are directly supported; generic static categoricals lack a typed schema | Incomplete |
| One static VSN, one past VSN, one future VSN | Four independent static VSNs plus history and future VSNs | Non-canonical and expensive |
| Four context GRNs produce `c_s`, `c_c`, `c_h`, `c_e` | Present, but static inputs are normalized away and LSTM state order is reversed | Currently broken for real static values |
| Recurrent local processing | LSTM, gated TCN, or hybrid; CLI defaults to hybrid | Useful extension, not canonical default |
| Interpretable shared-value attention | Available; full and dual attention are optional | Canonical only in interpretable mode |
| Direct probabilistic quantile output | Optional secondary head; default task is point MSE | Non-canonical objective contract |
| Variable weights support interpretation | Available only under interpretation mode; extensions can invalidate literal attribution | Conditional |
| Gating can skip unnecessary components | GLU path broadly follows intent; SwiGLU is not a bounded suppressive gate | Optional behavior change |

### Recommended fidelity baseline

Add an explicit `tft_profile=canonical` preset with:

- pointwise per-variable embeddings;
- one static VSN followed by four context GRNs;
- correct `(c_h, c_c)` LSTM initialization;
- LSTM local processing;
- softmax VSN without graph mixing or residual bypass;
- interpretable shared-value attention;
- primary quantile output with a coherent median point forecast;
- all experimental branches off.

Use this as the reference in every extension ablation. Calling a fully augmented configuration “TFT” without the profile makes it difficult to tell whether a result comes from TFT or from the added graph/TCN/FFT/MoE stack.

## 5. Verified Correctness Findings

### TFT-C01 — Static covariates are erased before embedding

**Severity:** P0 for datasets using static features

**Implementation status on 2026-07-28:** fixed in the current worktree. Raw encoder inputs are now split before window normalization, declared static channels are validated to remain constant across encoder time, and static embeddings receive raw per-sample values in dataset coordinates under both manual normalization and RevIN.

**Evidence**

`forecast()` normalizes every encoder channel over time before `TFTEmbedding` extracts `static_pos`:

- normalization: [`models/TemporalFusionTransformer.py` lines 1214-1228](models/TemporalFusionTransformer.py#L1214);
- static extraction: [`models/TemporalFusionTransformer.py` lines 197-205](models/TemporalFusionTransformer.py#L197).

For a truly static channel `s_b`:

```text
(s_b - mean_t(s_b)) / std_t(s_b) = 0
```

This is true in both manual normalization and RevIN. With affine RevIN, the original entity value is still lost and replaced by a shared learned channel constant.

A local probe with static values `5` and `50` produced:

```text
normalized_static_values = 0
static_embeddings_equal = True
```

**Impact**

- entity/location/category information cannot influence the forecast;
- all four static context vectors are value-independent across samples, apart from training-time dropout;
- the feature appears supported by configuration but is semantically non-functional;
- current ETT tests do not expose this because registered ETT datasets use no static fields.

**Fix**

Split raw/static inputs before window normalization. Normalize dynamic observed channels only. Process:

- categorical static variables with entity embeddings;
- continuous static variables with a dataset-level transform or dedicated static normalizer;
- static values once per sample, not through a temporal convolution.

Add an invariance test where identical dynamic histories with two different static IDs yield different static contexts and can learn different targets.

### TFT-C02 — LSTM hidden and cell contexts are reversed

**Severity:** P0 for LSTM/hybrid models with static inputs

**Implementation status on 2026-07-28:** fixed in the current worktree. Native TFT recurrent initialization now passes static contexts in PyTorch’s required `(h_0, c_0)` order for both the pure-LSTM and hybrid temporal backbones, with identity-level recorder tests covering each path.

**Evidence**

PyTorch LSTM state order is `(h_0, c_0)`. The code defines static contexts in the order `c_s, c_c, c_h, c_e`, but passes:

```python
(c_c.unsqueeze(0), c_h.unsqueeze(0))
```

in both the LSTM and hybrid paths at [`models/TemporalFusionTransformer.py` lines 727-736](models/TemporalFusionTransformer.py#L727).

**Impact**

The cell-state context initializes the hidden state and the hidden-state context initializes the cell state. This does not create a shape error, so smoke tests pass while static conditioning has the wrong semantics.

**Fix**

Pass `(c_h.unsqueeze(0), c_c.unsqueeze(0))`. Add a test with a recording LSTM or controlled context tensors that asserts state identity, not only shape.

### TFT-C03 — Quantile training and evaluation use different heads

**Severity:** P0 when quantile mode is enabled

**Implementation status on 2026-07-28:** fixed in the current worktree. The native TFT path now uses explicit `point` / `quantile` / `joint` modes, one canonicalized quantile order shared by model and loss, an ordered non-crossing quantile head, positive RevIN scale, trained-median evaluation in quantile mode, and saved calibration artifacts (`pinball`, `coverage`, `interval_width`, `crossing_rate`) during `test()`.

**Evidence**

The decoder always computes a point projection. A separate quantile projection is created at [`models/TemporalFusionTransformer.py` lines 1152-1173](models/TemporalFusionTransformer.py#L1152) and populated through mutable `last_quantile_predictions`.

The production experiment selects either:

- MSE on the returned point tensor; or
- pinball loss on `last_quantile_predictions`;

at [`exp/exp_long_term_forecasting.py` lines 38-45](exp/exp_long_term_forecasting.py#L38) and [135-168](exp/exp_long_term_forecasting.py#L135).

This creates two broken cases:

1. `loss=MSE` + quantile head: quantile parameters are computed but not trained.
2. `loss=Quantile`: point-head parameters receive no gradient, yet `test()` evaluates the returned point tensor at [`exp/exp_long_term_forecasting.py` lines 235-253](exp/exp_long_term_forecasting.py#L235).

A focused backward probe confirmed `point_head.weight.grad is None` under quantile-only loss.

There is also ordering drift: the model sorts quantiles at [`models/TemporalFusionTransformer.py` lines 1153-1159](models/TemporalFusionTransformer.py#L1153), while `QuantileLoss` retains the user-provided order at [`utils/losses.py` lines 91-106](utils/losses.py#L91). An unsorted CLI list silently assigns different quantile levels to output slots and loss weights.

Duplicate quantile levels are not rejected, and crossing is not constrained.

Mutable `last_*` outputs are broken as a `DataParallel` contract. [`utils/tools.py` lines 127-138](utils/tools.py#L127) unwraps the base module after replica forwards and reads attributes that replicas do not reliably propagate. Quantile mode can therefore report missing predictions. A configured MoE auxiliary loss is even easier to miss silently because `combine_primary_and_aux_loss(..., None)` returns the primary loss unchanged.

**Fix**

Use a structured forecast result and a single canonicalized quantile list. Define one of:

- quantile-primary mode, with the median quantile used as point output;
- point-primary mode;
- explicit joint mode with a configured point/quantile coefficient.

For non-crossing forecasts, prefer an ordered parameterization such as a base quantile plus cumulative positive increments instead of merely sorting predictions after the fact. Preserve that order through de-normalization: RevIN's unconstrained affine weight is divided out at [`layers/StandardNorm.py` lines 56-63](layers/StandardNorm.py#L56), so a negative learned weight can reverse an ordered head. Parameterize the affine scale as positive or enforce and test ordering after de-normalization.

Return a tensor-only dictionary or `NamedTuple` that `DataParallel` can gather; a plain dataclass is not handled safely by its normal recursive gather. Define a sample-weighted/global reduction for MoE routing statistics. Tests must assert gradients for exactly the heads later evaluated; cover unsorted and duplicate configuration, parallel gathering, positive-scale/post-denormalization ordering; and report pinball loss, coverage, interval width, and crossing rate.

### TFT-C04 — `top_amplitude` FFT mode selection can crash

**Severity:** P0 for the affected FFT option

**Implementation status on 2026-07-28:** fixed in the current worktree. The FFT branch now selects top-amplitude bins per sample and latent channel, uses physical bin indices only for gather/scatter, binds learnable complex weights by selected-rank position, and preserves batch permutation/composition invariance.

**Evidence**

`SpectralBranch` allocates weights with shape `[d_model, modes]` at [`layers/TemporalFusion_layers.py` lines 296-309](layers/TemporalFusion_layers.py#L296). `top_amplitude` returns physical FFT-bin indices in `[0, n_freqs)` at [321-330](layers/TemporalFusion_layers.py#L321), then uses those absolute indices to index the `modes`-wide weight tensor at [360-365](layers/TemporalFusion_layers.py#L360).

When `n_freqs > modes` and a dominant selected bin is at or above `modes`, indexing fails. A high-frequency probe with `modes=2` reproduced an out-of-bounds `IndexError`.

Existing tests use settings where `modes` covers the available bins or random energy happens not to expose the boundary.

`top_amplitude` also averages amplitudes over the current batch and latent channels. A sample's selected bins—and therefore its forecast—can change depending on unrelated samples in the same batch. That violates batch-composition invariance at inference.

**Fix**

Choose and document one parameterization:

- weights by selected rank: use the first `k` weight slots for the `k` selected bins; or
- weights by physical bin: allocate/derive weights for the maximum supported FFT grid and validate sequence lengths.

Then remove batch dependence by selecting bins per sample/per channel, or use fixed bins learned from training data/configuration. Add deterministic sine-wave tests whose dominant bins are below, equal to, and above `modes`, plus batch permutation and batch-composition invariance tests.

### TFT-C05 — Higher-order interaction gates have incompatible cardinality

**Severity:** P0 for order 3; P1 for order 2

**Implementation status on 2026-07-28:** fixed in the current worktree. The higher-order block now emits one independent sigmoid gate per actual interaction term, so order 2 controls the pair term directly, order 3 controls pair and triple terms separately, and interpretation payloads report gate shape `[B, T, interaction_order - 1]`.

**Evidence**

At [`layers/TemporalFusion_layers.py` lines 602-647](layers/TemporalFusion_layers.py#L602):

- order 2 creates one interaction term (`pair_term`) but two softmax gates;
- order 3 creates two terms (`pair_term`, `triple_term`) but three gates.

For order 2, broadcasting applies both gates to the same pair term, and the softmax gates sum to one. The result is the pair term regardless of gate logits. A probe found only numerical-noise gate gradients.

For order 3, the gate and term dimensions are `3` and `2`, causing a runtime size mismatch.

**Fix**

Make the number of gates equal the number of terms, or explicitly include a “no interaction” branch. A robust design is an independent sigmoid gate per optional term, which can suppress both pair and triple interactions. Add forward/backward tests for both orders and assert non-zero gate gradients on a non-degenerate loss.

### TFT-C06 — Static interpretation weights are dropped

**Severity:** P1

**Implementation status on 2026-07-28:** fixed in the current worktree. Static interpretation outputs now preserve per-context VSN weights and graph-attention metadata instead of collapsing to `None`, and interpretation summaries/export now carry static feature names from the resolved TFT schema.

**Evidence**

`StaticCovariateEncoder` returns:

```python
{"c_s": ..., "c_c": ..., "c_h": ..., "c_e": ...}
```

at [`models/TemporalFusionTransformer.py` lines 442-451](models/TemporalFusionTransformer.py#L442). `Model._split_vsn_weight_payload()` only reads top-level `selection` and `graph_attention` keys at [1208-1212](models/TemporalFusionTransformer.py#L1208). Therefore the nested static payload is converted to `(None, None)`.

A full static-input interpretation probe reproduced `static_vsn_weights=None`.

**Fix**

Preserve the per-context structure, or return one canonical static VSN payload after restoring the paper's single-static-VSN design. Add a full-model static interpretation test.

### TFT-C07 — Advertised short-term forecasting fails its experiment contract

**Severity:** P0 for `short_term_forecast`

**Evidence**

Normal M4 construction fails first: `data='m4'` is absent from `datatype_dict`, so `get_typepos()` requires an explicit `tft_observed_pos` that the normal short-term path does not supply. For a manually configured model that gets past construction, `Model.forward()` explicitly accepts both long- and short-term task names at [`models/TemporalFusionTransformer.py` lines 1443-1462](models/TemporalFusionTransformer.py#L1443). The short-term experiment calls the model with `x_mark_enc=None` and `x_mark_dec=None` at [`exp/exp_short_term_forecasting.py` lines 87-95](exp/exp_short_term_forecasting.py#L87), while `_validate_inputs()` immediately dereferences `.ndim` on all four inputs at [`models/TemporalFusionTransformer.py` lines 1176-1178](models/TemporalFusionTransformer.py#L1176).

Even if markless input were accepted, validation/forecast assignment in the short-term experiment expects prediction-length output, while native TFT returns zero history plus prediction.

**Fix**

Either implement a markless known-input strategy and prediction-only output contract for short-term forecasting, or reject the task during model construction with a clear supported-task error. Add an experiment-level test rather than only a direct model smoke test.

### TFT-C08 — Known-input and feature schema validation is incomplete

**Severity:** P1

`_validate_inputs()` checks the encoder value length but never checks `x_mark_enc.shape[1] == seq_len`. A short mark tensor can cause an early future mark to be sliced into the history while the decoder still returns the expected shape. A long mark tensor can insert extra tokens. This is a semantic alignment failure, not necessarily a shape failure.

Other schema gaps:

- registered dataset entries override explicit static/observed positions;
- registered and custom role indices are not checked or resolved against the effective post-`M`/`MS`/`S` feature schema; in particular, ETT's registered `0..6` observed positions index out of range with `features='S'` and `enc_in=1`;
- static and observed roles are not checked for invalid overlap;
- targets are not required or warned to be historically observed;
- `run.py` exposes `--tft_allow_custom_known` but not the required `tft_known_len` or optional maximum-channel setting;
- runner help allows detailed frequencies such as `15min` and `3h`, while `get_known_len()` uses exact short-code dictionary lookup.

**Fix**

Introduce a typed feature schema resolved once at construction. Validate names, indices, role overlap, target observability, exact mark lengths, known-input names, frequency aliases, and source availability before the first batch.

### TFT-C09 — The production experiment ignores the model target map

**Severity:** P0 for non-last MS targets and reduced/non-contiguous multi-output mappings

**Evidence**

`get_target_pos()` and model de-normalization honor `tft_target_pos`, but the production experiment still selects truth with the repository-wide `f_dim` convention in train, validation, and test at [`exp/exp_long_term_forecasting.py` lines 55-73, 122-168, and 222-253](exp/exp_long_term_forecasting.py#L55). MS always takes the last truth channel, regardless of `tft_target_pos`; reduced M outputs can be compared against incompatible channels or shapes.

The test inverse path also tiles `C_out` predictions to the full input width before applying the dataset scaler. This can produce numerically wrong inverse metrics even when the model's internal target de-normalization was correct.

**Fix**

Use one schema-aware target-index helper for train, validation, test, quantile loss, inverse metrics, and physics operands. Select truth by resolved target indices and inverse-transform each predicted target with its corresponding training-scaler mean/scale—never by tiling. Add non-last MS and non-contiguous multi-output M tests.

## 6. Architecture, Performance, and Interpretability Findings

### TFT-A01 — The “learned” FFT selector does not select frequencies

**Implementation status on 2026-07-28:** fixed in the current worktree. The learned FFT path now allocates spectral-anchor logits with shape `[1, d_model, modes]`, interpolates logits and complex weights onto the runtime FFT grid, and exports learned-mask summaries through the interpretation payload.

At the audited base commit, `freq_mask_logits` had shape `[1, d_model, 1]` and was expanded across every frequency at [`layers/TemporalFusion_layers.py` lines 310-312](layers/TemporalFusion_layers.py#L310) and [343-355](layers/TemporalFusion_layers.py#L343). It therefore learned one scalar amplitude gate per latent channel, identical for all FFT bins.

The repair keeps `low`, `top_amplitude`, and `learned` modes separate. `learned` now uses a length-independent anchor parameterization over `modes`, then linearly interpolates both mask logits and complex weights to the active `rfft` grid. That makes the option genuinely frequency-selective while preserving checkpoint portability across sequence lengths.

Regression coverage now includes:

- distinct learned logits producing different bin weights;
- runtime-length interpolation behavior;
- finite gradients through learned mask logits and complex weights;
- state-dict round-trip preservation of learned mask behavior;
- interpretation payload summaries for learned-mask mean, spread, and peak-bin location.

The FFT branch also processes the entire history-plus-known-future sequence at once. That is valid if all future inputs are truly known, but it means later known-future information can influence earlier latent positions before the causal attention mask. Temporal attention from this mode is not a complete causal explanation.

### TFT-A02 — Large parameter sets are structurally unused

Three cases are visible:

1. `StaticCovariateEncoder` always creates four GRNs even when `static_len == 0` at [`models/TemporalFusionTransformer.py` lines 419-440](models/TemporalFusionTransformer.py#L419). They can never execute.
2. `VariableSelectionNetwork` always allocates the residual projection and gate even when residual bypass is disabled at [321-324](models/TemporalFusionTransformer.py#L321).
3. When sigmoid gating is enabled, the early return at [363-379](models/TemporalFusionTransformer.py#L363) bypasses `head_grns`, `variable_grns`, and the low-rank selection path.

In a representative local probe:

```text
d_model=16:  4,480 dead static parameters, about 9.9% of model parameters
d_model=128: 265,216 dead static parameters, about 9.9%
sigmoid VSN, d=48/C=28: about 77% of VSN parameters structurally unused
```

Instantiate modules only for the selected execution path. Add a test that every expected trainable parameter receives a gradient for each named profile.

### TFT-A03 — Per-variable embedding is expensive and non-canonical

Every observed/static scalar owns a full `DataEmbedding`, which contains:

- a circular kernel-3 temporal convolution;
- a length-5000 positional buffer;
- temporal embedding modules that are unused when `x_mark=None`.

See [`models/TemporalFusionTransformer.py` lines 176-205](models/TemporalFusionTransformer.py#L176) and [`layers/Embed.py` lines 29-42 and 109-126](layers/Embed.py#L29).

Consequences:

- a variable representation is not pointwise as in the TFT paper;
- circular padding makes the first history embedding use the last history value;
- duplicated positional buffers scale as `O(C * max_len * d_model)` and are included in state dictionaries;
- feature-heavy models become unnecessarily large.

For ETT's seven observed variables, duplicated positional buffers alone were approximately:

```text
d_model=16:   2.24 MB
d_model=128: 17.92 MB
```

Use feature-specific pointwise linear/categorical embeddings and one shared positional representation only where the architecture needs it.

### TFT-A04 — Temporal graph evolution is not sparse

**Implementation status on 2026-07-28:** fixed in the current worktree. Temporal graph evolution now perturbs base logits rather than normalized probabilities, preserves the original structural support mask, re-masks removed edges to negative infinity before the final softmax, and uses a low-rank temporal state instead of a dense `C²` hidden state.

At the audited base commit, the base learner created top-k adjacency at [`layers/AdvancedDynamicGraph.py` lines 32-55](layers/AdvancedDynamicGraph.py#L32). Temporal evolution then added a dense perturbation and applied a full softmax at [133-153](layers/AdvancedDynamicGraph.py#L133). Every finite edge became positive, including edges removed by top-k.

Even a zero perturbation would apply `softmax()` to an already normalized probability matrix and make it dense. In addition, `alpha` was initialized to `0.1` but used through `sigmoid(alpha)`, giving an initial multiplier near `0.525`, not `0.1`.

The repair now defines `top_k=0` as truly dense, keeps sparse support fixed during temporal evolution, and initializes the effective evolution strength to `0.1` through a logit parameterization. The temporal evolution path now uses low-rank source/destination factors derived from a compact recurrent state, which removes the worst `O(C⁴)` recurrent parameter growth from the previous `C²` GRU design.

Edge-feature mode still materializes dense `[N,C,C,d]` tensors when enabled, but the module now raises early when that tensor would exceed a conservative size guard. Returned “multi-head” adjacency remains one learned graph broadcast across heads; the implementation is now at least honest about preserving one shared support rather than silently densifying it.

Regression coverage now includes:

- temporal sparse graphs staying within top-k support;
- `top_k=0` behaving as dense rather than all-zero;
- finite row-normalized temporal adjacency;
- initial effective evolution strength equal to `0.1`;
- temporal variation without support expansion;
- temporal-evolution parameter-growth regression.

### TFT-A05 — The MoE is sparse in routing only

**Implementation status on 2026-07-28:** fixed in the current worktree at the contract level. The module is now explicitly treated as dense-compute top-k mixing, capacity uses `ceil()` with a minimum of one routed token when tokens exist, zero-route tokens are restored to their strongest fallback expert, and the auxiliary loss now includes both importance and load-balancing terms with global-reducible summaries.

`RegimeAwareSparseMoE` still evaluates every expert through fused einsums at [`layers/TemporalFusion_layers.py` lines 770-777](layers/TemporalFusion_layers.py#L770). Top-k routing reduces mixing, not expert compute. The implementation is now honest about that behavior through the exported routing-mode label.

At the audited base commit, the capacity path at [729-748](layers/TemporalFusion_layers.py#L729):

- loops over experts in Python;
- calls `.item()`, synchronizing accelerator execution;
- can compute capacity `0` for small token counts;
- can drop every route for a token, producing a zero mixture with no fallback.

A `B=1, T=1, E=4, top_k=2` probe produced `capacity=0`, routing sum `0`, auxiliary loss `0`, and zero expert gradient. The repair now prevents that failure mode and propagates both expert-importance sums and expert-load sums through the structured output so experiment-side auxiliary-loss reduction remains replica-safe.

Regression coverage now includes:

- `B=1, T=1` retaining a non-zero route;
- every token retaining at least one route after capacity pruning;
- heavy expert-imbalance/capacity stress cases;
- global auxiliary-loss reduction using importance and load statistics;
- selected experts receiving gradients.

### TFT-A06 — Extension modes weaken literal interpretability

Interpretation payloads remain useful diagnostics, but they are not always faithful attributions:

- graph mixing changes each node representation before VSN selection;
- residual VSN bypass allows predictions to avoid the selected convex combination;
- sigmoid gates are independent and not unit-sum;
- full attention is not the shared-value interpretable attention from TFT;
- dual attention blends an interpretable and non-interpretable branch;
- FFT globally mixes time before causal attention;
- higher-order interactions occur after VSN collapse, so they are latent-coordinate interactions, not explicit original-covariate interactions;
- covariate reattention uses detached pre-VSN embeddings and padded zero tokens without a mask; after flattening there is no explicit 2-D time-by-covariate coordinate, identity is only partial/implicit, and padded tokens have neither identity nor a padding mask at [`models/TemporalFusionTransformer.py` lines 1246-1257](models/TemporalFusionTransformer.py#L1246) and [748-755](models/TemporalFusionTransformer.py#L748);
- MoE regime/routing tensors are detached in the interpretation payload.

This is now repaired in the current worktree: interpretation exports use named history/future feature summaries, explicit axis labels, and profile-aware caveat flags rather than flat VSN indices.

Expose interpretation-quality metadata such as:

```text
is_canonical_vsn_attribution
uses_graph_pre_mixing
uses_vsn_bypass
uses_noninterpretable_attention_branch
uses_global_spectral_mixing
```

Do not present canonical TFT variable importance or temporal attention claims when those assumptions are false.

### TFT-A07 — Lag attention needs physical and masking semantics

`MultiScaleLagAttention` shifts the sequence with zero padding, but does not mask padded keys at [`layers/TemporalFusion_layers.py` lines 490-520](layers/TemporalFusion_layers.py#L490). It then attends over all causal positions in the shifted sequence, rather than selecting only an exact lag.

When RoPE or ALiBi is enabled, shifted keys are not given positions offset by the physical lag. When `lag >= sequence_length`, the branch is entirely zero yet still receives a learned fusion weight.

This is now repaired in the current worktree: lag branches validate active length, mask shifted padding, label themselves honestly as shifted-history attention, and pass physical lagged key positions. Temporal compression also preserves explicit original coordinates for downstream RoPE/ALiBi use.

### TFT-A08 — Runtime safety checks can dominate accelerator execution

The model performs repeated `torch.isfinite(...).all()`, `.any()`, and warning checks throughout embeddings, temporal blocks, attention, graphs, and the top-level input validator. These reductions can synchronize GPU/ROCm execution each forward.

This is now repaired in the current worktree: expensive repeated finite-value checks are debug-gated, cheap structural validation stays on, and import-time hardware environment mutation has been removed from the model module.

### TFT-A09 — Configuration and benchmark semantics drift

- `d_ff` is printed and varied throughout `tests/test_tft_deep_ett.py`, but neither native TFT source file reads it. Those “d_ff” ablations do nothing.
- `tft_temporal_backbone_layers` controls TCN depth, not the plain LSTM path.
- direct model defaults and `run.py` defaults differ for important flags such as full attention, residual bypass, and interpretation payload stacking.
- `run.py` experiment identifiers omit TFT extension flags, so materially different models can share a checkpoint/result path.
- `tests/test_tft_deep_ett.py` references undefined `use_amp` in its final test loop at line 574.
- its advertised “2-epoch” physics verification would actually use `baseline_safe`'s 150 epochs; the parser has no epoch override.
- `run_tests.py` discovers `Tests`, while the repository directory is lowercase `tests`.
- the main native comprehensive test class is skipped when optional Nixtla TFT import fails, coupling native coverage to an out-of-scope implementation.

This is now repaired in the current worktree: native TFT defaults are centralized, `d_ff` is explicitly treated as ignored/non-material, backbone-layer scope is surfaced honestly, and the digest now tracks the resolved material semantics.

### TFT-A10 — Attention dropout configuration is incomplete

Full attention uses `dropout_p=0.0` in the SDPA call, and the exact path does not drop attention probabilities. The configured dropout is applied only after output projection. This can be a valid design choice, but it does not match the expectation of an attention-dropout argument and reduces regularization relative to common TFT implementations.

This is now repaired in the current worktree: attention-probability dropout is explicit, exact/SDPA paths match in evaluation, training-only dropout is applied correctly, and interpretation tensors remain pre-dropout.

## 7. What Is Already Correct or Improved

The prior report overstated several bugs that are not open in the audited commit:

| Previous claim | Current status |
|---|---|
| Non-persistent causal-mask buffers stay on CPU | False in general—registered non-persistent buffers migrate with the module—and current forwards explicitly move device/dtype |
| `x_dec` must equal `c_out` | Fixed: validation accepts `c_out` or `enc_in` at model lines 1190-1191 |
| Stochastic-depth split remains stale | Fixed at model lines 1022-1032 |
| MoE flat reshape does not propagate | Current code makes the tensor contiguous and reassigns the reshaped result |
| Pair/triple normalization is a bug | Not a bug; the actual higher-order defect is gate/term cardinality |
| `minute_x=0.` causes stack failure | The non-minute branch excludes it from the stack |

Other sound pieces include:

- explicit model-local `tft_target_pos` mapping with an MS fallback;
- cached `target_pos_buf`;
- target-aware internal de-normalization, including the RevIN scatter/select approach;
- causal attention masks with a dynamic fallback for longer sequences;
- exact/SDPA selection with exact fallback when attention weights are requested;
- partial input rank/length/known-feature validation;
- structured interpretation data for many optional branches;
- gradient checkpointing and stochastic-depth controls;
- a broad native TFT test suite.

These are worth preserving while simplifying the architecture around them.

## 8. Audit of the Original Physics-Informed Plan

### Overall assessment

The proposal can become a useful physics-/theory-guided loss framework. As written, it would not yet constitute a reliable implementation and should not be described as a classical PINN: it contains no collocation-domain residual construction, no coordinate differentiation, and no governing equation for ETTh1.

| Proposed item | Verdict | Required change |
|---|---|---|
| Generic equation residual | Feasible with constraints | Named sources, safe schema, physical scaling, masks, tolerances, future-availability rules |
| Monotonicity pairs | Underspecified | Paired counterfactual or Jacobian definition with perturbation/range/horizon |
| Rate bound | Closest to implementable | Physical `dt`, target mapping, last-history boundary, raw-unit conversion |
| L1 on VSN weights | Incorrect for softmax VSN | Entropy/KL/prior mass; L1 only for sigmoid gates |
| Integrate in model and deep test | Wrong production layer | Output losses in the experiment; model changes only for internal diagnostics |
| Two-epoch ETTh1 verification | Not what command does | Add epoch/batch override; fix undefined `use_amp`; use synthetic-law tests first |

### 8.1 Missing units and scaler contract

Physics equations must run in a declared coordinate system. The repository's sklearn/NumPy inverse transforms are used only after inference and are not differentiable. The implementation needs Torch buffers for training-split mean/scale, selected through `target_pos`.

Residuals with different units must be normalized, for example:

```text
L_k = mean( rho(residual_k / tolerance_k) )
```

Without this, a term measured in watts can dominate a term measured in degrees solely because of magnitude.

### 8.2 Missing data-source contract

The plan's coefficient/index lists do not identify whether an operand is:

- a predicted state;
- historical observation;
- true known-future driver;
- future training label;
- static variable.

The distinction is essential for deployability and leakage prevention. Standard TFT `x_mark` is calendar/time-feature data, not arbitrary future physical state. `x_dec` values are ignored by the model.

### 8.3 Rate loss boundary

The correct first finite difference is:

```text
(forecast[:, 0] - history_target[:, -1]) / dt
```

followed by differences between forecast steps. Applying `diff()` to the model's full returned tensor would compare a zero history pad with the first forecast and create a false violation.

### 8.4 Monotonicity is not a temporal-difference rule

If the intended statement is “increasing input `x_i`, all else equal, must not decrease output `y_j`,” use a paired perturbed forward or input derivative. A time-series co-movement penalty does not isolate that effect and can encode spurious correlation.

### 8.5 VSN priors need a lightweight differentiable output

Normal forward does not return VSN weights. Full interpretation mode also materializes attention and can disable the fast attention path. Add a selective `return_auxiliary` contract that can return differentiable VSN logits/weights without unrelated payloads.

Graph pre-mixing and residual bypass must be considered: a perfect prior on reported VSN weights does not constrain information that bypasses or has already mixed across variables.

### 8.6 ETTh1 is not a physics validation dataset by default

The repository supplies ETT feature names and sampling frequency, but not a governing equation, topology, rated capacities, units, or an energy-balance source. Rate or smoothness hypotheses can be tested as theory-guided regularizers. They are not evidence of physics consistency without an externally justified law.

## 9. Improved Physics/Theory-Guided Implementation Plan

The full actionable plan is in [`implementation_plan.md`](implementation_plan.md). The recommended sequence is:

### Phase 0 — Stabilize native TFT prerequisites

Fix and test:

1. static value preservation;
2. `(c_h, c_c)` state order;
3. quantile training/evaluation contract and ordering;
4. FFT top-amplitude indexing and learned selector;
5. higher-order interaction gating;
6. production use of `tft_target_pos`;
7. DataParallel-safe quantile/MoE outputs;
8. deep benchmark `use_amp` and epoch override.

Physics experiments should not be built on output modes whose trained/evaluated contracts are already inconsistent.

### Phase 1 — Add domain metadata and strict schema

Add:

- ordered feature names after the effective `M`/`MS`/`S` loader selection and column reordering;
- generated known-feature names and measured sample interval;
- full observed scaler plus explicit transforms for known/static namespaces;
- units from a versioned physics spec or sidecar, never guessed from CSV columns;
- source namespaces;
- versioned JSON constraint specifications;
- complete inequality/bound/polynomial, temporal-selector, mask, quantile, reduction, and feasibility semantics;
- startup resolution from names to indices.

Reject missing/unavailable operands, incompatible time axes, irregular cadence, and missing units before training. History operands require explicit `last`, `lag:k`, or reduction semantics; static broadcasting must also be declared. Use an allow-listed unit registry only during configuration resolution and precompute Torch affine transforms for the batch loop. Never evaluate free-form Python expressions from configuration.

### Phase 2 — Implement output-level constraints

Create:

- `utils/physics_config.py`
- `utils/physics_losses.py`

Initial constraints:

- linear equality and inequality;
- explicit polynomial monomials;
- physical value bounds;
- absolute rate bounds including the history boundary.

Pass the complete observed history, horizon-aligned known-future inputs and labels, and static values through qualified namespaces. Do not reduce history to target-only channels before resolving operands.

Return:

```python
PhysicsLossResult(
    total=...,
    terms={...},
    violation_rates={...},
    valid_counts={...},
    violation_counts={...},
)
```

Do not put this module under `layers/` and do not import it from the model.

### Phase 3 — Integrate the production experiment

Modify `exp/exp_long_term_forecasting.py` so AMP and non-AMP branches share one composition helper:

```text
total = primary + moe_coefficient * moe_aux
                  + scheduled_physics_weight * physics.total
```

The same resolved target-index helper must select labels for primary/quantile loss, physical operands, inverse transforms, and final metrics. Auxiliary model outputs must be tensor-only, gatherable values rather than mutable replica attributes.

Log primary task loss, the legacy validation objective, total loss, every constraint term, and violation rate separately. Preserve the current primary-plus-MoE validation objective and epoch-based LR schedule by default; allow explicit primary, fixed-weight-total, or feasibility-aware checkpoint policies.

Hash the raw config and relevant flags when constructing the run setting, then persist the fully resolved spec/mappings/scalers as a manifest after the dataset is available. In standalone test mode, load and verify that manifest or resolve only from metadata whose scaler was fitted on the training range. Add held-out physical-unit compliance metrics to `test()`.

### Phase 4 — Verify with a known synthetic law

Before ETTh1:

- exact satisfying examples must yield zero residual;
- violations must yield positive loss and finite non-zero gradients;
- standardized and raw-space computations must agree after inverse scaling;
- a synthetic conservation/dynamics dataset must reduce held-out violation versus a matched baseline;
- `physics_weight=0` must skip the training loss path and preserve baseline outputs and parameter updates in deterministic CPU tests; post-hoc compliance evaluation remains allowed.

### Phase 5 — Add VSN theory alignment

- softmax path: entropy minimization, KL to a theory prior, or forbidden-mass penalty;
- sigmoid path: L1 or target-cardinality penalty on pre-dropout gates/logits;
- use a selective differentiable auxiliary output;
- define history/future variable namespaces;
- report bypass and graph-mixing caveats.

The correct entropy sign for sparsity is:

```text
L_sparse = +lambda * H(w)
H(w) = -sum_i w_i log(w_i)
```

Minimizing positive entropy encourages concentration. The old report's subtraction would maximize entropy and encourage a dense/uniform distribution.

### Phase 6 — Add counterfactual monotonicity

Perturb a named covariate in physical units, transform it back to model coordinates, re-run the forecast, and penalize directional violations. Specify input range, perturbation, timesteps, horizons, and targets. Use identical stochastic behavior for the paired forwards, and reject whole-window shifts that normalization cancels. Treat Jacobian-based constraints as an advanced mode requiring second-order-gradient tests.

### Phase 7 — Consider hard physics architectures

Only after a validated domain law exists:

- project outputs onto an exact linear conservation manifold;
- predict a correction to a physics baseline;
- couple to a differentiable simulator;
- use augmented Lagrangian/adaptive multipliers if fixed weights cannot reach feasibility.

## 10. Recommended Native TFT Upgrade Roadmap

### Release blocker set

Status on Tuesday, July 28, 2026: this set is implemented in the current worktree except for the release-gate verification task `TFT-T01`.

1. Keep `TFT-C01` through `TFT-C09`, `TFT-H01`, and `TFT-O01` treated as closed unless a new regression is proven.
2. Finish `TFT-T01` with the exact recorded native release command.
3. Pass `G1` only after that command is green and logged.
4. Do not reopen already-fixed blockers opportunistically while working on post-`G1` upgrades.

### Canonicalization set

Recommended order after `G1`:

1. `TFT-P01`: add `canonical`, `extended_safe`, and `experimental_full` profiles.
2. `TFT-A03`: replace observed/static `DataEmbedding` modules with typed pointwise variable embeddings and restore a canonical static design.
3. `TFT-A02`: stop instantiating structurally dead branches under the canonical profile.
4. `TFT-E01`: benchmark the canonical profile so later extension claims have a trustworthy baseline.
5. `TFT-A09` is now complete in the current worktree.
6. Preserve the current tensor-only structured output contract as the long-term API baseline.

### Extension-hardening set

Recommended order after the canonicalization set:

1. `TFT-A01` is now complete in the current worktree.
2. `TFT-A04`: preserve temporal graph sparsity and replace `C²` recurrent adjacency evolution with something honest and scalable.
3. `TFT-A05`: either implement real sparse-dispatch MoE behavior or rename/document it honestly as dense-compute top-k mixing.
4. `TFT-A07` is now complete in the current worktree.
5. `TFT-A08` is now complete in the current worktree.
6. `TFT-A10` is now complete in the current worktree.
7. `TFT-A06` is now complete in the current worktree.

### Evaluation set

Every extension should be compared against the canonical profile with:

- identical seed, initialization policy, data split, optimizer, and parameter budget;
- forecast MSE/MAE or quantile risk;
- calibration/coverage for probabilistic output;
- latency, peak memory, parameter count, and checkpoint size;
- interpretation stability across seeds;
- constraint violation in physical units when physics mode is active.

Do not enable all extensions simultaneously and infer individual value from the aggregate result.

## 11. Test Additions

### Native TFT regression tests

- `test_static_values_survive_normalization`
- `test_static_context_changes_with_entity_value`
- `test_lstm_receives_hidden_then_cell_context`
- `test_static_interpretation_payload_preserved`
- `test_encoder_mark_length_must_equal_sequence_length`
- `test_registered_ett_single_feature_schema`
- `test_short_term_contract_or_explicit_rejection`
- `test_nonlast_ms_target_mapping_end_to_end`
- `test_noncontiguous_multioutput_target_mapping`
- `test_quantile_only_evaluates_trained_output`
- `test_mse_mode_does_not_expose_untrained_quantiles`
- `test_unsorted_quantiles_are_canonicalized_once`
- `test_quantile_outputs_do_not_cross`
- `test_quantile_order_survives_revin_denormalization`
- `test_parallel_output_gathers_quantiles_and_moe`
- `test_fft_top_amplitude_high_bin`
- `test_fft_selection_is_batch_composition_invariant`
- `test_fft_learned_mask_varies_by_frequency`
- `test_higher_order_two_has_gate_gradient`
- `test_higher_order_three_forward_backward`
- `test_temporal_sparse_graph_stays_top_k`
- `test_moe_small_batch_has_nonzero_route`
- `test_selected_profile_has_no_unexpected_dead_parameters`
- `test_cli_and_model_defaults_match`

### Physics tests

- satisfying/violating equations and gradients;
- physical-unit inverse-scaling equivalence;
- target-name and `target_pos` resolution;
- rate boundary and `dt`;
- masks and missing observations;
- temporal alignment and namespace-specific transforms;
- unit compatibility, including affine/offset conversions;
- unavailable future source rejection;
- softmax-L1 configuration rejection;
- primary + MoE + physics composition;
- quantile policy;
- feasibility-aware checkpoint semantics;
- zero-weight parity;
- held-out physical-unit compliance persistence;
- synthetic-law compliance improvement.

## 12. Final Priority Matrix

| Order | Work item | Reason |
|---:|---|---|
| 1 | Repair static normalization and LSTM state order | Core advertised TFT input class is otherwise unusable |
| 2 | Repair quantile output/loss/evaluation contract | Current probabilistic mode can score an untrained head |
| 3 | Make production losses/metrics honor `tft_target_pos` | Model-local mapping is currently undone by experiment slicing |
| 4 | Fix FFT top-amplitude and higher-order interaction failures | Deterministic optional-mode crashes |
| 5 | Close short-term, schema, static-interpretation, and parallel-output contracts | Advertised APIs otherwise fail or silently misalign inputs |
| 6 | Add semantic regression tests | Existing suite misses the failures above |
| 7 | Add canonical TFT profile and typed embeddings | Restores a trustworthy reference and reduces bloat |
| 8 | Add domain metadata and differentiable physical scaling | Prerequisite for meaningful physics loss |
| 9 | Implement output-level physics constraints in experiment layer | Lowest-risk useful physics milestone |
| 10 | Validate on synthetic governing laws | Establishes correctness before real-data claims |
| 11 | Add VSN priors through selective auxiliary outputs | Current L1 proposal is ineffective |
| 12 | Add counterfactual monotonicity | Higher compute and more semantic choices |
| 13 | Harden graph/MoE/lag/FFT extensions | Important, but separable from the canonical path |
| 14 | Explore hard projection/simulator coupling | Requires validated domain equations |

## 13. Domain Analysis: NIFTY 50 with Planetary Known-Future Covariates

This domain was opened explicitly on 2026-07-29. Its canonical cross-session
project is
[`projects/financial_astrology_tft/README.md`](projects/financial_astrology_tft/README.md);
the original detailed audit is
[`Vedic_Astrology_TFT_Implementation_Plan.md`](Vedic_Astrology_TFT_Implementation_Plan.md).

### 13.1 Scientific classification

PySwissEph ephemerides are deterministic, inference-available future covariates. The orbital physics has already been used to generate the input trajectory. It does not provide a governing equation from planetary state to NIFTY returns.

Therefore:

- the correct first description is **physics-respecting exogenous-covariate modeling** or **theory-guided planetary interaction modeling**;
- a classical PINN residual on market outputs is not justified;
- orbital consistency checks belong in data validation;
- planet-to-market duration/aspect priors must be tested as explicit ablations against matched null trajectories;
- any positive finding establishes out-of-sample predictive association for the tested representation, not causality.

### 13.2 Confirmed production blocker

The model-side custom-known path exists, but the production loader does not populate it:

1. `TFTCustomKnownEmbedding` accepts arbitrary known channels.
2. The model concatenates `x_mark_enc` and the future portion of `x_mark_dec`.
3. `Dataset_Custom` constructs both mark tensors only from timestamp/calendar fields.
4. All other CSV columns go into `data_x/data_y`.

Consequently, adding planetary columns to the CSV or enabling `--tft_allow_custom_known` does not create a valid end-to-end planetary-known-future run. A dedicated loader must split:

```text
historical market -> x_enc
market targets     -> batch_y
calendar + planets -> x_mark_enc and x_mark_dec
```

Custom-known mode also replaces rather than automatically augments the standard calendar embedding. The new known block must preserve a stable named calendar control alongside planetary fields.

### 13.3 Additional current implementation findings

- Dataset feature names are discovered after model construction, so custom observed names currently fall back to anonymous `f0`, `f1`, and so on unless the lifecycle is changed.
- Known radius/velocity features have no production train-only scaling contract.
- Plain Adam, MSE/Quantile-only loss selection, no gradient clipping, and no warmup are weak defaults for a noisy approximately 7–8k-session dataset.
- The production training loop evaluates the test set after every epoch; a confirmatory financial study must remove that feedback and use validation only.
- Disabling RevIN does not disable native TFT's manual per-window mean/std normalization. Return targets require an explicit normalization-mode ablation.
- `label_len` does not influence native TFT computation because decoder market values are not consumed.
- `d_ff` remains ignored.
- A direct reproduction found `tft_vsn_per_feature_gating=True` crashes because `variable_grns` is set to `None` and then passed to `len(...)`. This is now tracked as `AST-C00`.

### 13.4 Target and evaluation recommendation

The confirmatory first target should be next-trading-day close-to-close log return, not raw OHLC level. Recommended causal historical inputs are close return, overnight gap, intraday body, and log high/low range.

The primary null is:

> Conditional on market history and ordinary calendar/Fourier controls, real planetary covariates do not reduce paired future-date forecast loss beyond matched smooth null ephemerides.

Use expanding walk-forward development folds, multiple fixed seeds, a locked final temporal holdout, paired per-date losses, block-bootstrap/HAC-aware uncertainty, and multiplicity correction. Coherent date shifts, spectrum-preserving surrogates, and smooth pseudo-planets are required controls; independent row shuffling is too weak because it destroys ephemeris autocorrelation.

### 13.5 Architecture recommendation

Start with a small point-forecast `extended_safe`/LSTM TFT, `pred_len=1`,
`seq_len=252` trading sessions for the **local market branch**, all advanced
extensions off, and a parameter budget near or below 50k. Compare 64, 128, 252,
and 504 only inside development folds. First test a compact raw
longitude/velocity block through the corrected production known-future path.

The local market sequence is not the planetary memory. Use exact target-date
circular/rashi/nakshatra/retrograde state, a fast calendar-daily event grid, a
medium weekly/event grid, slow monthly or recursive calendar-time state, and
current/relative phase for outer planets. This represents long cycles without
feeding thirty years of daily rows through one LSTM.

Only after that raw incremental-value test should the architecture add:

1. grouped `[batch,time,planet,field]` tokens with a shared body encoder;
2. explicit relative-angle/aspect edge features before VSN collapse;
3. an 8–16 dimensional pooled planetary state;
4. calendar-time-aware fast/intermediate/slow response kernels, including
   preregistered classical fruition-delay centers;
5. a separately measurable, near-zero-initialized planetary residual branch.

The current higher-order block is post-VSN latent interaction, not an explicit Mercury–Moon or Jupiter–Saturn interaction. Generic lag attention counts sequence rows rather than elapsed calendar days. FFT, MoE, covariate reattention, and the full experimental profile should remain off in the first small-data experiment.

Slow-planet claims have an irreducible identification limit: NIFTY data from
1995 contains roughly one Saturn orbit, only a few Jupiter orbits, and fractions
of the Uranus, Neptune, and Pluto orbits. No sequence length or network size can
create independent cycles. The classical Navagraha profile and modern
outer-planet profile must remain separate. Longer history or preregistered
validation across other markets is required for stronger slow-cycle conclusions.

### 13.6 Supplied-data audit result and required remediation

The wide CSVs have now been inspected. Their continuous longitude/motion fields
are internally coherent, but all supplied rashi pairs implement an exact
off-by-one/clipping transform, the return file mixes session-date conventions,
Shadbala lacks reproducible provenance, and no Hilbert fields are identifiable.
The repository contains no PySwissEph generator or convention manifest. The
full evidence and admitted 37-column provisional slice are in
[`projects/financial_astrology_tft/DATA_AUDIT_REPORT.md`](projects/financial_astrology_tft/DATA_AUDIT_REPORT.md).

Before any loader or architecture patch, obtain authoritative raw NIFTY
OHLC/session keys plus the PySwissEph generator/settings, reproduce selected
rows, and resolve market/ephemeris timestamps. Shadbala remains quarantined;
Hilbert code is needed only if the user intended a separate absent feature arm.

## 14. Post-Matrix Semantic Audit — 2026-07-31

The July ETTh1 feature matrix adds a second kind of evidence: trained behavior,
checkpoint state, parameter gradients, and ablation comparability. It does not
invalidate the historical repairs recorded for `TFT-A01`–`TFT-A10`; it shows
that those structural/safety contracts were narrower than intended-feature
semantics and fair experimental integration.

The matrix subsequently completed all 14 cases at 23:22 Asia/Kolkata. The
experimental-full profile produced MSE `0.04245717` and MAE `0.16271803`, which
is `18.96%` worse than the baseline by MSE. Its nominal-80% interval covered
`67.67%`. The result is evidence against enabling the stacked legacy profile,
not a clean attribution to any one component, because the same seed,
initialization, neutrality, and semantic limitations described below remain.

### 14.1 Experiment-control defect

`run.py` exposes `--seed` but hardcodes `2021`. Optional modules are constructed
before shared downstream modules and before shuffled loaders are iterated.
Consequently a one-seed feature run changes common parameter initialization,
batch order, worker seeds, and later dropout streams merely by allocating the
extension. Baseline/SDPA parity proves deterministic replay only when the model
graph is unchanged; it does not make the other rows paired ablations.

The production loop also evaluates test data every epoch. Although checkpoint
selection uses validation, repeated human visibility makes the test period
unsuitable as a pristine confirmatory lockbox.

### 14.2 Feature-by-feature classification

| Feature | Test delta vs baseline | Post-matrix finding | Current classification |
|---|---:|---|---|
| Learned FFT | +1.08% MSE | Sixteen parameters are interpolated as control points over all 61 bins; the trained mask remained near-uniform around 0.5. Fusion is not baseline-neutral. | Inconclusive; selector/filter semantics ineffective in this run |
| Interpretable cross-attention | +1.34% | Wiring is correct and gradient-live. Best validation loss improved about 5.9% while test worsened, indicating split/seed generalization rather than a dead path. Metadata labels the interpretable branch incorrectly. | No isolated forecast defect; neutral integration/reporting needed |
| Latent higher-order | +3.31% | Runs after VSN collapse and multiplies projections of the same latent token. It is not a named original-covariate interaction; returned “contribution” is pre-projection. | Honest latent polynomial block, not evidence for covariate pairs |
| Lag attention | +3.50% | Causal and live on regular ETTh1, but lag `L` attends to the complete prefix through `t-L`, not exactly `x[t-L]`. Compressed/irregular key coordinates use the wrong source formula. | Shifted-prefix ablation; exact/calendar-time lag still absent |
| Temporal compression | +4.53% | Forced at history 96 despite a long-sequence purpose. With one decoder layer every decompressor parameter has zero gradient; final-layer decompression is dead for a future-only loss. | Invalid general compression verdict; codec semantics require redesign |
| Sparse cross-mixing | +8.11% | Strong non-identity transform; one adjacency is broadcast as multiple heads. `top_k=3` means 27% history density but 75% future density. Trained future selected-edge weights were nearly uniform. | Most concerning integration; graph idea not disproven |
| Experimental-full stack | +18.96% | Combines multiple non-neutral and semantically mismatched paths, so interaction and capacity effects cannot be separated. | Reject as a starting profile; not a component-level verdict |

All audited paths other than the dead final decompressor are active and receive
finite gradients. Eighty existing extension/comprehensive tests passed. Those
tests establish shapes, finiteness, and selected gradients; they do not prove
baseline-neutral initialization, semantic recovery, fair paired randomness, or
usefulness against matched controls.

### 14.3 Superseding qualifications to old extension claims

- `TFT-A01` established a per-frequency interpolated spectral mask. It did not
  establish that `modes=K` in learned mode retains/selects `K` bins or that the
  branch learns a useful sparse selector.
- `TFT-A04` preserved sparse support through temporal evolution. It did not
  establish identity-safe graph insertion, genuine multi-head adjacency, or
  comparable history/future density.
- `TFT-A07` established padding masks and regular shifted-history positions. It
  did not establish exact-lag semantics or correct source coordinates after
  irregular/compressed token mapping.
- `TFT-C05` repaired gate cardinality and order-3 execution. It did not make the
  post-VSN block an original-variable interaction mechanism.
- Temporal compression shape/reconstruction tests did not audit every
  decompressor parameter's gradient.

The corrective work is specified as `TFT-SR00`–`TFT-SR09` in
[`implementation_plan.md`](implementation_plan.md#14-post-matrix-native-tft-semantic-repair-wave).
`TFT-SR00` passed with `EV-IMP-025`, and `TFT-SR01` passed independent review
with `EV-IMP-026`; `TFT-SR02` is active.
The release gate uses exact no-op parity, paired initialization/data order,
known-answer synthetic tasks, all-parameter liveness, irregular-time tests, and
a short reproducibility replay rather than another multi-day ETT matrix.

### 14.4 Consequence for the financial-astrology design

The first NIFTY experiment must not combine these generic extensions. It starts
with a small market/calendar baseline and named planetary values as true
known-future covariates. Alleged planetary interactions are explicit pre-VSN
body/pair features; alleged persistence uses elapsed-calendar-time response
states; long orbital cycles use circular and relative phase. Generic FFT, row
lag, graph mixing, compression, MoE, and latent higher-order remain off until a
simple real ephemeris block beats matched smooth nulls.

## 15. Bottom Line

The repository has a promising, feature-rich temporal model. The original native-TFT safety and canonicalization roadmap is implemented, but the completed feature matrix exposed a second class of work: several advanced switches execute without yet implementing the scientific semantics their names imply. The first two post-matrix tasks (`TFT-SR00` and `TFT-SR01`) are closed; `TFT-SR02` through `TFT-SR09` remain the open native gate. This is not merely another hyperparameter comparison.

For domains with real governing equations, the generic physics effort should still begin outside the model with named, unit-aware, differentiable output constraints and synthetic-law validation.

For the NIFTY/planetary hypothesis opened on Wednesday, July 29, 2026, the
initial source audit is complete and found blocking rashi/date/provenance
defects. The next meaningful work is to rebuild authoritative session-dated
market targets, reproduce the planetary generator/conventions, make calendar
plus admitted planetary trajectories true production known-future covariates,
freeze a falsifiable incremental-value protocol, and test a small
capacity-matched raw representation against smooth null ephemerides before
building grouped aspect or multiscale modules.

That sequencing produces three durable assets:

1. a trustworthy canonical native TFT with clearly labeled extensions;
2. a reusable physics-/theory-guided loss framework for future domains with defensible laws; and
3. a separate, scientifically controlled planetary-covariate research lane whose evidence is based on out-of-sample incremental value rather than architectural complexity or attention weights.
