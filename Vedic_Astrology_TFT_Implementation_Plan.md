# Vedic-Astrology Planetary-Covariate TFT Research Plan

> Status: discussion draft and implementation roadmap. No planetary-market code has been implemented yet.
>
> Last updated: 2026-07-29 (Asia/Kolkata).
>
> Scope: repository-native `models/TemporalFusionTransformer.py`; `models/TFT_Nixtla.py` remains out of scope.
>
> First required input: a representative data sample, the column dictionary, and the PySwissEph generation code/configuration.
>
> Scientific scope: test whether a predeclared planetary representation adds out-of-sample predictive information for NIFTY 50 conditional on market history and ordinary calendar controls. A positive result would establish predictive association for that tested representation, not causality or validation of Vedic astrology as a whole.
>
> Canonical cross-session project control now lives in
> [`projects/financial_astrology_tft/README.md`](projects/financial_astrology_tft/README.md).
> This file remains the original repository audit and technical appendix. The
> canonical tracker/orchestrator use `FA-*` task IDs; they supersede the legacy
> `AST-*` IDs here for execution status.

## 1. Executive Decision

The planetary ephemeris should enter TFT as a set of **known-past and known-future covariates**. It should not initially enter as a conventional physics loss.

This distinction matters:

- PySwissEph already computes positions from a physical ephemeris.
- The planetary trajectory is therefore known at forecast time.
- There is no accepted governing equation that maps planetary state to NIFTY returns.
- A loss that forces market predictions to obey an invented planet-to-price equation would build the desired conclusion into the model instead of testing it.

The useful “physics-/theory-guided” components are therefore:

1. correct circular and relative-coordinate representations;
2. physically coherent planet-group and pair-interaction encoders;
3. response kernels for predeclared fast, intermediate, and slow effects;
4. trajectory-preserving placebo/null covariates;
5. optional self-supervised ephemeris pretraining, performed without market labels;
6. structural OHLC constraints, which describe price-bar geometry but do not validate astrology.

The theory audit also found a directly relevant classical source:
*Brihat Samhita* Chapter 42 is explicitly about price fluctuations, while
Chapter 97 assigns different fruition delays to different planetary phenomena.
These source-faithful rules form a distinct classical-text experiment family.
They do not validate astrology, and they must not be silently blended with
Uranus, Neptune, or Pluto. The outer planets belong to a separately reported
modern extension because they are not part of the classical Navagraha list.

The first model must be intentionally small and nested:

```text
market history + calendar
            |
            v
      baseline forecast
            |
            +---- small, separately measurable planetary residual branch
            |
            v
       final forecast
```

The first question is not “can a large TFT fit the data?” It is:

> Does the same forecasting system improve on the same future dates when the real planetary block replaces a matched null block?

## 2. Current Repository Audit

The native TFT contains useful pieces, but the production data path cannot yet run the proposed experiment correctly.

| Area | Current behavior | Consequence | Required action |
|---|---|---|---|
| Known-future model path | `TFTEmbedding` concatenates encoder marks with the last `pred_len` decoder marks and sends them through the future VSN | Planetary future values can be used once correctly supplied | Reuse this contract |
| Custom-known embedding | `TFTCustomKnownEmbedding` accepts an arbitrary named mark width | Synthetic model tests can already pass custom known tensors | Keep as the Stage-1 reference path |
| Production custom loader | `Dataset_Custom` builds `x_mark` only from calendar/time features | Planetary CSV columns never reach the known-future path | Add an explicit planetary-market dataset contract |
| Calendar plus custom-known composition | Custom-known mode replaces the standard timestamp embedding rather than composing with it | A planetary run can accidentally omit its calendar control | Construct one named known block containing calendar and planetary fields, or add separate calendar/planet encoders |
| Feature semantics | Custom-known values share one scalar projection and receive only a channel-ID embedding | Planet identity, field type, and circular pairing are only implicit | Add typed/grouped planetary encoding after the raw baseline |
| Static covariates | Native TFT supports static inputs | A single NIFTY series has no useful entity-static input | Keep `tft_static_pos=[]`; use static asset IDs only in a later multi-asset panel |
| Target mapping | `tft_target_pos` is shared across model and experiment paths | Non-last or multi-output targets are supported | Use explicit names and positions in the manifest |
| Output modes | Point, quantile, and joint heads are available | Probabilistic experiments are possible | Keep them off in the first incremental-value test |
| Advanced interactions | Graph mixing, lag attention, FFT, higher-order blocks, MoE, and cross-attention exist | Many high-capacity combinations are available | Keep them off initially; test one isolated addition at a time |
| Existing lag semantics | Lag attention uses sequence-step lags | Friday-to-Monday and normal one-day steps are treated alike | Add elapsed-calendar-time response features before interpreting lags physically |
| Higher-order block | Operates after VSN collapse in latent coordinates | It does not expose named planet-pair interactions | Do not use it as evidence for specific astrological interactions |
| Observed feature names | `Dataset_Custom` discovers names after the experiment has already built the model | Interpretation can fall back to `f0`, `f1`, and so on | Resolve dataset schema before model construction and persist real names |
| Known-feature scaling | Only market `df_data` is fitted by the current dataset scaler | Radius and velocity can enter the shared projection on incompatible scales | Add training-fold-fitted, per-field known transforms |
| Internal normalization | `tft_use_revin=False` still triggers manual per-window mean/std normalization | “No RevIN” is not a true unnormalized or dataset-only return experiment | Add explicit `dataset`, `window`, `revin`, and `none` normalization modes |
| Optimizer | Production experiment uses plain Adam and exposes no weight decay or gradient clipping | Small noisy data has weaker regularization than intended | Add AdamW, weight decay, clipping, and warmup controls |
| Point losses | Production experiment effectively supports MSE; quantile loss is separate | Robust return training is unavailable | Add Huber and MAE training choices |
| Validation lifecycle | Production training computes test loss after every epoch | Repeated test visibility can influence model choices | Remove epoch-level test evaluation from confirmatory runs |
| Financial splits | `Dataset_Custom` hard-codes one 70/10/20 split | One split cannot establish temporal robustness | Add manifest-driven purged walk-forward folds |
| Per-feature VSN gating | The optional path sets `variable_grns=None` and then evaluates `len(self.variable_grns)` | `--tft_vsn_per_feature_gating` currently crashes | Repair and regression-test before any use |
| Interpretation | Named VSN weights and attention payloads are exported | Diagnostics are possible | Treat weights as diagnostics, never causal proof |
| `d_ff` | Native TFT ignores `d_ff` | Changing it does not change capacity | Do not tune or advertise it for native TFT |

### 2.1 Immediate data-path blocker

`Dataset_Custom` currently returns:

```text
seq_x, seq_y, calendar_x_mark, calendar_y_mark
```

Adding planetary columns to `seq_x` would make them historically observed inputs only and would not expose their future values. Adding them to the CSV without loader changes does not solve the problem.

The required contract is:

```text
x_enc       = historical market-derived variables only
y           = historical/future target variables
x_mark_enc  = calendar controls + planetary values over encoder dates
x_mark_dec  = calendar controls + planetary values over label/future dates
```

No future OHLC-derived value may appear in either mark tensor.

## 3. Falsifiable Research Protocol

### 3.1 Primary null and alternative

Primary null hypothesis:

```text
H0:
Conditional on causal market history and ordinary calendar controls,
the real planetary covariate block does not reduce out-of-sample
next-day NIFTY forecast loss relative to a matched no-planet/null model.
```

Primary alternative:

```text
H1:
The preregistered real planetary covariate block produces a stable,
out-of-sample loss reduction that is larger than matched smooth placebo
planet blocks.
```

The primary comparison is paired on the same forecast dates:

```text
delta_loss[t] = loss_baseline[t] - loss_planet[t]
```

Positive `delta_loss` favors the planetary model.

### 3.2 Forecast decision and target

Provisional decision time:

```text
after the NIFTY trading session closes on date t
```

Provisional primary target:

```text
r_close[t+1] = log(Close[t+1] / Close[t])
```

Provisional primary metric:

```text
MAE of next-trading-day close-to-close log return
```

Secondary metrics:

- MSE/RMSE;
- directional accuracy;
- Spearman rank information coefficient;
- sign-balanced accuracy;
- calibration/coverage after a quantile model is introduced;
- economic utility only as a secondary analysis with declared costs and turnover.

The final definitions must be frozen in `configs/astrology/hypothesis_v1.yaml` before the locked holdout is inspected.

### 3.3 Why raw OHLC levels are not the first target

Raw price levels are nonstationary and highly persistent. A model can obtain low raw-level error by copying the latest price while learning no useful planetary relationship.

The initial test should use stationary market-derived variables. Recommended encoder variables are:

```text
r_close[t] = log(Close[t] / Close[t-1])
gap[t]     = log(Open[t] / Close[t-1])
body[t]    = log(Close[t] / Open[t])
range[t]   = log(High[t] / Low[t])
```

All values must be computed causally. Rolling features must be shifted so their latest member is available at the forecast decision.

### 3.4 Later structurally valid OHLC target

After the primary close-return experiment passes its advancement gate, a four-output head may predict:

```text
g = log(Open[t+1] / Close[t])
b = log(Close[t+1] / Open[t+1])
u = log(High[t+1] / max(Open[t+1], Close[t+1])) >= 0
d = log(min(Open[t+1], Close[t+1]) / Low[t+1])  >= 0
```

Parameterize `u` and `d` through `softplus`, then reconstruct:

```text
Open[t+1]  = Close[t] * exp(g)
Close[t+1] = Open[t+1] * exp(b)
High[t+1]  = max(Open[t+1], Close[t+1]) * exp(u)
Low[t+1]   = min(Open[t+1], Close[t+1]) * exp(-d)
```

This guarantees:

```text
Low <= min(Open, Close) <= max(Open, Close) <= High
```

These are financial bar-consistency constraints, not planet-to-market physical laws.

## 4. Required Data and Provenance Contract

### 4.1 Files to provide for the first audit

Provide:

1. 100–300 consecutive rows of the merged data, preferably including a weekend, an exchange holiday, and a year boundary;
2. the complete ordered column list;
3. a one-line description and unit for every column;
4. the PySwissEph generation script or notebook;
5. the exact ephemeris configuration;
6. the NIFTY source and whether the OHLC series is adjusted/revised;
7. the Hilbert-transform implementation and its window/boundary policy.

CSV or Parquet is acceptable. A sanitized sample is sufficient for the first review.

### 4.2 Required ephemeris manifest

The dataset build must save a machine-readable manifest containing:

```text
pyswisseph package version
Swiss Ephemeris data-file/version identifiers
body list and numeric IDs
geocentric/topocentric/heliocentric frame
tropical or sidereal setting
ayanamsha identifier when sidereal
true/mean node choice
longitude/latitude convention
distance unit
velocity units
ephemeris evaluation timestamp
source timezone and UTC conversion
Julian-day conversion policy
calendar and leap-second assumptions
feature formulas
Hilbert transform/window/boundary policy
market data source/version
market close/cutoff policy
```

Two datasets generated under different frames, ayanamshas, timestamps, or node definitions are different hypotheses and must receive different manifests/digests.

### 4.3 Recommended table roles

| Namespace | Examples | Availability | Model destination |
|---|---|---|---|
| `observed_market` | return, gap, body, range, causal volatility | Through forecast origin only | `x_enc` |
| `target` | next-day return; later OHLC transform | Label only | `batch_y` / loss |
| `known_calendar` | weekday, month, holiday proximity, elapsed calendar days | Past and future | `x_mark_*` |
| `known_planet_raw` | sine/cosine angles, radius, velocity, retrograde | Past and future | `x_mark_*` or grouped encoder |
| `known_planet_rel` | relative-angle harmonics/aspect activations | Past and future | `x_mark_*` or grouped encoder |
| `static` | asset ID in a later panel | Constant per series | static TFT path |

For the single NIFTY series:

```text
tft_static_pos = []
```

### 4.4 Timestamp and trading-calendar alignment

Freeze a single forecast convention. A sensible default is:

```text
market decision timestamp = NSE close, Asia/Kolkata
planetary timestamp        = the same timestamp converted to UTC
target date                = next actual NIFTY trading session
```

Add:

```text
calendar_days_since_previous_session
calendar_days_until_target_session
```

The model must distinguish a normal overnight interval from a weekend/holiday interval.

If an astrological event occurs between two market sessions, a point sample only at the next close may miss it. Later interval features may include preregistered summaries such as:

- minimum aspect distance over the closed-market interval;
- whether an aspect threshold was crossed;
- maximum absolute angular velocity;
- start/end relative angle.

These interval summaries remain known in advance.

### 4.5 Scaling

Use separate, training-fold-fitted transforms:

- market scaler for observed/target variables;
- known-feature scaler for radius and velocity fields;
- fixed identity scaling for bounded sine/cosine fields;
- explicit categorical encoding for flags such as retrograde.

Never fit scalers, clipping thresholds, PCA, or feature-selection rules on validation/test rows.

The loader must persist names and transform parameters in the run manifest.

### 4.6 Hilbert-transform rule

The phrase “Hilbert transformed” is insufficient to establish safety.

- A full-sample Hilbert transform of market values is future leakage.
- A causal/split-local market transform is allowed if its latency and boundary behavior are documented.
- Planetary trajectories are deterministic and future-computable, so an acausal planetary transform can be deployable in principle.
- Even for planets, a transform tied to the arbitrary start/end of the saved CSV can create boundary artifacts and non-reproducible results.

The confirmatory model should therefore start from direct sine/cosine coordinates and declared velocities. Hilbert-derived planetary fields belong in a separate exploratory ablation.

### 4.7 Rahu/Ketu redundancy

Under the usual node construction, Rahu and Ketu are antipodal. Their longitude sine/cosine values are therefore exact sign transforms.

Default confirmatory representation:

```text
one node token + an explicit opposition relation
```

The redundant two-node form may be evaluated only as a named ablation. It must not be interpreted as evidence from two independent bodies.

## 5. Planetary Feature Families

Every family must be named and preregistered. Do not generate a large feature library and report only the winning subset.

### 5.1 Family `RAW-LON`

Minimal per-body representation:

```text
sin(longitude)
cos(longitude)
scaled longitude angular velocity
retrograde flag
```

This is the preferred first planetary block.

### 5.2 Family `RAW-3D`

Exploratory extension:

```text
sin/cos longitude
sin/cos latitude
log or robustly scaled radius
longitude/latitude/radial velocity
retrograde flag
```

Do not include this family until the sample audit confirms units and sign conventions.

### 5.3 Family `REL-HARMONIC`

For bodies `i` and `j`, define:

```text
delta_lambda = longitude_i - longitude_j
sin(k * delta_lambda)
cos(k * delta_lambda)
```

Begin with:

```text
k in {1, 2, 3}
```

This respects circular geometry and can represent broad conjunction/opposition/trine-like patterns without discontinuous angle wrapping.

### 5.4 Family `VEDIC-ASPECT`

Only after the exact doctrinal hypothesis is written, add periodic soft activations around the specified aspect centers:

```text
activation_a(delta) = exp(-circular_distance(delta, aspect_a)^2 / (2 * orb_a^2))
```

The following must be fixed before training:

```text
eligible planet pairs
aspect centers
orb widths
directionality
sign or unsigned interpretation
whether absolute zodiac position also matters
```

Do not learn dozens of orb widths on the same validation period. That would turn the hypothesis test into an uncontrolled search.

### 5.5 Calendar/Fourier control family

Planetary positions are deterministic functions of time. A fair null must therefore include a flexible ordinary-time representation:

- weekday/month/holiday features;
- annual sine/cosine;
- a small preregistered set of long-period Fourier terms;
- elapsed-time variables.

If planetary features beat a weak calendar encoding but not this control, the result is date encoding rather than specific planetary information.

## 6. Evaluation Splits and Leakage Protection

### 6.1 Provisional expanding-window design

If the usable sample ends in 2025, use:

| Fold | Training | Validation |
|---|---|---|
| 1 | 1995–2007 | 2008–2010 |
| 2 | 1995–2010 | 2011–2013 |
| 3 | 1995–2013 | 2014–2016 |
| 4 | 1995–2016 | 2017–2020 |
| Locked test | retrain through 2020 | 2021–2025 |

Adjust the final years only after seeing the actual coverage, missingness, and market-data quality. Once frozen, do not move the boundary in response to performance.

### 6.2 Purging and context

- Purge at least `pred_len` target dates between training labels and validation labels.
- For targets derived from longer forward windows, purge the full maximum forward horizon.
- Historical context immediately before a validation target is legitimate if it would have been available at that forecast origin.
- Any transform fitted from values must be training-fold local.

### 6.3 Seeds

Use at least five fixed neural-network seeds:

```text
17, 42, 2021, 3407, 9001
```

Do not select the best seed. Aggregate the paired result distribution.

### 6.4 Locked test rule

The final test set is inspected once for the frozen primary comparison. Any modification prompted by the locked-test result creates a new exploratory version and requires a new external/temporal holdout before confirmatory claims.

## 7. Baseline and Control Ladder

### 7.1 Required predictive arms

| ID | Inputs | Purpose |
|---|---|---|
| `B0-ZERO` | none | zero-return baseline |
| `B1-RIDGE-M` | market history | low-variance linear baseline |
| `B2-RIDGE-MC` | market + calendar/Fourier | tests ordinary time encoding |
| `B3-TFT-M` | market history | small nonlinear sequence baseline |
| `B4-TFT-MC` | market + calendar/Fourier | primary neural null |
| `P1-RIDGE-MCP` | market + calendar + raw planets | cheap incremental screening |
| `P2-TFT-MCP` | market + calendar + raw planets | primary planetary TFT arm |
| `P3-TFT-REL` | plus relative geometry | interaction ablation |
| `P4-TFT-MULTI` | plus multiscale response encoder | duration ablation |
| `P-ONLY` | planetary/calendar only | diagnostic, not primary evidence |

The first neural claim is:

```text
P2-TFT-MCP versus B4-TFT-MC
```

### 7.2 Capacity matching

Planetary arms naturally add parameters. A fair comparison must use one of:

1. the same residual branch in every arm, with its input replaced by zeros/null covariates;
2. a fixed-size planetary encoder output and a matched nuisance encoder for the null;
3. an explicitly matched total parameter budget.

Do not compare a much larger planet model to a smaller calendar model and attribute the difference to planets.

### 7.3 Required null/placebo blocks

Generate complete, named placebo datasets:

- coherent circular date shifts of the full ephemeris block;
- shifts such as 30, 90, and 365 calendar days, frozen before evaluation;
- Fourier phase-randomized trajectories preserving spectrum/autocorrelation;
- smooth pseudo-planets matched for dimension and persistence;
- block-permuted target labels as a pipeline-failure check.

Independent row shuffling is not a sufficient null because it destroys planetary smoothness and makes the real block artificially easy to distinguish.

All planet bodies must move coherently under a trajectory shift. Independently perturbing one body creates an off-manifold solar-system state and is not the primary counterfactual.

## 8. Initial Native-TFT Configuration

The initial configuration is deliberately conservative.

```text
task_name                       long_term_forecast
data                            planetary_market (new loader key)
features                        MS
target                          r_close
freq                            B (business-day timestamps; elapsed calendar gaps remain explicit)
seq_len                         128
label_len                       32
pred_len                        1
enc_in                          4 (provisional market channels)
c_out                           1
tft_target_pos                  index of r_close in x_enc
tft_static_pos                  empty
tft_observed_pos                all market-channel indices
tft_allow_custom_known          true
tft_known_len                   resolved from manifest
tft_known_feature_names         resolved from manifest
tft_profile                     extended_safe
tft_temporal_backbone           lstm
tft_temporal_backbone_layers    1
d_model                         8 for raw high-width known input
n_heads                         1
e_layers                        1
dropout                         0.25
tft_attention_dropout           0.10
tft_use_revin                   false
tft_normalization_mode          dataset (new control; requires production patch)
tft_use_quantile_head           false
tft_output_mode                 point
tft_full_attention              false
tft_dual_attention_fusion       false
tft_use_explicit_cross_attention false
tft_use_lag_attention           false
tft_cross_variable_mixing       false
tft_use_higher_order            false
tft_use_regime_moe              false
tft_use_fft_branch              false
tft_use_temporal_compression    false
tft_covariate_reattention       false
tft_vsn_per_feature_gating      false until repaired
batch_size                      64
optimizer                       AdamW (requires production patch)
learning_rate                   5e-5
weight_decay                    0.005
gradient_clip_norm              1.0
warmup_epochs                   3
scheduler                       cosine
train_epochs                    50 maximum
early_stop_patience             8
training_loss                   Huber (requires production patch)
primary_selection_metric        validation MAE
```

Notes:

- `d_ff` is ignored by this native TFT and must not be treated as a capacity control.
- `label_len` is currently carried by the dataset/decoder contract, but decoder market values are not consumed by native TFT; it is not a meaningful hyperparameter for this model.
- `tft_use_revin=false` currently falls back to manual window normalization. `AST-B01` must add an explicit normalization mode; the provisional primary return experiment uses dataset-only scaling and treats window normalization as a development ablation.
- If the known block is compressed to at most 8–16 latent channels by the grouped encoder, compare `d_model=8` and `d_model=16` inside development folds.
- Keep total trainable parameters below approximately 50,000 for the first raw-covariate study where possible.

Current-model parameter audit for the provisional four-observed/one-target setup:

| Known width | `d_model=8`, `n_heads=1` | `d_model=16`, `n_heads=2` |
|---:|---:|---:|
| 12 | 20,181 | 66,749 |
| 32 | 37,901 | 133,061 |
| 64 | 57,933 | 205,445 |
| 96 | 82,189 | 294,725 |

This is why `d_model=8` is the raw high-width starting point. `d_model=16` becomes reasonable only after the grouped planetary encoder compresses the known block or a validation-backed parameter-budget decision is recorded.

### 8.1 Sequence-length policy

Primary local-market choice:

```text
seq_len = 252 trading sessions
```

Permitted development-only comparison:

```text
seq_len in {64, 128, 252, 504}
```

This `seq_len` is not the complete astrological memory. The corrected design
uses separate clocks:

| Clock | Initial representation |
|---|---|
| Target-date state | Exact current/future circular phase, rashi, nakshatra, speed, retrograde/station, ingress and pair geometry |
| Fast | Calendar-daily history of about 96 days for Moon/Mercury and exact events |
| Medium | Weekly or event-token history of about five years |
| Slow | Monthly history up to about 40 years plus recursive calendar-time response states |
| Secular outer-planet | Current/relative phase and event clocks; optional low-capacity annual diagnostic grid |

Do not use a many-year raw daily LSTM to represent Saturn, Jupiter, or the outer
planets. Circular phase already identifies the body's position in its cycle;
event clocks identify applying/exact/separating state; response kernels represent
the alleged duration of an effect. These are different quantities.

Thirty years contains only roughly 2.5 Jupiter periods and about one Saturn
period. NIFTY history contains only fractions of the Uranus, Neptune, and Pluto
periods. A longer input tensor cannot manufacture independent market cycles.

Long-duration hypotheses should use explicit, regularized response summaries and must be described as weakly identifiable on NIFTY alone.

### 8.2 Horizon policy

Run in order:

1. `pred_len=1`: primary next-session test;
2. `pred_len=5`: short persistence;
3. `pred_len=20`: approximately monthly persistence;
4. optional `pred_len=60`: exploratory slow response.

For multi-step runs, report every horizon separately and account for overlapping forecast errors. Do not hide weak horizons inside one averaged loss.

## 9. Recommended Planetary Architecture

Do not begin with the repository’s full experimental profile. Introduce a small, named planetary branch only after the raw known-covariate baseline is operational.

### 9.1 Nested residual formulation

Recommended prediction:

```text
y_hat = y_hat_market_calendar + alpha * delta_y_hat_planet
```

Requirements:

- the baseline and planetary arm share the market/calendar backbone;
- `alpha` is scalar or target-specific and initialized near zero;
- regularize `alpha` and/or the residual output toward zero;
- log the baseline forecast, planetary correction, and effective `alpha`;
- provide an exact mode that disables the planetary correction;
- use the same parameterized branch with matched null inputs for placebo comparisons.

A simpler first implementation is a cross-fitted residual study:

1. produce out-of-fold residuals from the market+calendar model;
2. train a regularized planetary model only on those training-fold residuals;
3. evaluate whether it predicts residuals on future folds.

This is cheaper and makes incremental information explicit before deep integration.

### 9.2 Grouped planet tokens

Represent known planetary input as:

```text
x_planet: [batch, time, planet, field]
```

Each planet token should include only fields present under a frozen schema. A small shared encoder is:

```text
planet_token[p] =
    SharedMLP(numeric_state[p])
  + PlanetIDEmbedding[p]
  + OptionalGroupEmbedding[fast/intermediate/slow]
```

Shared weights reduce capacity and force comparable state semantics across bodies.

### 9.3 Relative-geometry graph

Construct edges from physically coherent pair features:

```text
edge(i,j) = [
    sin(k * delta_longitude_ij),
    cos(k * delta_longitude_ij),
    relative_velocity_ij,
    optional preregistered aspect activations
]
```

Use:

- one message-passing layer initially;
- low-rank/shared edge functions;
- no unrestricted per-pair MLP bank;
- a declared Rahu/Ketu relation;
- optional sparse edges only when justified before outcome inspection.

Pool the graph to an 8- or 16-dimensional planetary latent state.

### 9.4 Fast, intermediate, and slow response paths

The following is a starting hypothesis grid, not an assertion that the effects exist:

| Path | Provisional bodies/relations | Calendar-day half-life emphasis |
|---|---|---|
| Fast | Moon, Mercury and exact fast triggers | 1, 3, 7, 14, 30 |
| Intermediate | Sun-related geometry, Venus, Mars | 14, 30, 90, 180, 365 |
| Slow | Jupiter, Saturn, lunar node | 180, 365, 730, 1,825, 3,650 |
| Secular exploratory | Uranus, Neptune, Pluto | 1,825, 3,650, 7,300, 10,958 |

Use fixed elapsed-calendar-time response banks first. The fixed grid does not
assert that every scale is real; shrinkage and preregistered group masks let the
data reject them. Later allow a small learned convex combination with:

- nonnegative normalized kernel weights;
- smoothness regularization;
- group sparsity;
- logged effective half-life.

Response time must use actual elapsed calendar time or interval-aware features. A raw trading-step lag is not a physical duration.

### 9.5 Optional ephemeris-only pretraining

PySwissEph can generate abundant planetary trajectories outside the market-label period. A later small-data strategy may:

1. generate a versioned ephemeris-only corpus;
2. train masked-state reconstruction, next-state prediction, or relative-geometry reconstruction;
3. freeze most of the planet encoder;
4. train only a small market adapter.

This is allowed because no future market labels are used. It is optional: handcrafted circular/relative features are a lower-risk baseline and must be tested first.

## 10. Mapping Existing Advanced TFT Features

| Native feature | Initial decision | Reason |
|---|---|---|
| Canonical/interpretable attention | Keep basic interpretable path | Smallest reference |
| Full/dual attention | Off | Adds capacity and weakens clean attribution |
| Generic lag attention | Off initially | Uses latent trading-step lags, not named planet-specific durations |
| Cross-variable graph mixing | Off initially | Entangles market/calendar/planet variables and changes parameter count |
| Higher-order latent interaction | Off initially | Not a named original-covariate planet interaction |
| FFT branch | Off initially | Strong global periodic inductive bias can confound “planet” with generic time periodicity |
| Regime MoE | Off initially | Too many degrees of freedom for the first small-data test |
| Temporal compression | Off at `seq_len<=252` | Unnecessary and complicates temporal interpretation |
| Covariate reattention | Off initially | Existing path is not a clean named group interaction test |
| Quantile head | Later | First establish point incremental value |
| Per-target heads | Later with structured OHLC | Irrelevant for the first single target |
| Per-feature gating | Do not use until fixed | Current forward path crashes |

Each advanced option must be an isolated ablation against the same frozen baseline. Never activate `experimental_full` and interpret the combined result as evidence for a particular mechanism.

## 11. Statistical Analysis

### 11.1 Paired inference

Store per-date predictions and losses for every arm, seed, fold, and placebo.

Use:

- stationary or moving-block bootstrap with a preregistered 20–60 session block range;
- Diebold–Mariano/HAC-aware comparisons for overlapping horizons;
- confidence intervals on paired loss improvement;
- effect distributions across seeds and folds.

### 11.2 Multiplicity

Treat the complete frozen planetary block as the one primary hypothesis.

- Use Holm correction for a small number of named secondary families.
- Use FDR for a larger exploratory feature family.
- If many model/trading-rule variants are searched, use a reality-check/SPA-style correction.

Never promote the single best planet, aspect, lag, seed, or crisis subperiod without correcting for the search that produced it.

### 11.3 Advancement gate

A planetary component advances only if it:

1. improves median development MAE by a predeclared practical margin, provisionally 1%;
2. improves at least three of four temporal development folds;
3. is stable across most fixed seeds;
4. beats at least 95% of matched smooth placebo blocks;
5. remains beneficial against the flexible calendar/Fourier control;
6. is not driven entirely by one crisis window;
7. produces a positive frozen-test effect with an uncertainty interval consistent with improvement.

Failure is informative. Report the null result without expanding the search until something becomes significant.

### 11.4 Interpretation rule

VSN weights, attention maps, graph weights, and planetary residual gates are diagnostics.

They become credible only when:

- stable across folds and seeds;
- confirmed by blocked leave-one-group-out ablations;
- stronger for real ephemerides than null ephemerides;
- not interchangeable with calendar/Fourier features.

They are not causal evidence.

## 12. Implementation Tasks

### `AST-H01` — Freeze the hypothesis protocol

**Dependencies:** `G2`.

**Files:**

- `configs/astrology/hypothesis_v1.yaml`
- `docs/astrology/HYPOTHESIS_PROTOCOL.md`
- tests for schema validation

**Steps:**

1. Freeze forecast decision time.
2. Freeze primary target, metric, fold boundaries, seeds, and comparison.
3. Separate confirmatory and exploratory feature families.
4. Freeze final-test access policy.
5. Define the claim language for positive, null, and unstable results.

**Done when:** the protocol parses, hashes deterministically, and cannot be changed without a new version.

### `AST-D01` — Audit data and ephemeris provenance

**Dependencies:** sample data and generation code supplied; may proceed alongside `AST-H01`.

**Files:**

- `data_provider/planetary_schema.py`
- `configs/astrology/dataset_manifest_v1.yaml`
- `tests/test_planetary_schema.py`

**Steps:**

1. Inspect every column, unit, range, missing value, and duplicate timestamp.
2. Validate OHLC inequalities and price positivity.
3. Validate sine/cosine unit-circle error.
4. Validate velocity against finite differences within declared tolerance.
5. Check Rahu/Ketu redundancy.
6. reproduce selected rows from the generation code;
7. audit Hilbert boundary behavior;
8. freeze timezone/trading-day joins.

**Done when:** the same source/config regenerates the audited sample and all provenance fields are resolved.

### `AST-C00` — Repair high-covariate TFT prerequisites

**Dependencies:** `G2`.

**Files:**

- `models/TemporalFusionTransformer.py`
- `tests/test_tft_extension_contracts.py`

**Steps:**

1. Add a regression test for `VariableSelectionNetwork(..., per_feature_gating=True)`.
2. Replace the invalid `len(self.variable_grns)` check with an explicit stored variable count.
3. Test rank-3 and rank-4 inputs, with and without returned weights.
4. Run the native TFT regression suite.

**Done when:** per-feature gating forwards/backpropagates correctly or the flag is explicitly rejected. The first planetary run still keeps it disabled.

### `AST-K01` — Add the production custom-known loader

**Dependencies:** `AST-D01`.

**Files:**

- `data_provider/data_loader.py` or a dedicated `data_provider/planetary_market.py`
- `data_provider/data_factory.py`
- `utils/tft_schema.py`
- `run.py`
- `tests/test_planetary_market_loader.py`

**Steps:**

1. Resolve observed, target, calendar, and planetary columns by name.
2. Build past and future known tensors for exact trading target dates.
3. Fit separate transforms on training data only.
4. Persist ordered names and transforms.
5. Verify future OHLC never appears in marks.
6. Verify decoder marks contain real future planetary values.
7. Verify calendar controls remain present in every planetary arm.
8. Verify split boundaries and holiday gaps.

**Done when:** an end-to-end batch has the exact names, shapes, dates, and availability contract expected by native TFT.

### `AST-B01` — Implement baselines and frozen metrics

**Dependencies:** `AST-H01`, `AST-K01`.

**Files:**

- `exp/exp_planetary_forecasting.py` or a reusable evaluator
- `utils/astrology_metrics.py`
- `scripts/long_term_forecast/planetary/`
- tests

**Steps:**

1. Add zero-return and previous-state baselines.
2. Add Ridge/ElasticNet market and market+calendar baselines.
3. Add identical small TFT `M` and `MC` arms.
4. Add production AdamW, weight decay, warmup, gradient clipping, Huber/MAE choices, and explicit normalization modes.
5. Remove epoch-level test evaluation from confirmatory training.
6. Save per-date predictions/losses.
7. Add block-bootstrap comparison.
8. Ensure validation, not test, selects checkpoints/hyperparameters.

**Done when:** all non-planet baselines are reproducible across folds and seeds.

### `AST-F01` — Build the circular planetary feature bank

**Dependencies:** `AST-D01`, `AST-K01`.

**Files:**

- `features/planetary_features.py`
- feature configuration under `configs/astrology/`
- tests

**Steps:**

1. Implement `RAW-LON`.
2. Implement optional `RAW-3D`.
3. Implement relative harmonics.
4. Implement only preregistered Vedic aspect activations.
5. Add Rahu/Ketu policy.
6. Add no-lookahead and transform-invariance tests.
7. Persist exact feature names/formulas.

**Done when:** features reproduce from raw ephemeris values and every feature belongs to a frozen family.

### `AST-N01` — Implement matched null ephemerides

**Dependencies:** `AST-D01`, `AST-F01`.

**Files:**

- `features/planetary_nulls.py`
- tests and saved null manifests

**Steps:**

1. Implement coherent date shifts.
2. Implement spectrum-preserving phase surrogates.
3. Implement smooth dimensionality-matched pseudo-planets.
4. Verify marginal scale, spectrum, and autocorrelation matching.
5. Keep all bodies coherent within each null trajectory.

**Done when:** every real-feature experiment has a reproducible null ensemble.

### `AST-E00` — Run the raw known-covariate TFT test

**Dependencies:** `AST-H01`, `AST-B01`, `AST-F01`, `AST-N01`.

**Steps:**

1. Run Ridge raw-planet screening.
2. Run the fixed small TFT market+calendar arm.
3. Run the capacity-matched raw-planet arm.
4. Run matched shifted/surrogate arms.
5. Compare paired development-fold losses only.

**Done when:** a decision is recorded to stop, revise data quality, or proceed to grouped interactions.

### `AST-M01` — Add the grouped planetary interaction encoder

**Dependencies:** `AST-E00` must show enough stable incremental signal or a separately justified representation test.

**Files:**

- `layers/PlanetaryKnownEncoder.py`
- `models/TemporalFusionTransformer.py`
- schema/config/tests

**Steps:**

1. Group scalar fields into planet tokens.
2. Add shared body encoder and ID embeddings.
3. Add low-rank relative-geometry messages.
4. Pool to a fixed-width latent known block.
5. Export group/pair diagnostics.
6. Add exact disable/null modes.
7. Benchmark parameter count and overfit gap.

**Done when:** isolated grouped encoding beats or simplifies the raw representation under the same protocol.

### `AST-L01` — Add multiscale response kernels

**Dependencies:** `AST-M01`.

**Files:**

- `layers/PlanetaryResponseKernels.py`
- integration/tests

**Steps:**

1. Implement calendar-time-aware fast/intermediate/slow fixed kernels.
2. Add a small regularized learned mixture.
3. Log effective half-lives and weights.
4. Verify causal historical summaries and known-future availability.
5. Compare each response path separately.

**Done when:** duration claims are tied to named bodies/relations and robust out-of-sample ablations.

### `AST-O01` — Add the structured OHLC head

**Dependencies:** primary return test and architecture gate passed.

**Steps:**

1. Implement gap/body/upper/lower outputs.
2. Enforce positive excursions.
3. Reconstruct OHLC.
4. Test inequalities and inverse scaling.
5. Compare with unconstrained multi-output baselines.

**Done when:** every forecast is a valid OHLC bar and target semantics are tested end to end.

### `AST-Q01` — Add probabilistic forecasts

**Dependencies:** stable point model.

**Steps:**

1. Add quantile-only and joint ablations.
2. Save coverage, interval width, pinball loss, and calibration.
3. Use the repaired quantile contract.
4. Do not use worse point MSE alone to reject useful uncertainty estimates.

**Done when:** probabilistic quality is measured separately from point accuracy.

### `AST-E01` — Run the full walk-forward and robustness study

**Dependencies:** selected frozen architecture, null ensemble, metrics.

**Steps:**

1. Run every frozen development fold and seed.
2. Apply multiplicity correction.
3. Freeze the selected configuration.
4. Retrain through the final development boundary.
5. Inspect the locked test once.
6. Run frame/timestamp/ayanamsha and rolling-window robustness variants as labelled secondary tests.

**Done when:** predictions, manifests, checkpoints, null comparisons, and confidence intervals are saved.

### `AST-R01` — Claim/evidence gate

**Dependencies:** `AST-E01`.

**Done when:** the result is classified as:

- stable incremental association;
- null/no detectable incremental association;
- unstable/regime-specific exploratory association; or
- invalid/inconclusive due to data or protocol failure.

No stronger wording is allowed without additional markets or longer independent history.

## 13. Dependency Graph

```text
G2
├─> AST-H01 ──────────────┐
├─> AST-C00               │
└─> AST-D01 ─> AST-K01 ─> AST-B01
             └> AST-F01 ─> AST-N01

AST-H01 + AST-B01 + AST-F01 + AST-N01
                    |
                    v
                 AST-E00
                    |
        stable representation signal?
             /                 \
           no                   yes
          stop          AST-M01 -> AST-L01
                              \       /
                               AST-E01 -> AST-R01

Stable point model -> AST-O01 / AST-Q01
```

Generic `PHY-*` output-law tasks are not dependencies of this branch. They remain useful for domains with defensible governing equations, but orbital correctness of already-computed covariates does not supply a price equation.

## 14. First Concrete Work Package

The first implementation package stops before modeling:

```text
WP-1
1. Receive representative sample and generator.
2. Complete AST-D01 data/provenance report.
3. Freeze AST-H01 protocol fields that depend on the schema.
4. Repair AST-C00 per-feature-gating regression.
5. Implement AST-K01 loader with anti-leakage tests.
6. Inspect one real batch and its exact forecast dates.
```

Only then create the first training script.

Expected first training comparison:

```text
B4-TFT-MC
versus
P2-TFT-MCP
versus
capacity-matched shifted/surrogate planet arms
```

This ordering prevents weeks of architecture work from being built on a wrong timestamp join, a leaking transform, or planetary columns that never reach the decoder.

## 15. Definition of Done

The project is not complete because one configuration reports a lower test loss.

It is complete only when:

1. data provenance and ephemeris generation are reproducible;
2. forecast-time availability is proven by tests;
3. target and calendar/planet roles are explicit;
4. the primary hypothesis and final holdout were frozen in advance;
5. real and null planet blocks share capacity and training budgets;
6. results are paired by date and uncertainty-aware;
7. multiplicity from feature/model search is reported;
8. fast/slow claims use named, calendar-time-aware response paths;
9. interpretation is stable and ablation-confirmed;
10. null, negative, and unstable results are reported honestly;
11. any positive conclusion is phrased as predictive association for the tested representation;
12. all code, configs, manifests, predictions, and evidence are reproducible.
