# Financial Astrology TFT Implementation Plan

> Status: canonical planning document.
>
> Implementation is paused pending completion of the native TFT feature matrix,
> user discussion, and data audit.
>
> This plan targets `models/TemporalFusionTransformer.py`, not `TFT_Nixtla.py`.

## 1. Outcome and Non-Outcome

The project succeeds operationally when it can determine whether a frozen
astronomical/Jyotisha feature family adds reproducible forward-period information
relative to:

- market history alone;
- market plus ordinary calendar/Fourier controls;
- capacity-matched random features;
- smooth, trajectory-matched false ephemerides;
- the same planetary architecture with its residual gate disabled.

The project does not succeed because a large model obtains one low test loss.

## 2. Research Formulation

Primary provisional decision time:

```text
after NIFTY trading session t closes
```

Primary provisional target:

```text
r[t+1] = log(close[t+1] / close[t])
```

Primary provisional selection metric:

```text
validation MAE
```

Required secondary target channels:

```text
absolute return / volatility
overnight gap
intraday body
high-low range
drawdown/crash hazard
1, 5, 20, 60, 126, and 252-session cumulative outcomes
```

The confirmatory target, metric, and horizon are not frozen until the data and
theory discussions close.

## 3. Why This Is Theory-Informed, Not a Market PINN

PySwissEph supplies a physically coherent ephemeris. Physical geometry justifies:

- circular encodings;
- relative phase;
- signed velocities;
- event timing;
- consistent coordinate frames;
- recursive state updates using real elapsed time.

Jyotisha supplies candidate:

- rashi/nakshatra state;
- retrograde/station/ingress events;
- drishti and conjunction relations;
- dignity/combustion/node/eclipse state;
- body-specific time of fruition;
- slow-background × fast-trigger interactions.

No accepted differential equation maps these states to NIFTY price. Inventing
such an equation and penalizing deviations would assume the result. The first
model therefore treats the theory as a feature/interaction/lag prior and tests
incremental predictive value.

## 4. Corrected Sequence and Memory Design

### 4.1 Core distinction

```text
market seq_len != planetary orbital period != alleged response duration
```

- `seq_len` captures recent market dynamics.
- circular state captures position within an orbital or synodic cycle.
- event clocks capture approach/exactness/separation.
- response kernels capture alleged persistence after an event.
- coarse grids provide additional historical context without a thirty-year
  daily LSTM.

### 4.2 Initial clocks

| Stream | Frequency | Length | Role |
|---|---|---:|---|
| Market | trading daily | 252 primary; 64/128/504 sensitivity | Recent returns, gaps, body, range |
| Target-date astronomy | exact timestamp | current + forecast dates | Known-future state |
| Fast astronomy | calendar daily | about 96 history days plus known future | Moon/Mercury and exact triggers |
| Medium astronomy | weekly or event tokens | about 260 weeks | Five-year loops/events |
| Slow astronomy | monthly or recursive state | about 480 months/all available | Jupiter/Saturn/node regime |
| Secular diagnostic | annual ephemeris-only | optional 256 tokens | Outer-planet representation diagnostic |

The first implementation uses target-date state and the fixed response bank. The
multi-grid encoders are added only if simpler representations survive matched
null tests.

### 4.3 Identifiability limit

Approximately 1995–2026 supplies:

```text
about 2.6 Jupiter orbits
about 1.05 Saturn orbits
about 0.37 Uranus orbit
about 0.19 Neptune orbit
about 0.12 Pluto orbit
```

No sequence length or pretrained ephemeris encoder creates missing market
outcomes. Outer-planet results on NIFTY alone are local-arc, secular-regime
associations. Stronger evidence requires older targets, commodities, or a
multi-market panel; shared dates still are not independent time replications.

## 5. Architecture Roadmap

### 5.1 Stage A — Transparent flat-feature baseline

Use the native TFT custom-known path:

```text
observed encoder:
    causal market returns/gap/body/range

known encoder/decoder:
    ordinary calendar controls
    + frozen planetary feature block
```

This stage answers whether any simple incremental signal exists before a bespoke
planetary network is built.

### 5.2 Stage B — Nested planetary residual

Required prediction structure:

```text
y_market = MarketCalendarBackbone(market, calendar)

z_planet = PlanetEncoder(
    target_state,
    response_state,
    event_state
)

delta_planet = PlanetResidual(z_planet, market_context)

y_hat = y_market + alpha * delta_planet
```

Requirements:

- initialize `alpha` at or near zero;
- allow an exact disabled-gate run;
- regularize the correction toward zero;
- log `y_market`, `delta_planet`, and `alpha`;
- use the same branch and parameter count for real and placebo ephemerides;
- do not describe VSN weights as causal effects.

### 5.3 Stage C — Typed body and pair encoder

Input shape:

```text
[batch, time, body, field]
```

Per-body token:

```text
SharedBodyMLP(numeric_state)
+ BodyIDEmbedding
+ OptionalClockGroupEmbedding
```

Pair edge:

```text
wrapped relative phase
relative speed
aspect activation
applying/separating
signed time to exactness
```

Use one low-rank directed graph/message layer, then pool to 8–16 named latent
known channels. Named aspect interactions occur before the native VSN. The
existing generic TFT higher-order block occurs after variable collapse and is
not a substitute.

### 5.4 Stage D — Fixed response bank

For half-life `H` and actual elapsed calendar days:

```text
h(t) = 2^(-delta_days/H) * h(t-1)
     + (1 - 2^(-delta_days/H)) * z(t)
```

Initial half-life grid:

```text
1, 3, 7, 14, 30, 90, 180, 365, 730,
1825, 3650, 7300, 10958 days
```

Use group masks and shrinkage:

- Moon/Mercury: fast-heavy;
- Sun/Venus/Mars: fast/medium;
- Jupiter/Saturn/nodes: medium/slow;
- outer planets: slowest, modern arm only.

Fixed states come before a learned SSM. A later 8–16-state stable continuous-time
SSM may learn small deviations from the fixed log-spaced rates.

### 5.5 Stage E — Multi-resolution encoder

Only after Stage D:

```text
FastDailyEncoder -> pooled fast state
WeeklyEventEncoder -> pooled medium state
MonthlySlowEncoder -> pooled slow state
```

Fuse with a small gate:

```text
z = Gate([market_context, fast, medium, slow]) *
    Projection([fast, medium, slow])
```

Do not concatenate all grids into the current single LSTM. Each grid has its own
mask, timestamp delta, and pooling.

## 6. Starting Model Configuration

Final values are frozen after the TFT feature matrix and data audit. Starting
development values:

| Parameter | Value |
|---|---|
| task | next-session return forecasting |
| `seq_len` | 252 |
| sensitivity | 64, 128, 504 |
| `pred_len` | 1 first; then 5, 20, 60 |
| observed features | close return, gap, body, range |
| static positions | none |
| `d_model` | 8 for wide flat input; 16 after 8–16-channel grouped compression |
| heads | 1 at `d_model=8`; 2 at 16 |
| encoder layers | 1 |
| temporal backbone | LSTM reference |
| dropout | 0.25 |
| attention dropout | 0.10 |
| optimizer | AdamW |
| learning rate | `5e-5` |
| weight decay | `0.005` |
| gradient clipping | `1.0` |
| warmup | 3 epochs |
| scheduler | cosine |
| maximum epochs | 50 |
| early-stop patience | 8 |
| batch size | 32 or 64 |
| initial loss | Huber on standardized target |
| RevIN | off for stationary returns |
| point/quantile | point first |
| parameter budget | preferably below 50k–100k |

Initial advanced switches:

```text
full attention: off
dual attention: off
explicit cross attention: off
generic lag attention: off
generic graph cross-mixing: off
generic higher order: off
FFT: off
MoE: off
temporal compression: off
covariate reattention: off
per-feature gating: off until its current crash is repaired
```

The live ETTh1 matrix may identify a stable attention/backend improvement. Once
complete, one reference configuration is frozen and held identical across all
financial-astrology arms. Quantile-only and ALiBi are currently promising, but
partial matrix results are not the final selection.

## 7. Baseline and Null Ladder

### 7.1 Required baselines

| ID | Model |
|---|---|
| B00 | zero-return / historical-volatility baseline |
| B01 | linear/ridge market-only |
| B02 | linear/ridge market + ordinary calendar/Fourier |
| B03 | small TFT market-only |
| B04 | same TFT market + ordinary calendar/Fourier |
| B05 | same TFT plus dimensionality-matched random smooth covariates |

### 7.2 Planet arms

| ID | Added block |
|---|---|
| P01 | raw continuous classical ephemeris |
| P02 | rashi/nakshatra/ingress |
| P03 | retrograde/station/combustion |
| P04 | classical aspects/conjunction graph |
| P05 | nodes/eclipses/panchanga |
| P06 | dignity/dispositor |
| P07 | fixed response bank |
| P08 | literal classical price/fruition rules |
| P09 | grouped graph encoder |
| P10 | multi-resolution memory |
| P11 | anchored mundane |
| P12 | modern outer planets |

### 7.3 Matched nulls

Every planet arm receives:

1. coherent calendar-date shifts of the entire ephemeris;
2. global longitude rotation, preserving relative aspects but breaking absolute
   rashi;
3. per-body phase rotations, breaking real pair relations;
4. spectrum/autocorrelation-matched pseudo-planets;
5. exact disabled residual gate.

Nulls have deterministic manifests and never use test outcomes to choose a shift.

## 8. Evaluation Protocol

### 8.1 Walk-forward

Use expanding training folds with forward validation/test blocks. The exact dates
are frozen after data inspection.

Rules:

- train-fold-only scaling;
- purge overlapping forecast labels;
- several fixed seeds;
- save per-date predictions and loss;
- select checkpoints on validation only;
- no test evaluation every epoch;
- inspect the final lockbox once.

### 8.2 Paired comparison

For the same date:

```text
delta_loss[t] = loss_baseline[t] - loss_planet[t]
```

Report:

- mean/median improvement;
- blocked-bootstrap interval;
- fold and seed consistency;
- crisis/non-crisis and bull/bear breakdown;
- comparison against every matched null;
- parameter count and wall time.

### 8.3 Multiple testing

The hypothesis registry defines families before outcomes. Report family-wise
adjustment or false-discovery control. Exploratory results do not become
confirmatory by being relabeled after inspection.

### 8.4 Advancement gate

A planetary family advances only when it:

1. improves the same forward dates across multiple folds/seeds;
2. beats calendar/Fourier and random-smooth controls;
3. beats the distribution of matched false ephemerides;
4. retains value after feature-family knockout checks;
5. does not rely on one crisis or one timestamp convention;
6. stays within the declared capacity/training budget.

## 9. Implementation Phases and Task Cards

### Phase 0 — Governance and TFT closeout

#### `FA-GOV-001` — Create the cross-session control plane

**Status:** complete when all canonical documents exist and link correctly.

**Files:**

- this project directory;
- project README, tracker, orchestrator, registries, decision log.

**Acceptance:**

- a new agent can identify phase, next task, blockers, and evidence without chat
  history;
- no modeling task is marked complete.

#### `FA-TFT-001` — Close the native TFT feature matrix

**Steps:**

1. Inspect the running process read-only.
2. Wait for natural completion.
3. Load every `metrics.npy`, prediction array, and quantile metric.
4. Verify shapes and failed/missing cases.
5. Rank MSE/MAE and calibration separately.
6. Record a frozen native reference decision.
7. Do not choose a financial-astrology configuration from partial output.

**Acceptance:**

- all 13 cases are complete or a documented failure state exists;
- result paths and metrics are in the scorecard;
- the run was not interrupted by the project.

### Phase 1 — Freeze theory and data

#### `FA-THEORY-001` — Freeze convention and first hypothesis

**Steps:**

1. Resolve every `OPEN-*` item in `DECISIONS.md`.
2. Select the primary theory profile.
3. Set exact formula, orb/bandwidth, latency, target channel, and direction for
   enabled rules.
4. Mark confirmatory versus exploratory families.
5. Hash the resulting manifest.

**Acceptance:**

- no enabled confirmatory rule remains `DRAFT`;
- outer planets are not in a classical profile;
- disputed alternatives have distinct IDs.

#### `FA-DATA-001` — Audit source data and ephemeris generator

**Steps:**

1. Obtain the sample and generator listed in `DATA_CONTRACT.md`.
2. Validate timestamps, units, frames, missing values, and OHLC geometry.
3. Regenerate selected ephemeris rows.
4. Check sine/cosine, velocity, acceleration, and node opposition.
5. Audit Hilbert transformations and weekend/holiday joins.
6. Produce stable dataset and convention hashes.

**Acceptance:**

- all data acceptance tests pass;
- unresolved fields block implementation rather than receive guessed defaults.

#### `FA-LEAK-001` — Prove causal transformations

**Steps:**

1. Perturb future market rows and assert all earlier model inputs remain equal.
2. Fit every scaler on train only.
3. verify market rolling features at split boundaries;
4. exclude or replace noncausal market Hilbert features;
5. prove that astronomy-only future features are deterministic and label-free.

**Acceptance:**

- automated prefix-invariance tests pass.

### Phase 2 — Production loader and non-astrology baselines

#### `FA-LOAD-001` — Add named known-future loader

**Recommended files:**

```text
data_provider/planetary_market.py
data_provider/data_factory.py
utils/tft_schema.py
tests/test_planetary_market_loader.py
```

**Steps:**

1. Resolve observed, target, calendar, and astronomy columns by name.
2. Create market `x_enc`.
3. Create calendar/planet `known_enc` and `known_dec`.
4. preserve exact forecast dates and elapsed time;
5. persist feature names, transforms, and hashes;
6. fail on any future market-derived known feature.

**Acceptance:**

- a real batch shows the correct dates/names/values;
- anti-leakage tests pass across a fold boundary and holiday.

#### `FA-BASE-001` — Implement frozen baselines

**Recommended files:**

```text
exp/exp_planetary_forecasting.py
utils/financial_astrology_metrics.py
scripts/financial_astrology/
tests/test_financial_astrology_baselines.py
```

**Steps:**

1. Implement B00–B05.
2. Add AdamW, warmup, clipping, Huber/MAE, and explicit normalization mode if
   still absent from the production path.
3. remove epoch-level test evaluation for this experiment;
4. save per-date predictions and losses;
5. persist commit/config/data/split/seed hashes.

**Acceptance:**

- repeated runs with the same seed reproduce metrics within tolerance;
- no planet column is consumed by B00–B05.

### Phase 3 — Feature and null engine

#### `FA-FEAT-001` — Implement continuous and categorical state

Implement `AST-C01` through `AST-C04` and the selected parts of `AST-C07/C08`.

**Acceptance:**

- formulas match `ASTROLOGY_FEATURE_SPEC.md`;
- exact ordered names and group IDs are returned;
- circular boundary tests pass at 0/360 degrees;
- retrograde re-entry and station tests pass.

#### `FA-EVENT-001` — Implement pair/event engine

Implement conjunctions, selected drishti, eclipses, panchanga, exact event
clocks, and applying/separating.

**Acceptance:**

- numerical event fixtures match independently checked dates;
- whole-sign and degree aspects remain separate;
- Rahu/Ketu variants cannot activate accidentally.

#### `FA-NULL-001` — Implement false ephemerides

Implement every null in Section 7.3.

**Acceptance:**

- spectra/autocorrelation/marginal scale diagnostics are saved;
- null generation is deterministic by seed and manifest;
- a null never changes the target or forecast dates.

### Phase 4 — Cheap falsification

#### `FA-SCREEN-001` — Run linear and flat-TFT screen

Run:

```text
B01/B02 versus P01–P08
B04 versus flat-known P01–P08
each real block versus all matched nulls
```

Do not use the locked final period.

**Decision:**

- stop if effects are absent/unstable and data quality is sound;
- revise only with a new hypothesis version;
- advance a family only through the stated gate.

### Phase 5 — Specialized architecture

#### `FA-MEM-001` — Add fixed calendar-time response bank

**Steps:**

1. Precompute causal state for every feature/event and half-life.
2. Warm astronomy-only states with earlier ephemeris.
3. add group masks and shrinkage;
4. export effective contribution by body/group/half-life.

**Acceptance:**

- irregular-time recurrence matches a reference computation;
- Friday-to-Monday decay uses elapsed calendar days;
- future market perturbation cannot alter prior states.

#### `FA-ENC-001` — Add grouped body/aspect encoder

**Recommended files:**

```text
layers/PlanetaryKnownEncoder.py
layers/PlanetaryResponseBank.py
models/TemporalFusionTransformer.py
tests/test_planetary_known_encoder.py
```

**Steps:**

1. Add shared body encoder and body IDs.
2. add one low-rank directed pair layer;
3. pool to a fixed 8–16-channel named latent block;
4. add the zero-initialized nested residual gate;
5. support exact real/null/disabled modes;
6. log group and pair diagnostics.

**Acceptance:**

- disabled mode exactly matches the market/calendar path;
- real and null modes have identical trainable parameter counts;
- gradients reach every enabled family;
- no future market data reaches the branch.

#### `FA-MULTI-001` — Add multi-resolution grids

Implement only if `FA-MEM-001` survives.

**Steps:**

1. Build masked daily, weekly/event, and monthly tensors.
2. encode and pool each clock separately;
3. fuse with a small regularized gate;
4. compare against fixed response bank at matched capacity.

**Acceptance:**

- padding masks and timestamps are correct;
- each clock has an exact disable ablation;
- complexity earns stable forward-period improvement.

### Phase 6 — Separate extensions

#### `FA-ANCHOR-001` — Add anchored mundane model

Blocked until `AST-A02` is frozen. Do not outcome-select the anchor.

#### `FA-OUTER-001` — Add modern outer-planet model

Run Uranus/Neptune/Pluto only as `MODERN_OUTER_V1`.

**Acceptance:**

- classical score remains separately reported;
- secular/Fourier controls and local-arc caveat are mandatory.

#### `FA-PROB-001` — Add probabilistic targets

Test quantile-only and joint modes after a stable point model.

Report:

- pinball;
- empirical coverage;
- interval width;
- crossing;
- calibration by regime.

#### `FA-OHLC-001` — Add structured OHLC output

Predict:

```text
gap
body
positive upper excursion
positive lower excursion
```

Reconstruct a valid bar and assert:

```text
low <= min(open, close) <= max(open, close) <= high
```

### Phase 7 — Locked evaluation and reporting

#### `FA-LOCK-001` — Run frozen walk-forward/lockbox evaluation

**Steps:**

1. Freeze code, data, conventions, hypotheses, folds, seeds, and nulls.
2. run development folds;
3. select exactly once by the preregistered rule;
4. train through final development boundary;
5. inspect locked period once;
6. append all artifacts to the experiment registry and scorecard.

#### `FA-REPORT-001` — Classify evidence

Allowed conclusions:

- stable incremental association for the tested family;
- no detectable incremental association;
- unstable/regime-specific exploratory association;
- invalid/inconclusive due to protocol/data failure.

Anything stronger requires additional markets, longer independent history, and a
separately designed causal study.

## 10. Dependency Graph

```text
FA-GOV-001
    |
    +--> FA-TFT-001 -------------------------------+
    |                                              |
    +--> FA-THEORY-001 ----+                       |
    |                      |                       |
    +--> FA-DATA-001 --> FA-LEAK-001 --> FA-LOAD-001
                               |              |
                               +----------> FA-BASE-001
                                              |
                  FA-FEAT-001 + FA-EVENT-001 + FA-NULL-001
                               |              |
                               +--> FA-SCREEN-001
                                        |
                                stable real>null?
                                  /           \
                                no             yes
                              report      FA-MEM-001
                                             |
                                         FA-ENC-001
                                             |
                                       FA-MULTI-001
                                      /      |       \
                            FA-ANCHOR-001 FA-OUTER-001 FA-PROB/OHLC
                                      \      |       /
                                         FA-LOCK-001
                                              |
                                        FA-REPORT-001
```

## 11. Definition of Done

The project is complete only when:

1. source data and ephemeris generation are reproducible;
2. conventions and feature rules are versioned;
3. future-market leakage is ruled out by tests;
4. market/calendar baselines are strong and frozen;
5. real and false ephemerides use matched capacity;
6. classical and modern families remain distinguishable;
7. long-cycle features use phase/multiscale state without false replication
   claims;
8. walk-forward predictions and per-date losses are saved;
9. multiplicity and regime instability are reported;
10. the final claim matches the evidence category.

