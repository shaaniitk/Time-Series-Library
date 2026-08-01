# Financial Astrology TFT Implementation Plan

> Status: canonical planning document.
>
> Detailed planning is authorized. The supplied-data audit found blocking
> rashi/date defects; data-dependent implementation is waiting for raw OHLC and
> the PySwissEph generator/convention package. No NIFTY/planet neural training
> is permitted until corrected-data gates and the native post-matrix semantic
> gate `TFT-SR09` pass.
>
> This plan targets `models/TemporalFusionTransformer.py`, not `TFT_Nixtla.py`.

## 0. Execution Boundary and Native Semantic Gate

The July 2026 ETTh1 matrix proved that advanced branches execute, but its
trained-checkpoint audit found semantic and experiment-control defects. The
root native plan now owns `TFT-SR00` through `TFT-SR09` in
[`implementation_plan.md`](../../implementation_plan.md#14-post-matrix-native-tft-semantic-repair-wave).

This project tracks that dependency through one umbrella task:

```text
FA-TFT-SEM-001 = complete only when root task TFT-SR09 is DONE
```

Execution rule:

```text
TFT-SR00..09 semantic repairs --------------------+
                                                   +--> first NIFTY training
FA-DATA-001 data/generator audit --> FA-LEAK-001 --+
                                  --> FA-LOAD-001 --+
                                  --> FA-BASE-001 --+
```

Data remediation can run while native repairs are implemented. Training cannot.
This avoids another long ETT matrix: the native repair gate uses deterministic
semantic tests, known-answer synthetic tasks, gradient audits, and a short
reproducibility micro-run. The next substantial training dataset is NIFTY.

The first NIFTY model keeps these generic switches off even after they are
repaired:

```text
FFT
explicit cross-attention
generic lag/shifted-history attention
latent-polynomial higher order
generic cross-variable graph
MoE
temporal compression
covariate reattention
```

Repairing a feature makes it honest and testable; it does not establish that a
small 1995-onward market dataset can support it.

### 0.1 Semantic-defect map for the astrology use case

| Defect | How it could corrupt the NIFTY hypothesis test | Native repair | First-study policy |
|---|---|---|---|
| hardcoded/global RNG and changed batch order | a “planet gain” could be a luckier backbone initialization or sample order | `TFT-SR01` separates seed streams and copies shared tensors | five frozen seeds; identical folds, batches, and base checkpoint |
| optional branch changes the base path immediately | enabling planets can worsen or improve forecasts before learning any planet signal | `TFT-SR02` exact zero-residual contract | disabled/null/real arms begin with identical predictions |
| row index treated as physical time | Friday-to-Monday, holidays, and coarse slow grids get false durations | `TFT-SR02/SR05/SR07` physical coordinates and masks | all effect clocks use elapsed calendar days |
| learned FFT mislabeled as selected modes | spectral diagnostics could be interpreted as planetary cycles they did not select | `TFT-SR03` separates hard selection and learned filtering | FFT off; circular phase is the primary cycle representation |
| generic cross-attention always active | apparent known-future benefit can be an unpaired extra transformation | `TFT-SR04` neutral residual, mask and metadata tests | off until a specific enrichment hypothesis earns an ablation |
| lag branch attends to a prefix, not an exact lag | a “Saturn 90-day lag” claim would not mean 90-day delayed effect | `TFT-SR05` separates prefix, exact-token, and response semantics | use calendar-time response bank, not generic lag attention |
| post-VSN polynomial called higher-order covariate interaction | latent products could be misreported as Mercury×Moon or Jupiter×Saturn | `TFT-SR06` named pre-VSN interaction interface | only preregistered body/pair and slow×fast edges |
| per-feature VSN path crashes | wide named planetary variables cannot be ablated reliably | `TFT-SR06` validation/forward/backward repair | keep off until gate passes; prefer grouped channels |
| compression contains dead decoder parameters | slow-clock claims could come from an untrained or lossy codec | `TFT-SR07` live anti-aliased K/V pooling and coordinate provenance | off at `seq_len=252`; separate coarse astronomy streams later |
| graph broadcasts one adjacency and strongly replaces inputs | a learned “planet network” may be dense, untyped, and non-neutral | `TFT-SR08` true head/topology/self-edge and residual semantics | generic graph off; later typed body/pair encoder only |
| test set evaluated every epoch | model choice can indirectly adapt to the held-out period | `TFT-SR01` validation-only experiment policy | forward test/lockbox remains unopened during fitting |

### 0.2 Astrology-aware semantic fixtures for `TFT-SR09`

The release gate includes generic fixtures, plus these domain-shaped cases. They
contain synthetic inputs and labels only; they are not claims that astrology is
true.

1. **Weekend coordinate fixture:** Friday close, Monday open, and Monday close
   retain their real elapsed-day offsets through masks, lagging, and any pooling.
2. **Known-future leakage fixture:** changing all target/future OHLC leaves past
   inputs, planetary decoder inputs, and predictions from a fixed model
   unchanged.
3. **Neutral planet fixture:** adding a constructed real/null/disabled planet
   residual with strength zero produces bitwise-identical base predictions and
   shared gradients.
4. **Circular-frequency fixture:** a single synthetic longitude frequency maps
   to the correct FFT bin when FFT selection is explicitly requested; no claim
   uses a learned-filter control point as a “selected orbit.”
5. **Exact-lag fixture:** one impulse is recoverable only at its declared token
   lag; a separate exponentially decayed calendar-time fixture verifies the
   response-bank equation.
6. **Named-pair fixture:** only the declared pair, such as a synthetic
   `Moon_phase × Saturn_state`, can solve a product target; undeclared pairs
   have zero route and zero reported contribution.
7. **Typed-graph fixture:** a planted directed body-pair graph is recovered with
   the declared self-edge/head policy and cannot mix forbidden market labels.
8. **Common-randomness fixture:** real and placebo arms have identical shared
   state hashes, ordered sample IDs, optimizer budget, and base predictions at
   initialization.

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

### 2.1 Forecast interval and event availability

A single snapshot on trading date `t` is insufficient. An event may occur after
the decision close, overnight, during a weekend, or before the next open/close.
For every forecast origin, construct an explicit interval:

```text
decision_timestamp(t)
target_open_timestamp(t+h)
target_close_timestamp(t+h)
```

Known-future astronomy may summarize that interval using preregistered fields:

```text
state at decision/open/close
minimum conjunction/aspect orb in interval
time and sign of closest approach
ingress/station/aspect/eclipse event count
first/last exact-event timestamp
pre-event and post-event response-state endpoints
```

These values are legitimate only because the ephemeris is deterministic and
known at decision time. The same interval may not summarize future OHLC.

### 2.2 Target horizon is not decoder length or effect duration

Keep three different quantities explicit:

```text
market_context_length   historical sessions consumed by the market encoder
target_horizon          future return/volatility interval being predicted
theory_response_scale   alleged calendar-time latency/persistence of a rule
```

A 60-day Saturn hypothesis does not require a 60-token decoder. The initial
implementation uses direct endpoint targets, often with `pred_len=1`, such as:

```text
return_h20[t] = log(close[t+20] / close[t])
vol_h60[t]    = realized volatility over sessions t+1..t+60
```

Every horizon has its own purge length. A later multi-horizon head may share an
encoder, but it must not silently reinterpret one decoder path as many distinct
astrological response durations.

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

Training protocol:

1. train and select the market-plus-calendar base without any planetary input;
2. copy that exact checkpoint and common-state hash into disabled, real, and
   placebo arms;
3. freeze the base during the first planetary residual screen;
4. train only the typed planet encoder, residual projection, and `alpha`;
5. compare paired predictions on identical dates and sampler order;
6. only after a family beats matched nulls, test low-learning-rate joint
   fine-tuning as a separately named arm.

This prevents a “planet model” from winning merely because its shared market
backbone received a luckier initialization or different batch order.

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

Final values are frozen after the native semantic release and data audit. The
legacy feature matrix is diagnostic input, not a selector. Starting development
values:

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
| normalization | fold-fitted market/target scaling plus explicit model mode `none` initially |
| RevIN | off for stationary returns; “off” must not silently mean window normalization |
| point/quantile | point first |
| parameter budget | preferably below 50k–100k |

Use two frozen neural profiles, not a fresh hyperparameter search for every
planet family:

```text
FA_TFT_BASE_V1
    inputs: market history + ordinary calendar/secular controls
    seq_len: 252
    target/pred_len: one direct stationary endpoint
    d_model/heads/layers: 8 / 1 / 1
    backbone: LSTM
    dropout/attention_dropout: 0.25 / 0.10
    optimizer: AdamW(lr=5e-5, weight_decay=0.005)
    loss: Huber; selection: validation MAE
    clip: 1.0; max_epochs: 50; patience: 8
    warmup: max(2 epochs, ceil(0.05 * max_epochs))
    extensions: all off

FA_PLANET_RESIDUAL_V1
    base: exact selected FA_TFT_BASE_V1 checkpoint, frozen
    planet width: 8
    residual gate alpha: exactly 0 at initialization
    trainable: planet encoder + residual projection + alpha only
    modes: disabled | matched_null | real
    optimizer budget, dates, batches, and seeds: identical for null and real
```

If `d_model=8` is unsupported by a repaired component, fail configuration; do
not silently expand it. Move to `d_model=16`, two heads, only as one declared
sensitivity arm with matched parameter controls.

Input-budget rule for the limited sample:

- run one preregistered feature family at a time;
- exclude Hilbert fields from the primary arm;
- do not expand every planet pair, aspect, harmonic, rashi, nakshatra, and
  response half-life into one flat tensor;
- cap the first raw continuous family at a documented ordered set, then pool a
  typed encoder to 8–16 named channels before adding richer relations;
- report model parameters per effective training sample and reject a profile
  that grows merely because the CSV has more derived columns.

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

The ETTh1 matrix is legacy diagnostic evidence, not an automatic NIFTY model
selector. After `TFT-SR09` and the data audit, one small reference configuration
is frozen and held identical across all financial-astrology arms. Quantile-only,
ALiBi, and covariate reattention looked promising in the single-seed screen, but
none enters the first NIFTY run by default.

### 6.1 Exact source-remediation work package

The initial inventory/schema/invariant audit is complete. When the missing
source package arrives, do not start by editing the model. Execute:

```text
WP-DATA-REMEDIATION-1
1. Preserve the audited CSVs and their recorded hashes unchanged.
2. Inventory authoritative raw OHLC/session keys and the exact retrieval or
   transformation path.
3. Rebuild close returns, gap, body, and range from exchange-valid sessions.
4. Inventory the PySwissEph generator, dependency/ephemeris versions, flags,
   timestamp/timezone, frame, ayanamsha, node, and observer conventions.
5. Reproduce selected planetary rows and resolve the observed 8.25-hour
   same-date artifact difference.
6. Regenerate rashi from audited longitude with all-boundary known-answer tests;
   do not use the rejected supplied sign pairs.
7. Keep Shadbala quarantined; audit a separate Hilbert implementation only if
   one is actually supplied.
8. Construct decision/open/close forecast intervals across holidays/weekends.
9. Produce data, convention, and ordered-schema hashes.
10. Implement prefix-invariance tests before a permanent loader.
11. Show one real loader batch with exact dates and named feature groups.
```

Minimum questions the audit must answer:

1. Is OHLC adjusted, backfilled, revised, or reconstructed, and from which
   source/timezone?
2. What instant does each planetary row represent?
3. Are positions ecliptic longitude/latitude or RA/declination, geocentric or
   topocentric, sidereal or tropical, and under which ayanamsha/flags?
4. What are the units of `r`, velocities, accelerations, and any separately
   supplied Hilbert fields?
5. Is Rahu mean or true; is Ketu derived exactly 180 degrees away?
6. Are calendar-day ephemerides available before 1995 and beyond every forecast
   horizon so astronomy-only response states can be warmed safely?
7. How are events between Friday close and Monday open/close represented?
8. Were any rules, columns, lags, anchors, or conventions already selected by
   looking at NIFTY outcomes?

The output is a data audit plus tests. It is not a trained model.

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

### 7.4 First training sequence after both gates pass

Do not begin with “all planets + all astrological rules + all TFT extensions.”
Run the following ladder on development folds only:

| Step | Run | Purpose | Advance condition |
|---:|---|---|---|
| 0 | zero return / historical volatility | irreducible naive reference | artifact and dates verified |
| 1 | Ridge market-only | transparent market baseline | reproducible across folds |
| 2 | Ridge market + Gregorian/exchange calendar + secular splines/Fourier | deterministic-time control | frozen control set |
| 3 | small TFT market + same calendar controls | nonlinear market baseline | beats or complements Ridge without leakage |
| 4 | linear raw classical ephemeris, one family at a time | cheap falsification | real family beats matched null distribution |
| 5 | flat-known TFT with the same one family | test nonlinear incremental value | stable paired gain across folds/seeds |
| 6 | fixed calendar-time response bank | test delayed/prolonged effects | beats current-state-only and null banks |
| 7 | frozen-base zero-gated typed planetary residual | isolate planet branch | real > disabled and real > null |
| 8 | named pair/slow×fast encoder | test astrological interactions | declared pairs survive knockouts/nulls |
| 9 | multi-clock daily/weekly/monthly encoder | add complexity only if earned | matched-capacity improvement |
| 10 | separate `MODERN_OUTER_V1` | exploratory outer-planet test | reported separately with secular controls |
| 11 | quantile/structured OHLC heads | uncertainty/bar geometry | stable point model already exists |

At Steps 4–8, tune shared model hyperparameters on the market/calendar baseline,
not separately on every planetary family. Each real/null/disabled triplet uses
the same base checkpoint, common initial tensors, fold dates, batch order,
optimizer budget, and seed list.

Initial confirmatory seed count is five. If seed variance is high, increase the
count using a frozen schedule rather than selecting favorable seeds.

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

- all 14 cases are complete or a documented failure state exists;
- result paths and metrics are in the scorecard;
- the run was not interrupted by the project.

The matrix closes an artifact inventory only. It does not certify extension
semantics because it used one hardcoded seed and non-paired initialization.

#### `FA-TFT-SEM-001` — Pass the post-matrix native semantic gate

**Owner of detailed work:** root tasks `TFT-SR00` through `TFT-SR09` in
[`implementation_plan.md`](../../implementation_plan.md#14-post-matrix-native-tft-semantic-repair-wave).

**Steps:**

1. Freeze the matrix as legacy semantic version 1.
2. Correct seed, initialization, sampler, and test-lifecycle controls.
3. add exact baseline-neutral extension adapters;
4. repair and rename FFT, lag, compression, graph, cross-attention, and
   interaction semantics;
5. pass known-answer synthetic and gradient-liveness tests;
6. publish semantic version 2 and migration metadata;
7. run the short deterministic semantic release micro-run.

**Acceptance:**

- root task `TFT-SR09` is recorded `DONE` with exact evidence;
- no full NIFTY/planet training occurred before the gate;
- the first astrology configuration explicitly disables every generic advanced
  extension listed in Section 0.

**Parallelism:** theory discussion may proceed while this gate is open.
`FA-DATA-001` resumes when its remediation package arrives. `FA-LOAD-001` may be
designed only after the corrected-data audit; no neural baseline or planetary
training starts until this task is complete.

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

**Current state (2026-07-31):** `WAITING_EXTERNAL`. The read-only source audit
is complete and recorded in [DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md). It
confirmed an exact all-row rashi-encoding defect, mixed one-day market-session
labels, absent generator/convention provenance, redundant Rahu/Ketu state, and
unauditable Shadbala. No production loader may be built from the current
same-date join.

**Steps:**

1. Obtain the missing raw source and generator package listed in
   `DATA_CONTRACT.md`.
2. Validate timestamps, units, frames, missing values, and OHLC geometry.
3. Regenerate selected ephemeris rows.
4. Check sine/cosine, velocity, acceleration, and node opposition.
5. Audit weekend/holiday joins and any separately supplied Hilbert transform.
6. Produce stable dataset and convention hashes.

**Remediation input before these steps can close:** authoritative raw NIFTY
OHLC/session keys; the PySwissEph generator and full calculation manifest; and,
only if retained, the Shadbala/Hilbert generators.

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

**Dependencies:** `FA-DATA-001`, `FA-LEAK-001`, frozen minimum schema, and the
coordinate/reproducibility contracts exposed through `FA-TFT-SEM-001`.

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
5. construct decision-to-target interval astronomy summaries without future
   market values;
6. persist feature names, transforms, and hashes;
7. fail on any future market-derived known feature.

**Acceptance:**

- a real batch shows the correct dates/names/values;
- anti-leakage tests pass across a fold boundary and holiday.

#### `FA-BASE-001` — Implement frozen baselines

**Dependencies:** `FA-LOAD-001`, `FA-TFT-SEM-001`, and frozen development folds.

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
4. use explicit sampler/model/extension seed streams and save shared-state and
   first-batch hashes;
5. save per-date predictions and losses;
6. persist commit/config/data/split/seed hashes.

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
    +--> FA-TFT-001 --> TFT-SR00..09 --> FA-TFT-SEM-001 --+
    |                                                       |
    +--> FA-THEORY-001 ----+                                |
    |                      |                                |
    +--> FA-DATA-001 --> FA-LEAK-001 --> FA-LOAD-001 -------+
                                                           |
                                                     FA-BASE-001
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
2. `FA-TFT-SEM-001` passed before any neural astrology training;
3. conventions and feature rules are versioned;
4. decision/target intervals and horizon semantics are explicit;
5. future-market leakage is ruled out by tests;
6. market/calendar baselines are strong and frozen;
7. real and false ephemerides use matched capacity and common initialization;
8. classical and modern families remain distinguishable;
9. long-cycle features use phase/multiscale state without false replication
   claims;
10. walk-forward predictions and per-date losses are saved;
11. multiplicity and regime instability are reported;
12. the final claim matches the evidence category.
