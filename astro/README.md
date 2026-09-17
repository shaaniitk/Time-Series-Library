# Astrology Rule Engine for the Native TFT

This package turns Vedic and Western astrology rules, written as JSON, into
forecasting inputs for the repository's native `TemporalFusionTransformer`
(TFT). It then measures whether those inputs actually improve forecasts, against
placebo versions of the same inputs.

You can use it without writing Python: rules are data. Code is only needed to add
a new *kind* of astronomical calculation (an "operator").

Contents:

1. [What "physics-informed" means here](#1-what-physics-informed-means-here)
2. [How data flows](#2-how-data-flows)
3. [Writing a ruleset](#3-writing-a-ruleset)
4. [Operator reference](#4-operator-reference)
5. [Worked examples](#5-worked-examples)
6. [Importance: declared, learned, measured](#6-importance-declared-learned-measured)
7. [Running a study](#7-running-a-study)
8. [Ephemeris data contract](#8-ephemeris-data-contract)
9. [Adding a new operator](#9-adding-a-new-operator)
10. [Validation errors you may see](#10-validation-errors-you-may-see)
11. [Limitations](#11-limitations)

---

## 1. What "physics-informed" means here

A physics-informed neural network (PINN) enforces a known law, such as a
differential equation, while it learns. Astrology does not supply a known law
linking planets to market prices, and treating a rule as a law would build the
conclusion into the model before testing it. So the design separates what is
genuinely law-like from what is a belief:

| Kind | What it covers | How it is enforced |
|---|---|---|
| **Hard constraints** (real astronomy and geometry) | Longitudes are circular (359° is next to 0°); Ketu is exactly opposite Rahu; relative rules (aspects, separations, speeds) do not change if every sidereal longitude is shifted by the same ayanamsha offset; every channel stays in a declared bounded range | Built into the operators and checked by the validator, the compiler and the tests. They cannot be violated, so no loss term is needed. |
| **Soft prior** (astrological belief) | "This rule matters about this much" | An optional training penalty pulls a learned per-rule gate toward the declared importance. Data can override it. |
| **Soft regularity** (inductive bias) | The forecast should not swing sharply for tiny changes in planetary inputs | An optional training penalty on forecast sensitivity to astrology inputs. It stops the model from memorizing specific dates through fast-moving channels. |

Neither soft term says anything about *how* markets respond to the sky. Whether
a rule helps is decided only by comparing trained models against placebo arms
(section 6).

## 2. How data flows

```text
ruleset JSON ──► schema validation ──► compiler ──► named channel matrix (daily grid)
                                          ▲                   │
ephemeris file + manifest ──► validator ──┘                   ▼
                                               sampled at market session dates
                                                              │
                         calendar features + rule channels ──►│ known-future inputs
                                                              ▼
                                      TemporalFusionTransformer (optional rule gates)
                                                              │
                              training loss (+ optional prior / regularity terms)
```

Key properties:

- **Rule channels are known-future inputs.** The ephemeris is deterministic, so a
  rule's value is available for the date being forecast. The model receives it
  through its known-future path, not as past-only history.
- **Rule channels are never z-scored.** The dataset scaler only touches market
  columns, so sin/cos pairs stay on the unit circle.
- **Compilation runs on the daily calendar grid**, so ingress timing and response
  kernels see weekends and holidays. The result is then sampled at trading
  sessions.
- **Market data cannot leak in.** The ephemeris provider the compiler reads from
  has no access to prices or returns.

Main files:

| File | Role |
|---|---|
| `astro/ephemeris/contract.py` | Manifest describing the ephemeris conventions |
| `astro/ephemeris/validator.py` | Hard checks on a supplied ephemeris |
| `astro/ephemeris/provider.py` | Frame-aware read access; synthetic ephemeris for testing |
| `astro/rules/schema.py` | Parses and validates the ruleset JSON |
| `astro/rules/operators.py` | The operator vocabulary (all astronomy math) |
| `astro/rules/registry.py` | Closed operator registry |
| `astro/rules/naming.py` | Deterministic channel names |
| `astro/compile/compiler.py` | JSON rules to named channel matrix |
| `astro/compile/response_bank.py` | Exponential decay "memory" channels |
| `astro/compile/nulls.py` | Placebo (null) arms |
| `astro/known.py` | Connects compiled channels to the TFT inputs |
| `data_provider/folds.py` | Walk-forward folds and locked holdout |
| `data_provider/data_loader.py` | `Dataset_PlanetaryMarket` (`--data planetary_market`) |
| `astro/torch/importance.py` | Learned per-rule gates |
| `astro/torch/losses.py` | Prior and regularity penalties |
| `astro/report/arm_study.py` | Paired arm comparison, verdict, family importance |
| `scripts/astro_arm_study.py` | Command-line study runner |
| `configs/astrology/ast_v1_core.json` | Example ruleset (14 rules, 11 families, 79 channels) |

## 3. Writing a ruleset

A ruleset is one JSON file with this shape:

```json
{
  "schema_version": 1,
  "ruleset_id": "my_rules_v1",
  "requires_ephemeris": { ... },
  "rules": [ { ... }, { ... } ],
  "null_arms": [ { ... } ],
  "interactions": { "pairs": [] }
}
```

Unknown keys anywhere are rejected, so a typo fails loudly instead of being
ignored.

### 3.1 Top-level fields

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | no (default `1`) | Must be `1`. |
| `ruleset_id` | yes | Lowercase letters, digits and `_`. |
| `requires_ephemeris` | no | Conventions the ephemeris must have, or compilation stops. `frame` takes a list such as `["sidereal", "tropical"]`; any other key must match a manifest field exactly, for example `"node_policy": "mean"`, `"center": "geocentric"`, `"ayanamsha": "lahiri"`. |
| `rules` | yes | Non-empty list of rules (section 3.2). |
| `null_arms` | no | Placebo versions of the whole ruleset (section 3.4). Needed for a verdict. |
| `interactions.pairs` | no | **Reserved.** Must be empty; the model does not use declared pairs yet. |

### 3.2 Rule fields

```json
{
  "rule_id": "saturn_drishti_jupiter",
  "school": "vedic",
  "family": "drishti",
  "operator": "degree_orb_drishti",
  "inputs": {"from_body": "Saturn", "to_body": "Jupiter", "frame": "sidereal"},
  "params": {"aspect_deg": [60, 180, 270], "sigma_deg": 6.0},
  "emits": ["act_a60", "act_a180", "act_a270", "applying_a180"],
  "prior": {"importance": 0.8, "l2_anchor": 0.5, "l1_weight": 0.01},
  "response_bank": {"half_life_days": [21, 90]},
  "enabled": true
}
```

| Field | Required | Meaning |
|---|---|---|
| `rule_id` | yes | Unique within the ruleset. Lowercase, digits, `_`. Becomes part of every channel name, so do not rename it casually. |
| `school` | yes | `vedic`, `western`, or `both`. A label only; it does not change the math. |
| `family` | yes | Grouping used for importance reporting and leave-one-family-out tests (for example `drishti`, `aspect`, `retrograde`). Lowercase, digits, `_`. |
| `operator` | yes | Name of a calculation from section 4. |
| `inputs` | depends on operator | Which bodies, and optionally which zodiac `frame`. |
| `params` | depends on operator | Numeric settings such as orb width. |
| `emits` | no | Subset of the operator's outputs to keep. Omit to keep all. |
| `prior` | no | Declared belief, used only if gates and the prior penalty are on (section 6). Defaults: `importance` 0.5, `l2_anchor` 0, `l1_weight` 0. |
| `response_bank` | no | Adds smoothed "memory" copies of every kept output (section 3.3). |
| `enabled` | no (default `true`) | `false` removes the rule's channels without deleting it from the file. |

**Bodies.** Use these exact names: `Sun`, `Moon`, `Mercury`, `Venus`, `Mars`,
`Jupiter`, `Saturn`, `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu`. They must also
be present in the ephemeris.

**Frames.** `sidereal` (Vedic zodiac) or `tropical` (Western zodiac). Operators
that accept `frame` default to the convention of their school (see section 4).
Operators without a `frame` input use a fixed frame; for example, rashi and
nakshatra are always sidereal.

**Channel names.** Every output becomes one named model input:

```text
astro.{school}.{family}.{rule_id}.{emit_key}          # raw value
astro.{school}.{family}.{rule_id}.{emit_key}.hl{N}    # response-bank copy
```

For example, `astro.vedic.drishti.saturn_drishti_jupiter.act_a180.hl21`. Names
are stable: adding a rule appends new channels and never renames existing ones.
Changing the layout changes the model's configuration digest, so an old
checkpoint cannot be silently loaded against a different ruleset.

**Aspect angle tags.** Angles appear in emit keys as `a{angle}`: `90` becomes
`a90`, and `22.5` becomes `a22p5`.

### 3.3 Response bank (how long an effect lasts)

`response_bank.half_life_days` adds exponentially decaying copies of each kept
output:

```text
h(t) = a^Δdays · h(t−1) + (1 − a^Δdays) · z(t),   a = 2^(−1/half_life)
```

A half-life of 21 means a signal's influence halves every 21 calendar days.
Allowed half-lives (fixed in advance so they cannot be tuned on results):

`1, 3, 5, 7, 14, 21, 30, 63, 90, 180, 365, 730, 1825, 3650, 7300, 10958`

Each half-life multiplies the rule's channel count, so add only what your
hypothesis needs. The total number of known inputs, calendar features included,
is capped at 512.

### 3.4 Null arms (placebos)

A null arm keeps the exact channel names, count and value ranges of the real
rules but breaks their alignment with the calendar. If the real rules only help
as much as a placebo does, the improvement is not evidence for astrology.

| `type` | Parameters | What it does |
|---|---|---|
| `date_shift` | `days` (integer, at least 30, shorter than the ephemeris) | Moves all rule channels together in time. Keeps relationships between channels, misaligns them with market dates. |
| `phase_randomize` | `seed` (integer) | Scrambles timing while keeping each channel's frequency content and exact value distribution. |
| `body_permute` | `seed` (integer) | Recompiles the rules with planetary tracks assigned to the wrong bodies (no body keeps its own). |

Short date shifts are weak placebos for slow planets. Jupiter and Saturn barely
move in 137 days, so a 137-day shift still correlates with the real channels.
Include a multi-year shift such as `{"type": "date_shift", "days": 1461}`.

Null arms are referred to by name: `null:date_shift:days=1461`,
`null:phase_randomize:seed=17`, `null:body_permute:seed=5`.

## 4. Operator reference

All outputs lie in `[0, 1]` or `[-1, 1]`. "Invariant" means the output does not
change when every sidereal longitude shifts by the same amount; "covariant"
means it depends on absolute zodiac position.

| Operator | School | `inputs` | `params` | Outputs (emit keys) | Default frame | Ayanamsha shift |
|---|---|---|---|---|---|---|
| `degree_orb_drishti` | Vedic | `from_body`, `to_body`, opt. `frame` | `aspect_deg` (list), `sigma_deg` | per angle: `act_a{A}`, `applying_a{A}`, `tte_a{A}` | sidereal | invariant |
| `whole_sign_drishti` | Vedic | `from_body`, `to_body` | `house_offsets` (list of 1–12) | per offset: `drishti_h{N}` | sidereal (fixed) | covariant |
| `rashi_cyclic` | Vedic | `body` | none | `rashi_sin`, `rashi_cos`, `degree_in_sign`, `to_next_sign` | sidereal (fixed) | covariant |
| `nakshatra_pada` | Vedic | `body` | none | `nak_sin`, `nak_cos`, `pada_sin`, `pada_cos`, `to_next_nakshatra` | sidereal (fixed) | covariant |
| `ingress_pulse` | Vedic | `body` | `width_days` | `pre_ingress`, `post_ingress` | sidereal (fixed) | covariant |
| `combustion` | Vedic | `body`, opt. `frame` | `width_deg` | `sun_proximity`, `sun_sep_sin`, `sun_sep_cos` | sidereal | invariant |
| `node_axis` | Vedic | `body`, opt. `frame` | `width_deg` | `axis_proximity`, `axis_sep_sin`, `axis_sep_cos` | sidereal | invariant |
| `aspect_activation` | Western | `from_body`, `to_body`, opt. `frame` | `aspect_deg` (list), `sigma_deg` | per angle: `act_a{A}`, `applying_a{A}`, `tte_a{A}` | tropical | invariant |
| `declination_parallel` | Western | `body_a`, `body_b` | `orb_deg` | `parallel`, `contraparallel`, `declination_gap` | uses declination | invariant |
| `midpoint_activation` | Western | `body_a`, `body_b`, `target_body`, opt. `frame` | `orb_deg` | `direct_midpoint`, `indirect_midpoint` | tropical | invariant |
| `retrograde_state` | Both | `body` | none | `is_retrograde`, `speed_norm` | n/a | invariant |
| `station_proximity` | Both | `body` | `speed_scale` | `station_score` | n/a | invariant |
| `wrapped_separation` | Both | `from_body`, `to_body`, opt. `frame` | none | `sep_sin`, `sep_cos`, `relative_speed` | sidereal | invariant |
| `harmonic_phase` | Both | `from_body`, `to_body`, opt. `frame` | `harmonics` (list of positive integers) | per k: `h{k}_sin`, `h{k}_cos` | sidereal | invariant |

What the outputs mean:

- **`act_a{A}`**: how close the pair is to angle `A`, as a Gaussian of the orb:
  `exp(−0.5 · (orb / sigma_deg)²)`. It is 1 at exact aspect and about 0.61 at one
  `sigma_deg` away.
- **`applying_a{A}`**: +1 while the aspect is tightening, −1 while it is separating,
  scaled by `act_a{A}`, so it fades when the aspect is far off.
- **`tte_a{A}`**: signed time to exact aspect, squashed with `tanh(days / 90)`.
  Reported as 0 near a station, where the estimate is unreliable.
- **`drishti_h{N}`**: 1 when the target sits `N` signs from the source in whole-sign
  counting (7 = opposite sign), else 0. Kept separate from the degree-orb form so
  the two schools are never averaged together.
- **`rashi_sin/cos`, `nak_sin/cos`, `pada_sin/cos`**: which sign, nakshatra or pada,
  encoded on a circle so the last sign sits next to the first.
- **`degree_in_sign`, `to_next_sign`, `to_next_nakshatra`**: position within the
  current division, as a fraction.
- **`pre_ingress`, `post_ingress`**: Gaussian closeness (in days, width
  `width_days`) to the next and the previous sign change. Kept separate so an
  anticipation effect and an aftermath effect are not forced to be symmetric.
- **`sun_proximity`, `axis_proximity`**: Gaussian closeness to the Sun, or to either
  end of the Rahu–Ketu axis. Continuous rather than a yes/no flag, because
  published combustion limits differ by planet.
- **`is_retrograde`**: 1 while longitudinal speed is negative. **`speed_norm`**:
  `tanh(speed)`.
- **`station_score`**: `exp(−(|speed| / speed_scale)²)`, near 1 when the planet
  appears to stand still.
- **`parallel`, `contraparallel`**: Gaussian closeness (width `orb_deg`) of the two
  declinations, or of one to the negative of the other.
- **`direct_midpoint`, `indirect_midpoint`**: closeness of `target_body` to the
  near midpoint of the pair, or to the point opposite it.
- **`h{k}_sin/cos`**: sin and cos of `k` times the pair's separation angle.

Traditional Vedic drishti angles, for reference: every planet aspects the 7th
(180°); Mars also 4th and 8th (90°, 210°); Jupiter also 5th and 9th (120°, 240°);
Saturn also 3rd and 10th (60°, 270°). Declare them per planet in `aspect_deg`.

## 5. Worked examples

**Vedic: Jupiter's special drishti on the Moon.**

```json
{
  "rule_id": "jupiter_drishti_moon",
  "school": "vedic",
  "family": "drishti",
  "operator": "degree_orb_drishti",
  "inputs": {"from_body": "Jupiter", "to_body": "Moon"},
  "params": {"aspect_deg": [120, 180, 240], "sigma_deg": 8.0},
  "emits": ["act_a120", "act_a180", "act_a240"],
  "prior": {"importance": 0.6, "l2_anchor": 0.3, "l1_weight": 0.01}
}
```

Produces 3 channels, such as `astro.vedic.drishti.jupiter_drishti_moon.act_a120`.

**Western: Jupiter–Saturn conjunction and square, with slow memory.**

```json
{
  "rule_id": "jupiter_saturn_hard_aspects",
  "school": "western",
  "family": "aspect",
  "operator": "aspect_activation",
  "inputs": {"from_body": "Jupiter", "to_body": "Saturn", "frame": "tropical"},
  "params": {"aspect_deg": [0, 90], "sigma_deg": 6.0},
  "emits": ["act_a0", "act_a90", "applying_a0"],
  "response_bank": {"half_life_days": [90, 365]}
}
```

Produces 3 × (1 raw + 2 half-lives) = 9 channels.

**Both schools: Mercury retrograde and its stations.**

```json
{
  "rule_id": "mercury_retrograde",
  "school": "both",
  "family": "retrograde",
  "operator": "retrograde_state",
  "inputs": {"body": "Mercury"},
  "params": {},
  "response_bank": {"half_life_days": [7]}
},
{
  "rule_id": "mercury_station",
  "school": "both",
  "family": "retrograde",
  "operator": "station_proximity",
  "inputs": {"body": "Mercury"},
  "params": {"speed_scale": 0.5}
}
```

**Western: Mars–Venus declination parallel.**

```json
{
  "rule_id": "mars_venus_declination",
  "school": "western",
  "family": "declination",
  "operator": "declination_parallel",
  "inputs": {"body_a": "Mars", "body_b": "Venus"},
  "params": {"orb_deg": 1.0}
}
```

To check a ruleset without training anything:

```bash
PYTHONPATH=. ./ai_env/bin/python -c "
from astro.rules.schema import load_ruleset
from astro.compile.compiler import compile_ruleset
from astro.ephemeris.provider import synthetic_provider
rules = load_ruleset('configs/astrology/my_rules_v1.json')
compiled = compile_ruleset(rules, synthetic_provider(periods=4000))
print(compiled.channel_count, 'channels')
print('\n'.join(compiled.names))
"
```

## 6. Importance: declared, learned, measured

Three levels, in increasing order of evidential weight:

| Level | Where it comes from | Counts as evidence? |
|---|---|---|
| **Declared** | `prior.importance` in the JSON | No. It is your belief. |
| **Learned** | One gate per rule, trained with the model (`--tft_astro_rule_gates`) | No. A diagnostic. |
| **Measured** | Retraining with rules removed or replaced by placebos, compared date by date | Yes. |

### Learned gates

With `--tft_astro_rule_gates`, each rule gets one number that multiplies all of
its input channels before the model sees them. Calendar features are not gated.

- **Gates start at exactly 0.** An untrained model sees what the zero arm sees
  (all astrology inputs zeroed), so astrology has to earn influence through the
  training signal.
- After training, `astro_importance.json` in the checkpoint folder lists
  `rule_gates` (value per rule) and `family_mean_abs_gate` (average absolute gate
  per family). Magnitude is what matters; sign is not interpretable at family
  level.

### Prior penalty (optional)

Enabled by `--astro_prior_coeff > 0` (requires gates):

```text
prior_coeff × Σ_rules [ l2_anchor · (gate − importance)² + l1_weight · |gate| ]
```

- `importance`: the gate value your belief points toward.
- `l2_anchor`: how strongly to pull toward it. 0 means no pull.
- `l1_weight`: pull toward zero, so rules are switched off unless data supports
  them.

Keep coefficients small. The penalty is meant to break ties, not decide outcomes.

### Regularity penalty (optional)

Enabled by `--astro_regularity_coeff > 0`:

```text
regularity_coeff × mean[(f(x + εu) − f(x − εu))²] / (2ε)²
```

Here `u` is a random unit direction over the astrology inputs only, and `ε` is
0.05. Both nudged forecasts use identical dropout, so the difference reflects the
astrology inputs alone. Because the nudge happens before the gates, the penalty
is near zero while gates are small and grows as the model starts relying on
astrology.

### Measured importance

- **Does the ruleset help at all?** Train the `real`, `zero` and `null:*` arms and
  read the verdict (section 7).
- **Which family matters?** Add `--leave_one_family_out`. For each family, a
  `zero:<family>` arm is retrained with only that family's channels zeroed. All
  arms keep the same inputs and parameter count, so with the same seed they share
  initialization and data order. The report gives `mae_increase` with a 95%
  interval; a positive value means removing the family made forecasts worse.

## 7. Running a study

### 7.1 Check the machinery on synthetic data first

This plants a known effect (returns shift while Mercury is retrograde) and checks
that the pipeline finds it:

```bash
PYTHONPATH=. ./ai_env/bin/python scripts/astro_arm_study.py \
    --synthetic --folds F1 --epochs 10 --patience 4 --work_dir ./astro_runs/synthetic
```

Reference result (fold F1, one seed), validation MAE: real 0.7827, best null
0.7997, zero 0.8396, and the verdict was PROCEED. If a real-data run later says
STOP, this check shows the pipeline can detect an effect when one exists.

### 7.2 Real study

```bash
PYTHONPATH=. ./ai_env/bin/python scripts/astro_arm_study.py \
    --root_path ./data --data_path nifty_market.csv \
    --ephemeris ./data/ephemeris.parquet --manifest ./data/ephemeris_manifest.json \
    --ruleset configs/astrology/ast_v1_core.json \
    --folds F1 F2 F3 F4 --seeds 2 3 4 \
    --work_dir ./astro_runs/nifty_v1
```

Optional additions: `--leave_one_family_out`, `--rule_gates`,
`--prior_coeff 0.01`, `--regularity_coeff 0.01`.

The market CSV needs a `date` column and the market columns, with the target last
(default target `log_Close`, 4 market columns). Adjust `--enc_in` and
`--target_pos` for other layouts.

**Verdict rule (fixed before running).** PROCEED only if the `real` arm has lower
validation MAE than both the `zero` arm and the best null arm in at least 3 of 4
folds (majority of seeds per fold). Otherwise STOP. The report also gives mean
per-date improvements with 95% intervals from a block bootstrap (20-session
blocks, which respects day-to-day dependence). Everything is written to
`verdict.json`, plus one `pred_*.csv` per arm.

**Folds** (`data_provider/folds.py`, from the frozen protocol):

| Fold | Training ends | Evaluation |
|---|---|---|
| F1 | 2007-12-31 | 2008-04-01 → 2010-12-31 |
| F2 | 2010-12-31 | 2011-04-01 → 2013-12-31 |
| F3 | 2013-12-31 | 2014-04-01 → 2016-12-31 |
| F4 | 2016-12-31 | 2017-04-01 → 2020-03-31 |
| HOLDOUT | 2020-03-31 | 2020-07-01 → end of data |

At least 63 trading sessions always separate the last training target from the
first evaluation target. `HOLDOUT` is refused unless `--astro_unlock_holdout` is
passed; open it once per protocol version.

### 7.3 Using `run.py` directly

| Flag | Meaning |
|---|---|
| `--data planetary_market` | Use the rule-aware dataset |
| `--astro_ruleset PATH` | Ruleset JSON |
| `--astro_ephemeris_path PATH` | Ephemeris `.csv` or `.parquet` |
| `--astro_ephemeris_manifest PATH` | Manifest JSON |
| `--astro_arm NAME` | `real` (default), `zero`, `zero:<family>`, or a null arm name |
| `--astro_fold NAME` | `F1`–`F4` or `HOLDOUT` |
| `--astro_unlock_holdout` | Allow `HOLDOUT` |
| `--tft_astro_rule_gates` | Learn one gate per rule |
| `--astro_prior_coeff X` | Prior penalty weight (0 disables it entirely) |
| `--astro_regularity_coeff X` | Regularity penalty weight (0 disables it entirely) |

The known-feature settings (`tft_allow_custom_known`, `tft_known_len`,
`tft_known_feature_names`) are derived automatically; do not set them by hand.
Keep the default `--embed timeF`.

## 8. Ephemeris data contract

This is the specification for whoever produces the planetary data.

**Table** (`.csv` or `.parquet`):

- `timestamp`: UTC, strictly increasing, **one row per UTC day**, covering every
  market date used plus the forecast horizon.
- Planets (`Sun` … `Pluto`): `{Body}_lon`, `{Body}_lat`, `{Body}_speed`,
  `{Body}_dist`, `{Body}_decl`.
- Nodes (`Rahu`, `Ketu`): `{Body}_lon`, `{Body}_speed`.
- `ayanamsha`: optional. Required to serve both sidereal and tropical frames.

**Units:** longitude, latitude, declination and ayanamsha in degrees
(longitude and ayanamsha in `[0, 360)`); speed in degrees per day (negative means
retrograde); distance in AU.

**Manifest** (JSON):

```json
{
  "schema_version": 1,
  "source_repo": "https://…/ephemeris-generator",
  "source_commit": "abc1234",
  "stored_frame": "tropical",
  "ayanamsha": "lahiri",
  "has_ayanamsha_column": true,
  "node_policy": "mean",
  "center": "geocentric",
  "bodies": ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
             "Uranus", "Neptune", "Pluto", "Rahu", "Ketu"],
  "coverage_start": "1995-01-01T00:00:00+00:00",
  "coverage_end": "2045-12-31T00:00:00+00:00",
  "row_count": 18627
}
```

`stored_frame` is the frame of the `_lon` columns. For `"center": "topocentric"`,
also give `topocentric_lat_deg` and `topocentric_lon_deg`
(`topocentric_alt_m` is optional).

**The validator rejects** missing columns, non-finite values, unsorted or
duplicate timestamps, longitudes outside `[0, 360)`, latitude or declination
beyond ±90°, non-positive distances, speeds above 20°/day, Ketu more than 1e-6°
away from exactly opposite Rahu, a prograde Rahu when `node_policy` is `mean`,
longitude jumps faster than any body can move, a mismatch between the ayanamsha
column and `has_ayanamsha_column`, and a manifest whose `row_count` or coverage
disagrees with the table. The error names the column and row.

The existing `data/comprehensive_dynamic_features_nifty.csv` does **not** follow
this contract. It stores `_sin/_cos` instead of longitudes, names nodes
`Mean Rahu`, and its timestamp and conventions are undocumented. It needs
regenerating or converting.

## 9. Adding a new operator

Only needed for a genuinely new calculation. New combinations of bodies, angles,
orbs, half-lives or schools are just JSON.

1. Add a function to `astro/rules/operators.py`:

   ```python
   @operator(
       name="my_operator",
       version=1,
       inputs=("body",),
       optional_inputs=("frame",),
       params=("width_deg",),
       emits=("my_signal",),
       ranges={"my_signal": (0.0, 1.0)},
       invariance="invariant",   # or "covariant"
   )
   def my_operator(provider, inputs, params):
       """One sentence on what this measures."""
       longitude = provider.longitude(inputs["body"], inputs.get("frame", "sidereal"))
       ...
       return {"my_signal": values}   # one array per emit, length = ephemeris rows
   ```

   If outputs depend on params (like one output per aspect angle), pass
   `emits_resolver=` instead of `emits`/`ranges`.

2. Rules for operators:
   - Use only the `provider` (ephemeris). Never read market data.
   - Return values inside the declared ranges. The compiler rejects anything
     outside them.
   - Be deterministic: same inputs, identical output.
   - Declare `invariance` truthfully. The test suite checks it by shifting the
     ayanamsha.

3. Add a case to `OPERATOR_CASES` in `tests/test_astro_rules.py`. That applies the
   determinism, range, completeness and invariance tests to the new operator; the
   suite fails if an operator has no case.

4. Bump `version` whenever an existing operator's math changes. Versions are
   recorded in the compiled manifest.

## 10. Validation errors you may see

| Message contains | Cause | Fix |
|---|---|---|
| `Unknown operator` | Operator name misspelled or not registered | Use a name from section 4 |
| `requires inputs … which are absent` / `unknown inputs` | Wrong keys in `inputs` | Match the operator's `inputs` column |
| `requires params` / `unknown params` | Wrong keys in `params` | Match the operator's `params` column |
| `does not produce` | `emits` lists a key the operator does not output | Check emit keys, including angle tags such as `a180` |
| `must match ^[a-z0-9_]+$` | Uppercase, spaces or hyphens in an id | Use lowercase, digits, `_` |
| `not in the preregistered grid` | Unsupported half-life | Pick from section 3.3 |
| `Duplicate rule_id` | Two rules share an id | Rename one |
| `interactions.pairs is reserved` | Non-empty `pairs` | Leave it empty |
| `requires … but the ephemeris declares` | `requires_ephemeris` does not match the manifest | Fix the requirement or the data |
| `is not present in this ephemeris` | Body missing from the data | Add it to the data or drop the rule |
| `Frame … is unavailable` | Asking for sidereal without an ayanamsha column, or the reverse | Supply the column or use the stored frame |
| `above tft_known_max_channels` | Too many channels | Trim `emits` or half-lives, or disable rules |
| `Ephemeris does not cover market session` | Ephemeris dates too short | Extend coverage |
| `Unknown astro_arm` | Arm name typo | Use `real`, `zero`, `zero:<family>` or a declared null arm name |

## 11. Limitations

- **No real-data result yet.** The pipeline has only been checked on synthetic
  data with a planted effect. The NIFTY study waits on a contract-compliant
  ephemeris and a market table rebuilt from raw OHLC.
- **Declared interaction pairs are not implemented.** The `interactions` field is
  reserved.
- **Learned gates are diagnostics.** They are not causal evidence; only the paired
  arm comparisons are.
- **The synthetic ephemeris is too regular.** Its planets move in perfectly regular
  cycles, which makes date-shift placebos stronger there than they will be on real
  ephemeris data. It is for testing the pipeline, not for astronomy.
- **The verdict rule adapts to fewer folds.** It requires `min(3, number of folds)`
  winning folds, so a single-fold run needs only one. Use all four folds for a
  real decision.
