# Data and Availability Contract

> Status: remediation draft. The supplied wide CSVs were audited on 2026-07-31;
> see [DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md). The contract remains
> unfrozen until raw market rows, generator provenance, and convention metadata
> pass the acceptance tests below.

## 1. Required Inputs

Received and audited:

```text
data/comprehensive_dynamic_features_nifty.csv
data/nifty50_returns.csv
data/nifty50_returns.parquet
```

The planetary table is structurally complete, but its rashi encodings are
invalid; the returns table has mixed session-date semantics. These files remain
immutable evidence and are not production inputs.

Still required:

1. raw NIFTY OHLC with true exchange session dates, or an exact retrieval and
   transformation script plus immutable source artifact;
2. a description, unit, and reference frame for every admitted planet column;
3. the PySwissEph generation script/notebook and dependency versions;
4. ephemeris files and checksum/source metadata;
5. the NIFTY adjustment/revision policy and timezone;
6. Shadbala formula/generator, location, and timestamp if that family remains
   in scope;
7. any actual Hilbert-transform code, padding, window, and boundary handling if
   the user intended a separate feature family;
8. an explanation of whether each planetary row is stamped at market open,
   close, local
   midnight, UTC midnight, or another instant.

## 2. Canonical Tables

Prefer three versioned tables rather than one ambiguous wide CSV.

### `market_daily`

One row per NIFTY trading session:

```text
session_date
session_open_timestamp_utc
session_close_timestamp_utc
open
high
low
close
optional_volume
source_revision
```

Required checks:

```text
open > 0
high > 0
low > 0
close > 0
low <= min(open, close)
high >= max(open, close)
session_date is unique and exchange-valid
```

### `ephemeris_daily`

One row per calendar timestamp, including weekends and holidays:

```text
timestamp_utc
body
frame
longitude_deg
latitude_deg
distance_au
longitude_speed_deg_per_day
latitude_speed_deg_per_day
distance_speed_au_per_day
optional acceleration fields
```

Long format is preferred for auditability. A deterministic feature builder may
later pivot it.

### `event_calendar`

One row per exact astronomical/astrological event:

```text
event_id
event_type
body_a
optional_body_b
exact_timestamp_utc
longitude_or_separation
applying_start
separating_end
convention_manifest_hash
```

Examples include ingress, station retrograde/direct, conjunction/aspect exactness,
eclipse, new/full Moon, and nakshatra boundary crossing.

## 3. Astronomical Convention Manifest

Every generated dataset must persist:

```text
generator_name
generator_version
python_version
pyswisseph_version
ephemeris_source_and_hash
geocentric_or_topocentric
observer_latitude_longitude_altitude
tropical_or_sidereal
ayanamsha_name_and_numeric_id
coordinate_frame
equinox
true_or_mean_node
apparent_or_astrometric_flags
light_time_aberration_nutation_flags
calculation_timestamp_rule
timezone_database_version
calendar_start_end
body_list
unit_table
```

The manifest is hashed. Two runs with different hashes are different datasets.

## 4. Forecast-Time Availability

Provisional decision:

```text
after the NIFTY session closes on trading date t
```

At that decision time:

- OHLC from sessions through `t` is observed;
- OHLC from `t+1` onward is unknown and may not enter any input;
- ephemeris values for any future timestamp are known;
- calendar and exchange schedule values are known;
- a feature derived from future market data is forbidden even if stored beside
  planetary columns.

Every supervised example must persist an availability record:

```text
forecast_origin_session
decision_timestamp_utc
target_session_or_end_session
target_open_timestamp_utc
target_close_timestamp_utc
label_interval_start_utc
label_interval_end_utc
maximum_market_timestamp_consumed
maximum_known_astronomy_timestamp_consumed
```

`maximum_known_astronomy_timestamp_consumed` may be after the decision time;
`maximum_market_timestamp_consumed` may not. This asymmetry is the central
known-future contract and must be visible in saved batch manifests.

Required model batch:

```text
x_enc:
    causal market-derived inputs through t

y:
    historical labels plus forecast targets

known_enc:
    calendar + planetary state on encoder timestamps

known_dec:
    calendar + planetary state on decoder/forecast timestamps

static:
    empty for single-index transit-only model
```

Future astronomical state is legitimate known-future information. Future
market-derived rolling values, normalizers, Hilbert components, or labels are
not.

### 4.1 Target-specific interval contract

Do not reuse one ambiguous “next day” join for all targets:

| Target | Decision information | Label interval | Astronomy interval summaries allowed |
|---|---|---|---|
| close-to-close return | through close of `t` | close `t` to close `t+h` | decision state; path to target open and close |
| overnight gap | through close of `t` | close `t` to open `t+1` | path from decision close to target open |
| intraday body | through close of `t` for next-session forecast | open `t+1` to close `t+1` | target open state, target-session path, target close state |
| range/realized volatility | through close of `t` | declared future sessions only | full declared label interval |

The target builder—not the neural decoder—owns these definitions. A horizon-20
endpoint target may still be one output token. Its purge/embargo length must
cover all sessions used by the label.

### 4.2 Interval-only astronomy summaries

For each declared event/aspect family, an interval feature builder may produce:

```text
state at decision, target open, and target close
minimum wrapped separation/orb
timestamp and signed offset of closest approach
number and type of exact crossings
whether an ingress or station occurs
first and last exact-event offsets
response-bank state at interval start and end
```

These summaries must be computed from ephemeris/event tables only. They must be
invariant when every future OHLC value is changed.

## 5. Calendar and Timestamp Join

The join algorithm must:

1. construct exact target trading dates from the exchange calendar;
2. construct the declared astronomical evaluation timestamp for each target;
3. derive target open/close timestamps and the complete decision-to-target
   interval from the versioned exchange calendar;
4. query/interpolate ephemeris at exact boundary instants;
5. solve declared extrema/crossings inside the interval rather than inferring
   them from trading-row endpoints;
6. retain calendar-day event history across weekends and holidays;
7. expose actual elapsed time `delta_days`, not merely row lag;
8. fail on duplicate, missing, or timezone-ambiguous keys.

Never silently forward-fill an angle across missing astronomical timestamps.
Never assume that Friday-to-Monday is a one-day physical interval.

## 6. Causal Market Features

Initial observed inputs:

```text
close_return[t] = log(close[t] / close[t-1])
gap[t]          = log(open[t] / close[t-1])
body[t]         = log(close[t] / open[t])
range[t]        = log(high[t] / low[t])
```

Any rolling feature must use values available at the decision timestamp. Code
must state whether the current session is included. Fitted transforms use only
the training portion of each walk-forward fold.

## 7. Hilbert-Transform Rule

A conventional Hilbert transform applied to the complete series is two-sided and
can leak future information.

Therefore each Hilbert-derived column must be classified:

| Source | Policy |
|---|---|
| Pure ephemeris sequence, deterministically known in advance | Allowed as a known feature, but raw ephemeris baseline is still required |
| Market OHLC or a market-derived sequence | Forbidden unless produced by a proven causal/one-sided online algorithm |
| Mixed market/planet series | Forbidden until the derivation is audited |

The first experiment should exclude Hilbert features and add them only as a named
ablation after equivalence/leakage tests.

## 8. Scaling

Use family-specific transforms fitted on the training fold:

| Family | Transform |
|---|---|
| `sin`, `cos`, binary flags | None |
| longitude | Do not scale raw longitude; use circular encoding |
| latitude, distance, signed velocity/acceleration | Robust or standard scaling fitted on train |
| event distances and times | Declared clipping plus signed `log1p` or RBF basis |
| categorical IDs | Embedding/one-hot; no numeric z-score |
| returns/gap/body/range | Robust or standard scaling fitted on train |

Persist fitted statistics, ordered feature names, and schema hash with every
checkpoint.

## 9. Split Contract

Final dates remain provisional until authoritative session dates and the
corrected market range pass source remediation. The mechanism is:

```text
expanding training window
fixed validation block
fixed forward test block
purge >= maximum forecast-label overlap
no transform fit outside training
locked final holdout accessed once
```

For the initial development study:

- minimum training history: preferably 8–10 years;
- validation/test blocks: approximately 1–2 years each;
- roll step: 1 year;
- report results per period and regime, not only pooled.

The slow-memory states may be warmed with pre-1995 ephemeris because those states
contain no market labels. Market-derived state may not be warmed using data
before the available market series unless an audited source is added.

## 10. Data Acceptance Tests

`FA-DATA-001` is complete only when automated tests prove:

1. unique, ordered trading sessions;
2. valid OHLC geometry;
3. no future market value in `known_enc` or `known_dec`;
4. exact decoder-date ephemeris values;
5. unit-circle agreement for stored longitude sine/cosine;
6. signed speed agreement with finite differences within tolerance;
7. Ketu/Rahu opposition under the selected node convention;
8. reproducibility of selected rows from the generator;
9. no full-series fitted scaler;
10. correct weekend/holiday elapsed time;
11. stable schema and convention hashes;
12. explicit handling of missing values, never silent imputation;
13. exact decision/open/close timestamps for Friday-to-Monday and holiday gaps;
14. interval event/extremum fixtures match an independently evaluated dense
    ephemeris calculation;
15. changing every future OHLC value leaves all known-future astronomy and
    every earlier sample unchanged;
16. each target's label interval and purge end are present in the fold manifest.
17. every regenerated rashi ID equals
    `floor((longitude_deg mod 360) / 30)` and all 12 boundary categories pass
    known-answer tests;
18. no rejected supplied `*_sign_sin/cos`, quarantined Shadbala, duplicate Ketu
    state, or legacy `time_delta` field enters the primary schema;
19. each market row maps to an authoritative session key without a weekend-only
    or global-day-shift repair heuristic;
20. the first admissible schema has a frozen ordered list and contains only its
    declared 37 planetary columns before ordinary controls.
