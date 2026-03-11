# Financial-Astrology Source Data Audit

> Audit task: `FA-DATA-001` / root alias `AST-D01`
>
> Audit date: 2026-07-31
>
> State: `WAITING_EXTERNAL — BLOCKING SEMANTIC DEFECTS CONFIRMED`
>
> This was a read-only audit. No source row was changed and no training was
> launched.

## 1. Executive Verdict

The continuous planetary coordinates are numerically coherent and potentially
useful, but the supplied files are **not training-ready**.

Three issues would directly invalidate an astrology experiment:

1. every stored `*_sign_sin/cos` pair implements a provably incorrect rashi
   transformation;
2. the NIFTY return dates use mixed session-label conventions, shifting entire
   blocks by one calendar day;
3. the PySwissEph calculation timestamp, frame, ayanamsha, flags, location, and
   Shadbala formula are not present in the repository.

The first loader must therefore be built from a corrected, timestamped market
session table and a reproducible ephemeris generator. It must not simply
inner-join the two current CSVs on their displayed date.

## 2. Audited Artifacts

The actual planetary filename uses singular `dynamic`:

```text
data/comprehensive_dynamic_features_nifty.csv
```

| Artifact | Shape | Date range | SHA-256 |
|---|---:|---|---|
| `comprehensive_dynamic_features_nifty.csv` | 18,251 × 88 | 1995-01-01 to 2044-12-19 | `3036dc79790fed35d3a5f9cf2b4e7726bb00eec7915b54f839aba4a99671fa6d` |
| `nifty50_returns.csv` | 7,109 × 6 | 1996-11-05 to 2025-06-09 | `6765711ed03964f06fec725683e28bc1e83461c36feba09181c080d9a05485b6` |
| `nifty50_returns.parquet` | 7,109 × 5 plus date index | same | `9798dd12442edeab0e57a88d0de744c8b9da1eba2d4582c88d208e79af33cf1c` |

The CSV and Parquet return values agree to approximately `1e-16`. In the
Parquet artifact, `Date` is the index rather than a value column.

All artifacts are structurally clean:

- dates are sorted and unique;
- no numeric cell is missing or infinite;
- the planetary table has every calendar day in its declared range;
- all 7,109 displayed return dates exist in the planetary table.

Structural joinability is not semantic alignment.

## 3. Planetary Schema

The 87 numeric columns contain:

| Family | Count |
|---|---:|
| longitude `sin/cos` pairs for 12 bodies/nodes | 24 |
| longitude speeds | 12 |
| purported rashi `sign_sin/cos` pairs | 24 |
| distances for Sun through Pluto | 10 |
| latitudes for Sun through Pluto | 10 |
| Shadbala for the seven traditional visible grahas | 7 |

The nodes have longitude, speed, and purported sign fields only. Despite the
user's broad description of two spherical angles, latitude is stored as one raw
degree-like scalar; it does not have a `sin/cos` pair.

## 4. Confirmed Rashi-Encoding Defect

For body `b`, reconstruct:

```text
longitude_deg = degrees(atan2(b_sin, b_cos)) mod 360
stored_sign_deg = degrees(atan2(b_sign_sin, b_sign_cos)) mod 360
```

For all 12 bodies/nodes and all 18,251 rows, with zero exceptions, the file
obeys:

```text
stored_sign_deg = 30 * max(floor(longitude_deg / 30) - 1, 0)
```

This is not a valid 12-rashi encoding:

- longitudes from `0°` through just below `60°` collapse into one category;
- every true sign from index 2 through 11 is shifted backward one;
- sign index 11 (`330°`) never appears;
- fast bodies expose only 11 stored categories;
- Rahu/Ketu rashi opposition is broken even though their continuous longitude
  opposition is exact.

The 30-row artifact `data/comprehensive_dynamic_features.csv` stores the
correct zero-based integer `floor(longitude/30)`. The defect was introduced in a
later categorical/circular transformation.

Required policy:

```text
reject every existing *_sign_sin and *_sign_cos column
```

After the coordinate convention is confirmed, regenerate with:

```text
rashi_id = floor((longitude_deg mod 360) / 30)       # 0..11
rashi_sin = sin(2*pi*rashi_id/12)
rashi_cos = cos(2*pi*rashi_id/12)
```

Tests must cover both sides of every boundary, retrograde re-entry, and all 12
categories.

## 5. Continuous Coordinate Integrity

The direct longitude pairs are numerically strong:

```text
abs(sin^2 + cos^2 - 1) <= 4.44e-16
```

Unwrapped one-day angular displacement agrees closely with the stored midpoint
speed. This is strong evidence that the fields are ordinary circular encodings
of ephemeris longitude.

Observed ranges are physically plausible for geocentric ecliptic-style output:

- Moon distance: approximately `0.00238`–`0.00272`;
- Sun distance: approximately `0.983`–`1.017`;
- Mercury, Venus, Mars, Jupiter, and Saturn contain coherent negative-speed
  retrograde episodes;
- outer-planet speeds and distances evolve smoothly.

Likely units are degrees/day, degrees, and AU, but inferred units are not a
substitute for generator metadata.

The January Sun longitudes look compatible with a sidereal representation, but
the CSV alone cannot establish the ayanamsha or calculation flags. Do not label
the data “Lahiri” or another convention without the generator.

## 6. These Are Not Hilbert Features

No column name contains `hilbert`, `analytic`, `amplitude`, `instantaneous`, or
an equivalent transform field. Direct unit-circle geometry and speed agreement
show that `*_sin/cos` are trigonometric encodings of longitude, not a Hilbert
analytic signal.

This is useful: the primary study does not need a Hilbert-leakage arm. If a
separate Hilbert transform exists outside these files, it must be supplied and
audited independently. Any market-derived two-sided Hilbert transform remains
forbidden.

## 7. Unresolved Ephemeris Timestamp

No PySwissEph generator, notebook, ephemeris manifest, or dependency declaration
exists in the current tree or Git history.

There is also direct evidence that local planetary artifacts use different
times for the same displayed date. Across every day in the 30-row January 2023
artifact and all tested bodies, `comprehensive_dynamic_features.csv` is exactly
about `8.25` hours of angular motion later than
`comprehensive_dynamic_features_nifty.csv`:

| Body | Median implied offset |
|---|---:|
| Sun | 8.2500 h |
| Moon | 8.2498 h |
| Mars | 8.2502 h |
| Jupiter | 8.2500 h |
| Mean Rahu | 8.2500 h |

The displayed date is therefore not enough to identify the instant. This is
material for the Moon and for station/aspect/ingress boundaries.

Required metadata:

```text
PySwissEph version
ephemeris source/files and hashes
calc_ut/calc flags
geocentric or topocentric
tropical or sidereal
sidereal mode / ayanamsha
mean or true node
apparent/astrometric, nutation, aberration, and light-time flags
calculation time and timezone for each row
observer/location if any
```

## 8. Rahu and Ketu

The continuous node geometry is internally exact:

- Ketu `sin/cos` is antipodal to Mean Rahu within about `1.5e-15`;
- both speeds are identical;
- the mean-node speed is always retrograde and nearly constant around
  `-0.052992°/day`.

Model policy:

1. retain one continuous node axis;
2. derive the antipode exactly;
3. use separate semantic role IDs only when a theory distinguishes Rahu and
   Ketu;
4. do not feed both duplicated speeds;
5. do not standardize the nearly constant numerical speed noise into a large
   artificial signal.

## 9. Shadbala Quarantine

The seven Shadbala columns are finite and variable, but are not currently
auditable:

- all values are quantized to exactly `0.01`;
- daily changes can be large;
- Saturn reaches `-0.12` on 2001-05-07;
- no component formula, location, timestamp, chart convention, or generator is
  supplied;
- future availability is not proven.

Shadbala often contains time- and location-dependent components. It is not
enough that the values sit beside deterministic ephemerides.

Required policy:

```text
exclude all *_shadbala from the primary and first exploratory runs
```

They may return only after independent row reproduction and a frozen formula
manifest.

## 10. Confirmed Market-Date Defect

`nifty50_returns.csv` contains 120 weekend-labelled rows:

```text
32 Saturdays
88 Sundays
```

Some isolated older weekend dates may be real special or Muhurat sessions. Two
blocks are unambiguously inconsistent with ordinary NSE session labels:

```text
calendar year 2023:             regular Sundays, zero Fridays
October 2024 through June 2025: regular Sundays, zero Fridays
```

January–September 2024 returns to a normal Monday–Friday pattern, proving that
the file mixes date conventions rather than applying one declared global rule.
A secondary anomaly check used Yahoo's public `^NSEI` chart JSON for
2007-01-01 through 2025-07-01. Of 4,388 locally comparable rows, 4,288 matched a
Yahoo-derived four-return vector within `1e-7`: 3,876 on the displayed date and
412 on the following calendar date. All 412 shifted cases also had unique exact
four-field fingerprints after rounding to ten decimals. For example, the row
labelled Sunday 2025-05-11 matches the OHLC changes for Monday 2025-05-12 to
numerical precision.

```text
https://query2.finance.yahoo.com/v8/finance/chart/%5ENSEI
  ?period1=1167609600&period2=1751328000&interval=1d&events=history
```

Yahoo is a secondary, revisable source and omitted or disagreed on 100
comparable rows, including boundary/special-session cases. This check is an
anomaly detector, not production provenance and not a repair map.

Consequences:

- a raw same-date join assigns the wrong planetary day to affected sessions;
- moving only weekend rows is insufficient because ordinary weekday labels in
  each defective block are also shifted;
- globally adding one day is invalid because other blocks and genuine special
  sessions are correctly labelled;
- the Moon may move roughly 12–15 degrees during the error.

Required repair:

1. obtain raw NIFTY OHLC with an authoritative session date;
2. validate against the exchange calendar, holidays, and special sessions;
3. recompute the returns;
4. never heuristically relabel the current derived file in place.

The official [NSE historical index data page](https://www.nseindia.com/reports-indices-historical-index-data)
or another frozen, documented source should supply the canonical market rows.

## 11. Return Semantics

The supplied columns are consistent with the user's declaration:

```text
log_Open[t]  = log(Open[t]  / Open[previous session])
log_High[t]  = log(High[t]  / High[previous session])
log_Low[t]   = log(Low[t]   / Low[previous session])
log_Close[t] = log(Close[t] / Close[previous session])
```

These are same-field session returns. They are not the causal candle channels
declared in the project plan:

```text
overnight_gap[t] = log(Open[t] / Close[t-1])
intraday_body[t] = log(Close[t] / Open[t])
range[t]         = log(High[t] / Low[t])
```

Therefore:

- `log_Close[t+1]` is a suitable provisional primary target after date repair;
- `log_Open` is open-to-open, not an overnight gap;
- `log_High` and `log_Low` are not range geometry;
- raw OHLC is required for valid gap/body/range targets and later structured
  OHLC reconstruction.

The first row's prior price is not present, so even the stated formulas cannot
be fully reproduced from the repository alone.

## 12. `time_delta` Is Not Physical Elapsed Time

For every row after the first:

```text
time_delta = calendar_date_gap_days / 5
```

The first value is arbitrarily `0.2`. Replace it with separately named fields:

```text
elapsed_calendar_days
elapsed_trading_sessions
```

Calendar-time planetary response kernels consume the first in actual days, not
an arbitrary division by five.

## 13. Long-Memory Warm-Up Limit

The planetary file begins only 674 calendar days before the first return row.
That is adequate for short fast-body history, but not for long response states.

The unforgotten initialization fraction after 674 days is approximately:

```text
half-life 365 days:    27.8%
half-life 730 days:    52.7%
half-life 1,825 days:  77.4%
half-life 10,958 days: 95.8%
```

Because earlier ephemeris contains no market labels, the corrected generator
should safely create a much longer pre-1995 astronomy-only warm-up. A learned
model must not interpret an arbitrary zero initialization as a Saturn or
outer-planet state.

## 14. Feature Admission Table

| Feature family | Current decision | Reason |
|---|---|---|
| Sun–Saturn longitude `sin/cos` | `PROVISIONAL AFTER TIMESTAMP CONFIRMATION` | Unit-circle and speed checks pass |
| Sun–Saturn speed/distance/latitude | `PROVISIONAL AFTER UNIT CONFIRMATION` | Continuous and physically plausible |
| stored `*_sign_sin/cos` | `REJECT` | Exact off-by-one/clipping defect |
| corrected rashi derived from longitude | `LATER NAMED ARM` | Must be regenerated and boundary-tested |
| Mean Rahu longitude axis | `PROVISIONAL` | Exact coherent mean-node geometry |
| Ketu numeric duplicate | `DERIVE, DO NOT LEARN INDEPENDENTLY` | Exact antipode |
| node speed duplicated twice | `DROP` | Constant/redundant |
| Shadbala | `QUARANTINE` | Formula/time/location unavailable |
| Uranus/Neptune/Pluto | `SEPARATE MODERN_OUTER ARM` | Too few target cycles for a core classical claim |
| Hilbert features | `ABSENT` | None identifiable in supplied schema |
| `time_delta` | `REBUILD` | Gap divided by five and based on defective dates |
| `log_Close` | `PRIMARY TARGET AFTER REBUILD` | Stationary close-to-close target |
| `log_Open/High/Low` | `SECONDARY/OBSERVED AFTER REBUILD` | Not gap/body/range definitions |

## 15. First Admissible Feature Slice

After timestamps and market dates are repaired, the cheap first family is:

```text
CLASSICAL_CONTINUOUS_V1
Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn:
    longitude_sin
    longitude_cos
    signed_longitude_speed
    distance
    latitude

Mean node axis:
    longitude_sin
    longitude_cos

Excluded:
    existing rashi pairs
    Shadbala
    Ketu duplicate numeric state
    Uranus/Neptune/Pluto
    generic TFT advanced extensions
```

This is 37 transparent physical columns before ordinary calendar controls. It
is tested first in Ridge, then in the small flat-known TFT, and always against
matched smooth/null ephemerides. Corrected rashi, stations/retrograde loops,
aspects, response banks, outer planets, and Shadbala enter as separate arms.

## 16. Canonical Loader Record

The rebuilt dataset needs one unambiguous supervised record:

```text
origin_session_date
origin_close_timestamp_Asia_Kolkata
target_session_date
target_open_timestamp_Asia_Kolkata
target_close_timestamp_Asia_Kolkata
target_close_return
target_open_gap
target_intraday_body
target_range
elapsed_calendar_days
elapsed_trading_sessions
market_history_through_origin_only
known_astronomy_over_decision_to_target_interval
```

Do not reduce calendar-daily astronomy to trading dates before computing event
crossings and response states; doing so would discard weekend motion.

## 17. Required External Inputs

`FA-DATA-001` cannot close until the following are supplied:

1. raw NIFTY OHLC, or the exact retrieval/transformation script and immutable
   source artifact, with true exchange session dates;
2. the PySwissEph generator and version/ephemeris files;
3. calculation time/timezone and coordinate/ayanamsha/flag manifest;
4. the Shadbala generator and complete formula/location/time convention if that
   family is to remain in scope;
5. any separate Hilbert-generation code if the user intended features not
   present in the audited CSV.

## 18. Current Gate State

```text
FA-DATA-001: WAITING_EXTERNAL
FA-LEAK-001: NOT_STARTED
FA-LOAD-001: NOT_STARTED
NIFTY neural training: NOT AUTHORIZED
```

The next safe action is to receive the missing source/generator package, rebuild
canonical dates and features without overwriting the supplied artifacts, and
then convert the invariants in this report into automated acceptance tests.
