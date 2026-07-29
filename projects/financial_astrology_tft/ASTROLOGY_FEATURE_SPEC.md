# Astrology Feature Specification

> Status: architecture draft. Exact constants and disputed conventions are not
> frozen.

## 1. Circular Geometry

Convert degrees to radians:

```text
theta = pi * longitude_deg / 180
```

Base longitude representation:

```text
lon_sin = sin(theta)
lon_cos = cos(theta)
```

Wrapped directed separation from body `i` to body `j`:

```text
delta_ij = atan2(
    sin(theta_j - theta_i),
    cos(theta_j - theta_i)
)
```

Pair representation:

```text
sin(k * delta_ij), cos(k * delta_ij)
```

for a preregistered small harmonic set `k`. Do not tune a large harmonic bank on
the final test.

## 2. Rashi

For sidereal longitude `L` in `[0, 360)`:

```text
rashi_id          = floor(L / 30)
degree_in_rashi   = L - 30 * rashi_id
distance_to_left  = degree_in_rashi
distance_to_right = 30 - degree_in_rashi
```

Store:

- rashi embedding or one-hot;
- circular degree-within-sign;
- sign lord;
- modality and element lookup;
- days since previous ingress;
- days until next ingress;
- applying/separating boundary flag;
- ingress event basis.

All lookup tables are versioned. Retrograde re-entry is an event, not erased as
duplicate categorical state.

## 3. Nakshatra and Pada

For the 27-mansion convention:

```text
nakshatra_width = 360 / 27
nakshatra_id    = floor(L / nakshatra_width)
pada_id         = floor((L mod nakshatra_width) / (nakshatra_width / 4))
```

The 28-mansion variant, if used, receives a separate manifest and formula. Store
boundary distance and time-to-crossing so a category change is not the only
signal.

## 4. Retrograde and Station

Let signed longitude velocity be `v`.

```text
retrograde = 1[v < 0]
station_score(scale) = exp(-(abs(v) / scale)^2)
```

Event solver outputs:

```text
days_to_station
days_since_station
station_type: retrograde | direct
applying_to_station
separating_from_station
retrograde_loop_phase
```

The loop phase distinguishes pre-shadow, first crossing, retrograde crossing,
direct crossing, and post-shadow where such a convention is enabled.

## 5. Aspects and Applying/Separating

For target aspect angle `phi`:

```text
orb = abs(wrap(delta_ij - phi))
activation = exp(-0.5 * (orb / sigma_phi)^2)
relative_speed = velocity_j - velocity_i
```

Use signed time-to-exactness from a numerical ephemeris event solver where
possible. The linear estimate `-wrapped_error / relative_speed` is only a local
diagnostic and fails near stations.

Represent whole-sign Parashari drishti separately from degree-orb activation:

```text
directed_whole_sign_edge
continuous_degree_edge
```

This prevents two different schools from being silently averaged.

## 6. Conjunction and Graha-Yuddha

Ordinary conjunction basis:

```text
same_rashi
wrapped_gap
smooth_gap_kernel
applying_or_separating
signed_days_to_closest_approach
minimum_gap
```

Strict graha-yuddha rules and optical-disc criteria require a dedicated frozen
table. Until that table is selected, the model may use ordinary conjunction
geometry but must not label it graha-yuddha.

## 7. Dignity and Dispositor

Candidate categorical fields:

```text
exalted
moolatrikona
own_sign
great_friend
friend
neutral
enemy
great_enemy
debilitated
```

Candidate continuous fields:

```text
circular distance from exact exaltation
circular distance from exact debilitation
```

Dispositor chains are capped to a small fixed depth and include a cycle flag.
Node dignity is absent from the core profile.

## 8. Combustion

Primary fields:

```text
sun_separation
relative_speed_to_sun
applying_or_separating
smooth proximity bases at preregistered widths
```

Binary combustion flags are secondary because published thresholds differ by
planet and retrograde state.

## 9. Nodes and Eclipses

Choose either mean or true node in the convention manifest.

```text
ketu_longitude = wrap(rahu_longitude + 180 degrees)
```

Candidate eclipse features:

```text
sun_distance_to_nodal_axis
moon_distance_to_nodal_axis
sun_moon_phase
days_to_or_since_eclipse
eclipse_type
magnitude
local_visibility
rashi
nakshatra
```

Time-to-event is known from the ephemeris and therefore may be supplied for
future forecast dates. Market-derived event outcomes may not.

## 10. Panchanga

Compute tithi, vara, Moon nakshatra, yoga, karana, waxing/waning, and lunation
proximity from the same convention manifest. Do not obtain them from a
third-party calendar using different ayanamsha/timestamp assumptions.

## 11. Calendar-Time Response Bank

For feature/event intensity `z(t)` and half-life `H` calendar days:

```text
a = 2^(-1 / H)
h(t) = a^delta_days * h(previous)
     + (1 - a^delta_days) * z(t)
```

Initial half-life grid:

```text
1, 3, 7, 14, 30, 90, 180, 365, 730,
1825, 3650, 7300, 10958 calendar days
```

Interpretation:

- orbital phase says where a body is in its cycle;
- event response says how long an alleged impact persists;
- response state is updated across weekends with real `delta_days`;
- pre-1995 ephemeris can initialize astronomy-only state.

The first implementation precomputes fixed states. A later continuous-time SSM
may learn small stable perturbations around the fixed decay grid.

## 12. Event-Relative Basis

For signed days `d` relative to exact event and width `s`:

```text
pre_event  = 1[d < 0] * exp(-0.5 * (d / s)^2)
post_event = 1[d >= 0] * exp(-0.5 * (d / s)^2)
```

Candidate widths:

```text
1, 3, 7, 14, 30, 90, 180, 365 days
```

Pre/post channels remain separate so anticipatory and aftermath hypotheses are
not forced to be symmetric.

## 13. Multi-Clock Inputs

### Market grid

```text
frequency: trading daily
initial length: 252
sensitivity: 64, 128, 504
```

### Fast astronomical grid

```text
frequency: calendar daily
history: approximately 96 days
known future: through longest active forecast/event window
primary bodies: Moon, Mercury; selected Mars/Sun events
```

### Medium grid

```text
frequency: weekly or event tokens
history: approximately 5 years / 260 weekly tokens
```

### Slow grid

```text
frequency: monthly
history: all available or approximately 40 years / 480 tokens
```

### Secular diagnostic grid

```text
frequency: annual
history: optional ephemeris-only 256-year context
role: representation diagnostic only
```

The secular grid cannot supply missing market outcomes and cannot establish a
Pluto-cycle effect from NIFTY history.

## 14. Feature Group Output

The feature builder returns a typed object:

```text
continuous_state
rashi_nakshatra_state
retrograde_station_state
conjunction_edges
drishti_edges
dignity_state
combustion_state
node_eclipse_state
panchanga_state
response_bank
event_tokens
feature_names
group_names
convention_hash
feature_spec_hash
```

No anonymous `f0`, `f1`, … output is accepted for confirmatory runs.

