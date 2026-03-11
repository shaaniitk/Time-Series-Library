# Hypothesis and Theory Registry

> Status: `DISCUSSION DRAFT`; no family is confirmatory until its row says
> `FROZEN`.
>
> This file defines what the project means by “thinking like a financial
> astrologer.” It is a hypothesis library, not a statement that the rules are
> scientifically true.

## 1. Mandatory Registry Fields

Every enabled rule must eventually have:

```text
id
version
status: DRAFT | FROZEN-DEVELOPMENT | FROZEN-CONFIRMATORY | RETIRED
tradition: CLASSICAL | CLASSICAL-ADAPTED | MODERN
source
bodies
astronomical convention
exact formula
event condition
applying/separating rule
orb or bandwidth
decision timestamp
forecast target and label interval
latency
persistence
phase representation
decision-to-target interval summary fields
target channel
expected direction, or explicitly NONDIRECTIONAL
interaction partners
experiment arm
matched null
known disputes
change reason
```

Any outcome-guided change to a frozen rule creates a new ID/version. It does not
overwrite the old rule.

## 2. Theory Profiles

| Profile | Contents | Initial role |
|---|---|---|
| `CLASSICAL_TRANSIT_V1` | Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu; transit-only | First structured theory profile |
| `CLASSICAL_PRICE_TEXT_V1` | Literal computable features from *Brihat Samhita* Chapters 42 and 97 | Separate classical textual test |
| `ANCHORED_MUNDANE_V1` | Transits to frozen NIFTY/NSE/India event chart | Disabled pending anchor decision |
| `MODERN_OUTER_V1` | Uranus, Neptune, Pluto and modern financial-cycle relations | Exploratory, separate from classical claims |

Classical and modern features may be compared or combined only in explicitly
named arms. A run containing Uranus, Neptune, or Pluto cannot be reported simply
as a classical Vedic model.

## 3. Candidate Feature Families

### `AST-C01` — Continuous astronomical state

| Field | Draft definition |
|---|---|
| Tradition | Classical-compatible physical state |
| Bodies | Navagraha; outer planets only in `MODERN_OUTER_V1` |
| Formula | `sin(lon)`, `cos(lon)`, latitude, distance, signed angular velocity, acceleration, radial velocity where available |
| Purpose | Preserve continuous geometry before imposing categories |
| Direction | `NONDIRECTIONAL` |
| Latency | Current target-date state plus response-bank ablations |
| Null | Coherent date shift; circular phase rotation |
| State | `DRAFT` |

Rules:

- longitude is circular and never enters as an unwrapped scalar alone;
- units and reference frame are persisted;
- signed velocity is retained so retrograde is not collapsed into a bit;
- Ketu is derived as the opposite node rather than learned as an independent
  unconstrained orbit.

### `AST-C02` — Rashi and ingress state

| Field | Draft definition |
|---|---|
| Tradition | Classical |
| Features | 12-sign ID, degree within sign, sign lord, movable/fixed/dual, element, distance/time to prior and next boundary |
| Events | Ingress impulse; days since/until ingress; repeated crossing during retrograde loop |
| Direction | Initially `NONDIRECTIONAL`; sign lookup tables are frozen before results |
| State | `DRAFT` |

Use smooth boundary-distance features in addition to categorical IDs. This lets
the model distinguish the approach, exact crossing, and aftermath of an ingress.

### `AST-C03` — Nakshatra and pada

| Field | Draft definition |
|---|---|
| Tradition | Classical |
| Features | Nakshatra ID, pada, nakshatra lord, boundary distance, ingress/re-entry event |
| Primary convention | Unresolved: 27 versus 28 |
| Direction | `NONDIRECTIONAL` until a sourced rule is frozen |
| State | `DRAFT` |

The primary 27/28-mansion convention must be frozen; the other may be a
sensitivity test. It must not be selected after seeing performance.

### `AST-C04` — Retrograde, station, and visibility loop

| Field | Draft definition |
|---|---|
| Tradition | Classical |
| Continuous features | Signed speed, acceleration, `exp(-(abs(speed)/scale)^2)` station proximity |
| State features | Direct/retrograde, applying to station, separating from station, days since/until station |
| Loop features | Pre-shadow, first crossing, retrograde crossing, direct crossing, post-shadow |
| Direction | Competing interpretations; primary representation is unsigned/nondirectional |
| State | `DRAFT` |

A single retrograde flag is inadequate. A planet slowing toward station and one
moving rapidly in the middle of a retrograde loop are distinct states.

### `AST-C05` — Conjunction, graha-yuddha, and multi-planet meetings

| Field | Draft definition |
|---|---|
| Tradition | Classical |
| Pair geometry | Wrapped longitude gap, relative speed, same-rashi flag, smooth conjunction kernel |
| Timing | Applying/separating, signed days to exactness, closest-approach distance |
| Rule families | Ordinary conjunction and strict graha-yuddha remain separate |
| State | `DRAFT` |

Do not use one post-VSN latent “higher-order interaction” as evidence for named
planetary meetings. These directed pair features must be calculated before
variable selection.

### `AST-C06` — Parashari graha drishti

Confirmatory-core candidate:

| Graha | Whole-sign directed aspects from itself |
|---|---|
| All classical grahas | 7th |
| Mars | 4th, 7th, 8th |
| Jupiter | 5th, 7th, 9th |
| Saturn | 3rd, 7th, 10th |

Representations:

1. a categorical whole-sign directed edge;
2. an optional degree/orb activation as a separate ablation;
3. applying/separating and time-to-exactness where degree geometry is enabled.

Rahu/Ketu aspects are disputed and excluded from the core until a tradition is
selected. Their 5/7/9-style variants must have a separate ID.

State: `DRAFT`.

### `AST-C07` — Dignity, relationships, and dispositor state

Candidate fields:

- exaltation and exact exaltation distance;
- debilitation and exact debilitation distance;
- moolatrikona;
- own sign;
- natural and temporary friend/neutral/enemy relationship;
- sign lord and a bounded dispositor chain;
- optional natural benefic/malefic classification.

Node dignity is disputed and excluded from the core. Signed market-return
direction must not be inferred merely from “benefic” or “malefic”; the first
target channel for these states is regime/volatility modulation.

State: `DRAFT`.

### `AST-C08` — Combustion and heliacal state

Primary representation:

```text
shortest Sun-planet angular separation
applying/separating
relative speed
smooth proximity basis
```

Binary combustion thresholds vary by source and retrograde state. They are
secondary ablations with their exact table stored in the run manifest.

State: `DRAFT`.

### `AST-C09` — Nodes, eclipses, and eclipse windows

Candidate fields:

- mean or true Rahu longitude, explicitly selected;
- Ketu as the exact opposite axis;
- Sun and Moon angular distance to the nodal axis;
- eclipse-season proximity;
- time since/until solar or lunar eclipse;
- event type, magnitude, local visibility, rashi, and nakshatra when available;
- pre/post-event response kernels.

Mean and true nodes must never be mixed silently. Local visibility requires a
declared location.

State: `DRAFT`.

### `AST-C10` — Panchanga and lunar timing

Candidate fields:

- tithi;
- vara;
- Moon nakshatra;
- yoga;
- karana;
- waxing/waning;
- new/full Moon proximity.

These form the astrological calendar baseline. The ordinary Gregorian/calendar
and Fourier-time controls remain a separate non-astrological arm.

State: `DRAFT`.

### `AST-C11` — Planetary significator groups

Theory-derived group masks may encode themes such as:

| Group | Candidate grahas | Candidate market channel |
|---|---|---|
| Commerce/communication | Mercury | gaps, short-term direction, trading activity |
| Wealth/liquidity/expansion | Jupiter, Venus | medium return/volatility regime |
| Restriction/scarcity | Saturn | drawdown, persistence, range regime |
| Conflict/shock | Mars, nodes | range, absolute return, crash hazard |
| Public/mood/food-water | Moon | daily trigger, gap, absolute return |
| Authority/government | Sun | event/regime interaction |

These mappings are adaptations. They do not establish a classical signed NIFTY
rule. Test group masks against an unstructured, capacity-matched feature model.

State: `DRAFT`.

### `AST-C12` — Delayed fruition

The classical-text candidate uses body-specific timing rather than forcing all
effects into next-day return.

Provisional source-derived centers:

| Body/phenomenon | Textual center | ML representation |
|---|---:|---|
| Sun | about a fortnight | smooth kernel around 14 calendar days |
| Moon | about one month | smooth kernel around 30 calendar days |
| Mercury | tied to disappearance | event-relative clock |
| Jupiter | about one year | smooth kernel around 365 days |
| Venus | about six months | smooth kernel around 182 days |
| Saturn | about one year | smooth kernel around 365 days |
| Rahu | about six months | smooth kernel around 182 days |
| Solar eclipse | about one year | event-response kernel |

These are tested as source-faithful hypotheses, not asserted facts. Exact point
lags are too brittle, so the center and a preregistered bandwidth define a
distributed-lag kernel.

State: `DRAFT`.

### `AST-C13` — Literal classical price doctrine

This family encodes only the computable parts of *Brihat Samhita* Chapter 42:

- new/full Moon;
- eclipses and halos where reliable data exists;
- solar rashi;
- benefic/malefic accompaniment or aspect;
- sign strength;
- source-specified holding-delay category;
- commodity category where a mapping can be declared without hindsight.

Because the text addresses ancient commodities rather than a modern equity
index, the strongest external check is a commodity-market panel. NIFTY results
are labeled an adaptation.

State: `DRAFT`.

### `AST-A01` — Transit-only mundane state

This is the clean initial theory:

```text
absolute sidereal state
+ rashi/nakshatra/retrograde/station
+ planetary pair relations
+ eclipse/panchanga state
```

It uses no natal houses or lagna. It is the first structured classical profile.

State: `DRAFT`, recommended first.

### `AST-A02` — Event/natal-anchor transits

Candidate anchors:

- NSE first trading event;
- NIFTY base/launch event;
- India independence/event chart.

Rules:

- exact date, time, timezone, location, and rationale are preregistered;
- if time is uncertain, omit lagna/houses or use a frozen time-uncertainty
  ensemble;
- an anchor is never selected on the final test;
- transit-only results remain separately reported.

State: `DISABLED-PENDING-DECISION`.

### `AST-M01` — Modern outer-planet state

Bodies:

```text
Uranus, Neptune, Pluto
```

Allowed first features:

- circular longitude and current rashi/nakshatra state;
- signed speed, retrograde, station and ingress clocks;
- aspects/relative phase to classical grahas;
- aspects to a frozen anchor, only if `AST-A02` is enabled;
- low-capacity secular regime modulation.

No classical sign rulership, dignity, or Navagraha label is assigned to these
bodies. Results are local-arc associations because NIFTY history contains only a
fraction of each orbit.

State: `DRAFT-EXPLORATORY`.

### `AST-M02` — Modern financial cycles

Candidate relations:

- Jupiter-Saturn phase and conjunction/opposition cycle;
- Saturn-Uranus and other outer/classical relations;
- outer-planet conjunction/opposition;
- modern IPO/first-trade chart theories;
- slow-background × fast-trigger hypotheses.

This is a practitioner/modern family, not a direct classical claim.

State: `DRAFT-EXPLORATORY`.

## 4. Slow-Background × Fast-Trigger Hypothesis

The primary interaction architecture will be able to test:

```text
slow state sets a regime
fast event triggers a local response
```

Candidate slow state:

- Jupiter, Saturn, Rahu/Ketu rashi/nakshatra/dignity/aspect state;
- Jupiter-Saturn relative phase;
- eclipse-season state;
- modern outer-planet state only in the modern arm.

Candidate fast trigger:

- Moon, Mercury, or Mars ingress;
- station or retrograde-loop transition;
- conjunction/aspect becoming exact;
- fast contact with a slow graha or node.

Use a low-rank gated bilinear interaction rather than enumerating an unrestricted
cross-product. Every enabled slow-fast pair group is declared before the run.

## 5. Experiment Ladder

| ID | Added family | Claim class |
|---|---|---|
| H00 | Market + Gregorian calendar/Fourier controls | Baseline |
| H01 | Raw classical continuous ephemeris | Astronomy state |
| H02 | Rashi, nakshatra, ingress | Classical structure |
| H03 | Retrograde, station, combustion | Classical state/event |
| H04 | Parashari aspects + conjunction graph | Classical interactions |
| H05 | Eclipses, nodes, panchanga | Classical timing |
| H06 | Dignity/dispositor masks | Classical structure |
| H07 | Slow-background × fast-trigger | Classical adapted architecture |
| H08 | Literal Chapters 42/97 rules and lags | Classical textual |
| H09 | Frozen event-anchor transits | Anchored mundane |
| H10 | Outer planets | Modern exploratory |
| H11 | All preregistered classical families | Classical combined |
| H12 | Classical + modern | Combined exploratory |

Each rung must use the same forecast dates, split policy, seeds, training budget,
and capacity accounting. Every planet-enabled rung receives matched null and
feature-knockout counterparts.

## 6. Target Channels

Astrological rules need not manifest as precise next-day signed return. Candidate
targets are:

```text
signed close return
absolute return / realized volatility
overnight gap
intraday body
high-low range
drawdown or crash hazard
cumulative return over 1, 5, 20, 60, 126, and 252 sessions
```

The confirmatory target and horizon must be frozen before the final holdout. A
family that predicts volatility but not direction must be reported that way.

## 7. Source Register

These sources define hypotheses, not scientific validation:

1. [Brihat Samhita, Chapter 42 — Fluctuation of prices](https://www.wisdomlib.org/hinduism/book/brihat-samhita/d/doc229187.html)
2. [Brihat Samhita, Chapter 97 — Time of fruition](https://www.wisdomlib.org/hinduism/book/brihat-samhita/d/doc229360.html)
3. [Brihat Samhita, Chapter 17 — Planetary conjunctions](https://www.wisdomlib.org/hinduism/book/brihat-samhita/d/doc228918.html)
4. [Brihat Samhita, Chapter 20 — Multi-planet meetings](https://www.wisdomlib.org/hinduism/book/brihat-samhita/d/doc228921.html)
5. [Brihat Samhita, Chapter 15 — Nakshatras](https://www.wisdomlib.org/hinduism/book/brihat-samhita/d/doc228916.html)
6. [Brihat Samhita, Chapter 5 — Rahu and eclipses](https://www.wisdomlib.org/hinduism/book/brihat-samhita/d/doc226721.html)
7. [Brihat Samhita, Chapter 104 — Planetary transits](https://www.wisdomlib.org/hinduism/book/brihat-samhita/d/doc229368.html)
8. [Brihat Parashara Hora Shastra, Chapter 3](https://parashara.net/index.php?page=chapter3)
9. [Brihat Parashara Hora Shastra aspect chapter summary](https://www.wisdomlib.org/shop/books/jyotisha/brihat-parashara-hora-shastra/doc234202.html)
10. [Swiss Ephemeris technical documentation](https://www.astro.com/swisseph-download/doc/swisseph.pdf)
11. [NASA/JPL planetary physical parameters](https://ssd.jpl.nasa.gov/planets/phys_par.html)
12. [Wiley, A Trader's Guide to Financial Astrology](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118646953)
