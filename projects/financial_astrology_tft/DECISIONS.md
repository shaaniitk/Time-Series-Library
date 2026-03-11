# Decision Log

> Append-only log of material project decisions. Corrections receive a new row;
> old rows are not silently rewritten.

| ID | Date | State | Decision | Reason |
|---|---|---|---|---|
| DEC-001 | 2026-07-29 | ACTIVE | Use the repository-native `TemporalFusionTransformer`; exclude `TFT_Nixtla`. | User scope |
| DEC-002 | 2026-07-29 | ACTIVE | Treat planetary trajectories as known-past/known-future covariates, not as an invented price-law physics loss. | Ephemerides are physically computed; no accepted planet-to-price governing equation exists |
| DEC-003 | 2026-07-29 | ACTIVE | Model Jyotisha principles faithfully enough to test them, while using skeptical out-of-sample evaluation and matched nulls. | Separates theory representation from evidential claims |
| DEC-004 | 2026-07-29 | ACTIVE | Do not use `seq_len=128` as the whole astrological memory. Start the local market branch at 252 sessions and represent long cycles through phase, event clocks, multiresolution summaries, and calendar-time response states. | Saturn and outer-planet hypotheses operate on longer clocks; a huge daily LSTM is also statistically inappropriate |
| DEC-005 | 2026-07-29 | ACTIVE | Keep classical Navagraha, literal classical price doctrine, anchored mundane astrology, and modern outer-planet astrology in separate experiment families. | Prevents historical/traditional claims from being blended post hoc |
| DEC-006 | 2026-07-29 | ACTIVE | Use transit-only classical features before any natal/event-anchor model. | NIFTY/NSE/India anchor choice and time are unresolved |
| DEC-007 | 2026-07-29 | ACTIVE | First targets are stationary returns and volatility/range variables; structured OHLC comes later. | Raw OHLC levels reward persistence and obscure incremental signal |
| DEC-008 | 2026-07-29 | ACTIVE | Pause financial-astrology implementation while the native TFT feature matrix runs and theory/data choices are discussed. | Explicit user direction |
| DEC-009 | 2026-07-29 | ACTIVE | Use *Brihat Samhita* Chapters 42 and 97 as a distinct literal classical hypothesis family. | The text explicitly discusses price fluctuations and differing fruition delays |
| DEC-010 | 2026-07-31 | ACTIVE; supersedes DEC-008 | Authorize the detailed plan and data intake, but prohibit financial-astrology neural training until native semantics, data, leakage, loader, and baseline gates close. | The user asked to start the main project while ensuring the next training run is not based on defective semantics |
| DEC-011 | 2026-07-31 | ACTIVE | Treat `TFT-SR00`–`TFT-SR09` as a mandatory semantic release, not as another feature benchmark. | Several advanced switches run but do not yet implement the scientific meaning implied by their names |
| DEC-012 | 2026-07-31 | ACTIVE | Keep target horizon, local `seq_len`, orbital phase, and hypothesized effect duration as four separate quantities. | A one-day prediction may depend on a long-lived state without requiring decades of daily recurrent context |
| DEC-013 | 2026-07-31 | ACTIVE | For the first planet test, train and freeze one market/calendar base checkpoint, then compare copied `disabled`, matched-null, and real-planet residual arms with common seeds and folds. | This isolates incremental planetary information from initialization and base-model variation |
| DEC-014 | 2026-07-31 | ACTIVE | Keep generic FFT, graph, lag-attention, compression, MoE, higher-order, and cross-attention switches off in the first NIFTY neural arm, even after repair. | Each must earn entry through a named hypothesis and matched ablation; repaired does not mean useful |
| DEC-015 | 2026-07-31 | ACTIVE | Build known-future astronomical features for the exact decision-time-to-target-time interval in addition to instantaneous state. | Intraday motion, station crossings, ingresses, and aspects can occur between the last observed close and the predicted session |
| DEC-016 | 2026-07-31 | ACTIVE | Close semantic repair with focused contracts and one deterministic micro-run; do not repeat the full 36-hour ETTh1 matrix. | The next substantive training budget belongs to the NIFTY hypothesis test |
| DEC-017 | 2026-07-31 | ACTIVE | Reject every supplied `*_sign_sin/cos` column and regenerate rashi only after the coordinate convention is frozen. | All 12 families and all 18,251 rows implement the exact same off-by-one/clipping transform, collapsing Aries/Taurus and removing sign 11 |
| DEC-018 | 2026-07-31 | ACTIVE | Rebuild the NIFTY table from authoritative raw OHLC/session keys; never fix the current dates with a weekend-only or global `+1 day` heuristic. | The file mixes correct and one-day-shifted regimes, including at least 412 exact four-return fingerprints |
| DEC-019 | 2026-07-31 | ACTIVE | Quarantine Shadbala from the primary experiment. | Formula, units, location, timestamp, and generator are absent, so future availability and row reproduction are unproven |
| DEC-020 | 2026-07-31 | ACTIVE | After timestamp, unit, frame, generator, and row-reproduction checks pass, start with `CLASSICAL_CONTINUOUS_V1`: 35 Sun-through-Saturn continuous fields plus one two-dimensional mean-node axis. | These fields pass internal geometry checks; this low-capacity slice avoids defective, redundant, and unauditable families but remains provisional until provenance closes |
| DEC-021 | 2026-07-31 | ACTIVE | Treat Ketu as an exact derived antipode of one Rahu/node axis, not as independent numeric evidence. | Supplied Rahu/Ketu coordinates are exactly antipodal and their speeds are duplicated |
| DEC-022 | 2026-07-31 | ACTIVE | Do not describe the supplied circular longitude encodings as Hilbert features. | No Hilbert fields or generator exist; unit-circle and velocity checks support direct trigonometric ephemeris encoding |

## Open Decisions

| ID | Required choice | Why it matters |
|---|---|---|
| OPEN-001 | Lahiri, Raman, or another ayanamsha; primary and sensitivity variants | Changes rashi/nakshatra boundaries |
| OPEN-002 | Market decision timestamp and location | Determines daily astronomical state and any lagna |
| OPEN-003 | Mean or true Rahu for the primary profile | Changes nodal timing; Ketu must remain opposite |
| OPEN-004 | 27 or 28 nakshatra convention | Changes categorical boundaries |
| OPEN-005 | Whole-sign, degree-orb, or dual representation of graha drishti | Changes aspect semantics |
| OPEN-006 | Whether Rahu/Ketu receive aspects and dignity in the selected tradition | Disputed across schools |
| OPEN-007 | Combustion thresholds and graha-yuddha definition | Tables and orbs vary |
| OPEN-008 | Whether retrograde is encoded as strength, affliction, or unsigned state only | Astrological interpretations differ |
| OPEN-009 | Whether an event/natal anchor is tested, and which chart(s) | Prevents outcome-selected anchor fishing |
| OPEN-010 | Confirmatory horizon/target and final lockbox boundary | Required before model comparison |
| OPEN-011 | Whether the primary target is close-to-close return, open-to-close return, next-open gap, or volatility/range | Fixes the decision timestamp, available information set, and interval ephemeris summary |
| OPEN-012 | Market data adjustment policy for splits/dividends and the exact NIFTY history source | Prevents target discontinuities and provenance ambiguity |
