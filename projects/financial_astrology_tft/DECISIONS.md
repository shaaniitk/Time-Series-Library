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

