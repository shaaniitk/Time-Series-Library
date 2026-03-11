# Risks and Validity Controls

## 1. Fundamental Identifiability

NIFTY history from approximately 1995 contains roughly one Saturn orbit and only
fractions of the Uranus, Neptune, and Pluto orbits. The model can test local
states and relative phases, but it cannot validate full-cycle outer-planet laws.

Control:

- state claims as local-arc associations;
- use low-capacity fixed features;
- compare secular-time controls;
- seek older commodity/market targets and multi-market panels.

## 2. Deterministic-Time Confounding

Planet positions are deterministic smooth functions of time. They can proxy:

- secular market trend;
- regulation/technology eras;
- inflation regimes;
- calendar seasonality;
- crisis dates.

Control:

- ordinary calendar and Fourier/spline time baselines;
- matched smooth pseudo-planets;
- coherent date shifts and longitude rotations;
- multiple forward folds;
- regime leave-out analysis.

## 3. Multiple Testing

Many bodies, pairs, rashis, nakshatras, aspects, orbs, lags, targets, horizons,
anchors, and ayanamshas create enormous researcher degrees of freedom.

Control:

- frozen hypothesis registry;
- limited family-level ladder;
- multiplicity correction;
- locked final period;
- new version for every outcome-guided change.

## 4. Leakage

Risks:

- two-sided Hilbert transforms on market data;
- full-series scaling;
- rolling features crossing folds;
- wrong calendar/ephemeris timestamp join;
- future OHLC accidentally placed in decoder marks;
- selecting on epoch-level test metrics.

Control:

- prefix-invariance tests;
- train-only transforms;
- explicit known/observed schema;
- no epoch-level final-test evaluation;
- saved exact forecast dates.

## 5. Tradition Ambiguity

Ayanamsha, mean/true node, nakshatra count, node aspects, combustion, dignity,
graha-yuddha, and anchor charts differ by school.

Control:

- one primary convention manifest;
- separate sensitivity variants;
- no silent fallback;
- no best-performing tradition selected on lockbox results.

## 6. Classical/Modern Contamination

Uranus, Neptune, and Pluto are not part of the classical Navagraha list.

Control:

- separate `MODERN_OUTER` profile;
- separate scorecard rows and claims;
- no default merger into classical runs.

## 7. Nonstationary Target and Easy Persistence

Raw OHLC level prediction can look accurate by copying the last price.

Control:

- begin with stationary return/gap/body/range targets;
- compare zero/previous-value baselines;
- add structured OHLC only after incremental-value tests.

## 8. Deep-Model Overfit

The dataset is small relative to a high-width TFT and thousands of possible
interactions.

Control:

- linear screens before bespoke deep architecture;
- 50k–100k parameter budget;
- shared body encoder and low-rank graph;
- dropout, AdamW, clipping, early stopping;
- matched capacity;
- several seeds and folds.

## 9. Interpretability Overclaim

VSN or attention weights are not causal planetary effects.

Control:

- use them as diagnostics;
- require group knockouts, false ephemerides, and paired forward loss;
- report instability across seeds/folds.

## 10. Cross-Market Pseudoreplication

Adding many markets at the same dates increases rows but not independent
astronomical histories.

Control:

- cluster uncertainty by date;
- distinguish cross-sectional generalization from temporal replication;
- use older histories where possible.

## 11. Text-to-Market Adaptation

Classical sources often discuss commodities, kingdoms, scarcity, conflict, or
delayed phenomena, not modern NIFTY next-day returns.

Control:

- label NIFTY use as an adaptation;
- test volatility/range and longer horizons;
- use commodities as an external domain for literal price doctrine;
- do not rewrite a textual rule after seeing modern results.

## 12. Named-Feature Semantic Mismatch

An executable switch may not implement the scientific object implied by its
name. Examples already found include shifted-prefix “lag” attention, an all-bin
FFT filter described as mode selection, post-VSN latent products described as
original-covariate interactions, broadcast graph heads, and dead compression
decoder parameters.

Control:

- close `TFT-SR00`–`TFT-SR09` before neural NIFTY work;
- preserve legacy-v1 results rather than silently reinterpreting them;
- require exact no-op initialization and shared-state/batch hashes;
- use synthetic known-answer, gradient-liveness, mask, and coordinate tests;
- keep generic advanced switches off in the first planetary arm.

## 13. Horizon, Duration, and Availability Conflation

A long hypothesized Saturn effect can be confused with model `seq_len`, decoder
length, trading-row lag, or orbital period. Similarly, a deterministic event
known to occur tomorrow can be joined at the wrong market timestamp.

Control:

- store decision, target-open, target-close, and label-interval timestamps;
- represent elapsed time in calendar days;
- keep market context length, target horizon, orbital phase, and response
  half-life as separate configuration fields;
- compute interval astronomy from ephemerides only;
- prove future-OHLC perturbations cannot change known-future features.

## 14. Derived-Date Corruption

The supplied returns file mixes exchange-session dates with labels one calendar
day earlier in long blocks. A same-date merge can therefore attach the wrong
Moon position and wrong interval events while still passing ordinary shape and
missing-value checks.

Control:

- reconstruct returns from source OHLC keyed by authoritative sessions;
- validate holidays and special sessions against an exchange calendar;
- forbid weekend-only and global-offset repair rules;
- persist both displayed source date and canonical session key during audit;
- add known-answer joins across Friday/Monday, holidays, and special sessions.

## 15. Circular-Category Transformation Corruption

Every supplied rashi pair is shifted/clipped even though the underlying
longitude pair is coherent. A derived category can therefore be wrong while
its `sin/cos` norm remains exactly one.

Control:

- derive rashi from audited longitude rather than trusting stored category
  pairs;
- test both sides of all 12 boundaries and retrograde re-entry;
- require all categories to occur on an adequate date range;
- retain the rejected columns only as immutable audit evidence;
- hash the derivation code and astronomical convention manifest.

## 16. Generator and Timestamp Non-Reproducibility

Two local planetary artifacts with the same displayed date differ by exactly
8.25 hours of angular motion, while neither records the calculation instant.
The PySwissEph generator, flags, ayanamsha, ephemeris source, and Shadbala
formula are absent.

Control:

- block generator-dependent features rather than guess defaults;
- require exact UTC calculation instants and convention hashes;
- independently reproduce selected rows before loader acceptance;
- quarantine Shadbala until its full formula and availability are proven;
- generate a longer astronomy-only warm-up for slow response states.
