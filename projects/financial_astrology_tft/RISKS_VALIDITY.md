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

