# Results Scorecard

> Financial-astrology experiment section is intentionally empty. No such model
> has been trained.

## 1. Native TFT Prerequisite Matrix — 14/14 Complete

Completed: 2026-07-31 23:22 Asia/Kolkata. These are single-seed ETTh1 results,
not automatic NIFTY architecture selections. The matrix is legacy-v1 screening
evidence because initialization, seed, neutrality, and several operator
semantics are not controlled well enough for paired causal attribution.

| Case | Status | MSE | MAE | RMSE | Notes |
|---|---|---:|---:|---:|---|
| baseline | complete | 0.03568881 | 0.14699268 | 0.18891482 | Reference |
| joint quantile | complete | 0.03587884 | 0.14622945 | 0.18941712 | Coverage 68.04%, width 0.36935 |
| quantile-only | complete | 0.03435775 | 0.14262331 | 0.18535845 | Current leader; coverage 71.91%, width 0.39879 |
| ALiBi | complete | 0.03467850 | 0.14418337 | 0.18622163 | Third by MSE |
| SDPA | complete | 0.03568881 | 0.14699268 | 0.18891482 | Predictions bit-for-bit equal to baseline |
| covariate reattention | complete | 0.03445451 | 0.14307196 | 0.18561927 | -3.46% MSE; promising, requires semantic/seed confirmation |
| interpretable cross-attention | complete | 0.03616716 | 0.14684366 | 0.19017665 | +1.34%; neutral/semantic repair required |
| lag attention | complete | 0.03693667 | 0.14863680 | 0.19218914 | +3.50%; implemented shifted prefix is not a causal response bank |
| sparse cross-mixing | complete | 0.03858219 | 0.15343992 | 0.19642351 | +8.11%; graph semantics/integration require repair |
| learned FFT | complete | 0.03607463 | 0.14592449 | 0.18993324 | +1.08%; selector does not yet match advertised modes |
| higher-order | complete | 0.03687086 | 0.14817350 | 0.19201785 | +3.31%; latent polynomial is not named covariate interaction |
| regime MoE | complete | 0.03547186 | 0.14576912 | 0.18833976 | -0.61%; near neutral, not a default |
| temporal compression | complete | 0.03730480 | 0.15012075 | 0.19314452 | +4.53%; short-sequence/decompression semantics require repair |
| experimental profile | complete | 0.04245717 | 0.16271803 | 0.20605137 | +18.96%; stacked non-neutral/defective extensions, not a clean component test |

The experimental profile's nominal 80% interval covered 67.67%, with width
0.41058 and zero crossing. Nominal intervals from all three quantile-producing
cases under-cover.

### Semantic interpretation rule

A negative delta cannot rescue a semantically invalid implementation, and a
positive delta cannot reject the intended operator when the implementation does
not embody it. `TFT-SR00`–`TFT-SR09` therefore close semantic correctness with
contract tests before NIFTY training. The matrix is retained as legacy-v1
evidence, not silently reinterpreted after repair.

## 2. Financial-Astrology Baselines

No results.

## 3. Classical Transit Families

No results.

## 4. Literal Classical Price/Fruition Family

No results.

## 5. Anchored Mundane Family

No results.

## 6. Modern Outer-Planet Family

No results.

## 7. Matched Nulls

No results.

## 8. Locked Holdout

```text
state: UNOPENED
```
