# Results Scorecard

> Financial-astrology experiment section is intentionally empty. No such model
> has been trained.

## 1. Native TFT Prerequisite Matrix — Partial

Last inspected: 2026-07-29 21:07 Asia/Kolkata. The same cross-attention case was
still active.

| Case | Status | MSE | MAE | RMSE | Notes |
|---|---|---:|---:|---:|---|
| baseline | complete | 0.03568881 | 0.14699268 | 0.18891482 | Reference |
| joint quantile | complete | 0.03587884 | 0.14622945 | 0.18941712 | Coverage 68.04%, width 0.36935 |
| quantile-only | complete | 0.03435775 | 0.14262331 | 0.18535845 | Current leader; coverage 71.91%, width 0.39879 |
| ALiBi | complete | 0.03467850 | 0.14418337 | 0.18622163 | Current second |
| SDPA | complete | 0.03568881 | 0.14699268 | 0.18891482 | Predictions bit-for-bit equal to baseline |
| interpretable cross-attention | running at inspection | — | — | — | Checkpoint still improving |
| lag attention | queued | — | — | — | — |
| sparse cross-mixing | queued | — | — | — | — |
| learned FFT | queued | — | — | — | — |
| higher-order | queued | — | — | — | — |
| regime MoE | queued | — | — | — | — |
| temporal compression | queued | — | — | — | — |
| covariate reattention | queued | — | — | — | — |
| experimental profile | queued | — | — | — | — |

Nominal quantile interval is 80%; both completed quantile models under-cover.

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
