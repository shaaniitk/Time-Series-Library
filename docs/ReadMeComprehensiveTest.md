# Comprehensive Component Test Runner

This repository includes a single entrypoint script to run all tests in the modular components testing framework.

## What it runs
The runner executes all pytest suites under `TestsModule/components`, covering:
- Attention: autocorrelation, enhanced autocorrelation, fourier, wavelet, bayesian, causal convolution, plus cross-attention checks where supported
- Embeddings: temporal, value, covariate, hybrid (utils global registry)
- Decomposition: series, stable, learnable, wavelet
- Encoder: standard, enhanced, stable, hierarchical
- Decoder: standard, enhanced, stable
- Processors (utils registry): decomposition wrappers, encoder/decoder wrappers, hierarchical fusion (weighted_concat, weighted_sum, attention_fusion), specialized utilities registration (frequency_domain, structural_patch, dtw_alignment, trend_analysis, quantile_analysis, integrated_signal)
- Output heads: regression/quantile/etc. shape checks
- Losses: standard/advanced/adaptive/bayesian/uncertainty (incl. quantile pinball 4D support)
- Sampling: representative variants

## How to run (Windows PowerShell)
- Preferred (uses repo venv and single runner):
  ```powershell
  .\tsl-env\Scripts\python.exe run_all_component_tests.py
  ```
- Direct pytest alternative:
  ```powershell
  .\tsl-env\Scripts\python.exe -X utf8 -m pytest -q TestsModule/components -q
  ```

Notes:
- Tests are lightweight and marked `extended`. Some attention variants may be skipped for cross-attention if unsupported.
- See `docs/TESTING_FRAMEWORK.md` for more details and troubleshooting.