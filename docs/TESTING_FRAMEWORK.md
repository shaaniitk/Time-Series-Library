# Modular Components Testing Framework

This repo includes a lightweight, component-centric pytest suite that validates the modular architecture across attention, embeddings, decomposition, encoder, decoder, output heads, losses, sampling, and processors.

## Quickstart

- Use the project venv:
  - Windows PowerShell
    - .\\tsl-env\\Scripts\\python.exe -m pytest -q TestsModule/components -q

- Run a focused slice:
  - .\\tsl-env\\Scripts\\python.exe -m pytest -q TestsModule/components/attention -q
  - .\\tsl-env\\Scripts\\python.exe -m pytest -q TestsModule/components/processors -q

Notes:
- Tests are marked with `@pytest.mark.extended`. Your runner may auto-assign markers.
- Tests are synthetic and tiny; they check registry presence, forward shape, and gradient flow.

## Coverage Highlights

- Attention: autocorrelation, enhanced autocorrelation, fourier, wavelet, bayesian, causal conv, plus cross-attention pattern where supported.
- Embeddings: temporal, value, covariate, hybrid via utils global registry.
- Decomposition: series, stable, learnable, wavelet.
- Encoder: standard, enhanced, stable, hierarchical.
- Decoder: standard, enhanced, stable.
- Processors (utils registry):
  - Decomposition wrappers: series, stable, learnable, wavelet.
  - Encoder/Decoder wrappers: standard, enhanced, stable, hierarchical.
  - Fusion: hierarchical with strategies weighted_concat, weighted_sum, attention_fusion.
  - Specialized utilities presence: frequency_domain, structural_patch, dtw_alignment, trend_analysis, quantile_analysis, integrated_signal.
- Output heads and Losses: standard/advanced/adaptive/bayesian/uncertainty, plus quantile pinball shapes.

## Design Principles

- Registry-driven: Tests ensure components are discoverable and instantiable via registries.
- Minimal tensors: Keep it fast and reliable; catch shape and basic behavior issues early.
- Graceful skips: Variants with different signatures are skipped if incompatible.

## Troubleshooting

- If imports fail, ensure the venv is active and dependencies installed.
- Some attention variants may not support cross-attention (Q != K/V); tests skip those gracefully.
- A FutureWarning from local_attention is expected and harmless.
