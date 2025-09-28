# Generalized n-Wave / m-Target Forecasting Strategy

## 1. Executive Overview

This document distills a repeatable strategy for forecasting across **n heterogeneous wave inputs** and **m downstream targets** using the modular Autoformer stack. It aligns with the principles in `docs/MODULAR_AUTOFORMER_ARCHITECTURE.md`, `docs/HF_MODULAR_ARCHITECTURE_DOCUMENTATION.md`, `docs/COMPLETE_MODEL_TECHNICAL_DOCUMENTATION.md`, and the validation disciplines in `docs/TESTING_FRAMEWORK_DOCUMENTATION.md`. The goal is to provide:

- A reusable data contract for organizing multi-wave / multi-target datasets
- A component-driven modeling recipe that scales from custom modular builds to Hugging Face integrations
- Guidance on when to activate frequency-domain, graph-based, and attention-centric enhancements informed by recent literature
- Operational guardrails covering evaluation, deployment, and governance of the resulting pipelines

## 2. System Notation and Data Contract

### 2.1 Vocabulary

- **Wave input** \(W_i\): a temporally ordered sequence representing a specific wave-type signal (e.g., ocean buoy, power waveform, seismic band), indexed by \(i \in [1, n]\).
- **Target channel** \(Y_j\): a forecasted output (e.g., load, safety threshold, probabilistic bound), indexed by \(j \in [1, m]\).
- **Resolution tuple** \((\Delta_t, H)\): sampling interval and forecast horizon.
- **Context window** \(L\): encoder length feeding historical samples to the model.

### 2.2 Canonical Sample Schema

| Field | Shape | Description |
|-------|-------|-------------|
| `waves` | `[batch, L, n]` | Normalized historical wave magnitudes aligned to the global timeline |
| `wave_metadata` | `[batch, L, n, f_w]` | Feature embeddings (e.g., direction, depth, sensor lineage) |
| `covariates` | `[batch, L, c]` | Shared exogenous drivers (calendar, weather, control states) |
| `targets` | `[batch, H, m]` | Ground-truth forecast values used for supervision |
| `target_metadata` | `[batch, H, m, f_y]` | Optional future covariates per target |
| `time_features` | `[batch, L+H, f_t]` | Calendrical encodings consumed by embeddings |

### 2.3 Storage and Versioning

1. Serialize each dataset slice in columnar storage (Parquet or Feather) to preserve schema evolution.
2. Track provenance of each wave input via `wave_metadata` to enable slicing by instrument, location, and calibration.
3. Version-control schema contracts alongside configuration commits so that component registries stay synchronized with data expectations.

## 3. Data Pipeline Blueprint

### 3.1 Ingestion and Validation

1. **Schema enforcement**: use Pydantic validators mirroring the contract above. Violations should halt ingestion to prevent downstream silent errors.
2. **Temporal alignment**: resample all wave channels to the canonical \(\Delta_t\) using forward-fill within a tolerable gap threshold, then mark residual gaps for model-side masking.
3. **Amplitude normalization**: apply per-wave z-score normalization with rolling statistics where long-term drift is present.
4. **Target smoothing**: optionally denoise using robust filters (e.g., Hampel) before forecasting if the evaluation metric is sensitive to outliers.

### 3.2 Feature Engineering Tracks

- **Frequency transforms**: maintain both Fourier and wavelet features for each wave input. A dual representation supports hybrid encoder-decoder strategies such as AFE-TFNet[^1].
- **Graph embeddings**: construct adjacency matrices capturing spatial / physical couplings among wave sensors, as required by spatio-temporal graph networks (e.g., STGWN[^2]).
- **Lagged cross-target context**: build summary statistics (rolling correlations, co-integration indicators) across \(Y_j\) to inform shared decoders.

### 3.3 Petri Net Workflow Modeling

Represent the end-to-end ingestion → feature → training pipeline as a **Petri net** with places (`P`) denoting data states and transitions (`T`) representing transformations. This formalism highlights invariants such as “all wave channels validated” before the modeling transition fires, aiding auditability and concurrency reasoning[^3].

## 4. Modular Architecture Mapping

### 4.1 Component Selection Heuristics

| Forecast Need | Component Mapping | Registry Anchor |
|---------------|-------------------|-----------------|
| Baseline multi-wave forecasting | `AUTOCORRELATION` attention, `MOVING_AVG` decomposition, `STANDARD_ENCODER`/`DECODER`, `STANDARD_HEAD` | `configs/concrete_components.py` |
| Frequency-aware modeling (AFE-TFNet) | Pair `FOURIER_ATTENTION` with `WAVELET_DECOMP`; enable dual-path encoders to fuse time and frequency embeddings; adopt `ADAPTIVE_LOSS_WEIGHTING` | Attention + Decomposition registries |
| Spatio-temporal coupling (STGWN) | Integrate graph-aware embeddings via `TEMPORAL_CONV_ENCODER` and custom adjacency processors before encoder stack; optionally augment with `CROSS_RESOLUTION` attention | Encoder registry + processor extensions |
| Multi-horizon interpretability (TFT) | Leverage `MULTI_HEAD` or `ADAPTIVE_AUTOCORRELATION` attention, gating via `ENHANCED_ENCODER`/`DECODER`, and `QUANTILE` heads for probabilistic outputs | Attention + Output Head registries |
| Uncertainty quantification | Activate `BAYESIAN` sampling and `BAYESIAN_QUANTILE` loss; align head dimensions to \(m \times q\) quantile outputs | Sampling + Loss registries |

### 4.2 Unified Factory Guidance

- **Custom stack**: configure `ModularAutoformer` by composing components listed above. Validate dimension compatibility through `ComponentCompatibilityValidator`.
- **Hugging Face stack**: instantiate through `UnifiedAutoformerFactory` with `hf_*` variants (e.g., `hf_enhanced` for production inference, `hf_bayesian` when Monte Carlo dropout is required). The factory auto-completes HF configuration defaults as described in `docs/HF_MODULAR_ARCHITECTURE_DOCUMENTATION.md`.
- **Fallback choreography**: ensure each deployment configuration carries a “plan B” component profile (e.g., revert to custom `FOURIER_BLOCK` if HF backbone download fails). Document the mapping alongside environment manifests.

### 4.3 Configuration Templates

1. Define base YAML in `configs/modular_components.py` leveraging the schema above.
2. Layer scenario-specific overrides (frequency, graph, probabilistic) in separate YAMLs to promote reproducibility.
3. Store parameter sweeps (e.g., number of wave filters, graph hop counts) in `configs/schemas.py` for audit.

## 5. Modeling Enhancements Informed by Literature

### 5.1 Dual-Domain Fusion (AFE-TFNet)

AFE-TFNet couples wavelet and Fourier encoders with a **Domain-Harmonic subspace energy weighting** module that balances contributions from time and frequency streams[^1]. Within the modular stack:

1. Add a second encoder branch operating on frequency-domain tensors produced in §3.2.
2. Implement a gating module as a custom component (e.g., `ComponentType.ADAPTIVE_MIXTURE`) to learn cross-domain weights.
3. Route fused representations into the decoder, maintaining gradient flow to both branches.

### 5.2 Spatio-Temporal Graph Wave Networks

STGWN introduces position-aware graph convolutions and multi-scale fusion to capture spatial wave propagation and temporal lag interactions[^2]. Adaptation steps:

1. Generate adjacency matrices from `wave_metadata` (distance, propagation delays).
2. Extend encoders with graph convolutional layers ahead of temporal blocks.
3. Leverage multi-resolution fusion to reconcile high- and low-frequency responses across sensors.

### 5.3 Temporal Fusion Transformer Concepts

The Temporal Fusion Transformer provides interpretable multi-horizon attention through gating, variable selection, and static covariate encoders[^4]. Reuse these ideas by:

1. Applying variable selection networks per wave group before embeddings.
2. Incorporating gating layers (already supported by `ENHANCED_ENCODER`/`DECODER`) to suppress irrelevant signals.
3. Surfacing attention weights for decision support dashboards.

## 6. Training, Validation, and Testing Alignment

### 6.1 Experiment Orchestration

- Utilize the existing CLI and runner patterns described in `docs/TESTING_FRAMEWORK_DOCUMENTATION.md` for batch experimentation.
- Register scenario-specific tests under `tests/modular_framework/` or `TestsModule/...` ensuring PEP 257-compliant docstrings and Ruff formatting.
- For new components (e.g., dual-branch fusion modules), add targeted unit tests plus integration coverage (see §"Component Testing Requirements" in the modular architecture doc).

### 6.2 Progressive Validation Steps

1. **Component smoke tests** (`quick` category) to confirm registry wiring.
2. **Core algorithm tests** focusing on new decomposition / attention modules.
3. **Integration tests** verifying data contract adherence end-to-end.
4. **ChronosX / HF tests** when deploying HF-backed variants.

### 6.3 Metrics and Monitoring

- Track MAE / RMSE per target plus aggregated wave-group metrics.
- For probabilistic runs, monitor pinball loss and calibration error.
- Instrument training scripts with uncertainty diagnostics (variance across Monte Carlo samples).

## 7. Deployment and Operations

### 7.1 Packaging and Containerization

- Package inference artifacts via OCI-compliant containers; prefer `uv`-managed virtual environments to guarantee dependency parity.
- Keep model weights, configuration YAMLs, and component metadata under `/models/artifacts/<scenario>`.
- Ensure container start-up scripts validate availability of HF weights (download or fallback).

### 7.2 Runtime Architecture

1. Deploy a service layer that wraps the `UnifiedModelInterface` for both custom and HF models.
2. Expose health endpoints returning the active component stack and data contract hash.
3. Cache precomputed wavelet / Fourier features when latency-sensitive deployments lack GPU capacity.

### 7.3 Governance and Auditability

- Persist configuration hashes and Petri net state transitions for every production forecast batch.
- Record attention weights or graph activations for post-event forensics.
- Maintain rollback bundles containing prior component combinations.

## 8. Edge Cases, Risks, and Mitigations

| Risk | Mitigation |
|------|------------|
| Ambiguous alignment across wave cadences | Normalize to canonical \(\Delta_t\) and store masking metadata; decoder must respect mask during attention scoring |
| Sensor dropouts or corrupted waves | Implement sparse attention masks and data quality gates; fall back to learned imputation via decomposition components |
| Target explosion with large \(m\) | Use grouped decoding (shared trunk with per-target heads) and quantile head factorization |
| HF dependency outages | Ship custom component equivalents (Fourier blocks, graph convolutions) and toggle via configuration |
| Uncertainty drift | Schedule calibration tests under `bayesian` and `quantile` categories to detect distribution shifts |

## 9. Performance and Scalability Considerations

- **Memory footprint**: monitor when enabling multi-branch fusion or graph components; consider `hf_enhanced_advanced` for flash-attention efficiency when GPU constrained.
- **Training throughput**: precompute frequency features offline to reduce on-the-fly FFT overhead.
- **Inference latency**: adopt mixed-precision inference for HF stacks and cache decoder warm states for rolling forecasts.

## 10. Roadmap and Future Work

1. Automate component selection via search over registry metadata (e.g., choose among Fourier vs graph enhancements based on validation scores).
2. Extend Petri net instrumentation to include probabilistic guard conditions (tokens carrying uncertainty thresholds).
3. Investigate federated variants for geographically distributed wave fleets where data residency constraints apply.

---

[^1]: Y. Zhou et al., "AFE-TFNet: Adaptive Fusion Encoder Temporal Frequency Network for Multivariate Time Series Forecasting," arXiv:2505.06688, 2025. https://arxiv.org/abs/2505.06688
[^2]: X. Xu et al., "Spatiotemporal graph wave network for significant wave height forecasting," *Applied Ocean Research*, 2024. https://doi.org/10.1016/j.apor.2023.103730
[^3]: "Petri net," *Wikipedia*, revision accessed 2025-09-28. https://en.wikipedia.org/wiki/Petri_net
[^4]: B. Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," arXiv:1912.09363, 2020. https://arxiv.org/abs/1912.09363
