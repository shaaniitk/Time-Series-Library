# Enhanced SOTA PGAT Algorithmic Upgrade Plan

## Guiding Priorities
- Maximize predictive accuracy and theoretical expressiveness, accepting higher compute or memory cost when necessary.
- Preserve modular replaceability of components so research variants can be A/B tested rapidly.
- Maintain reproducibility via deterministic seeds, configuration snapshots, and explicit dataset lineage tracking.

## Algorithmic Improvement Themes
1. **Adaptive multi-scale patching**: Expand beyond a single patch length by composing multiple `PatchingLayer` variants, add attention-gated fusion for overlapping patches, and learn dynamic stride/length schedules. Inspired by the adaptive pathway strategy in [Pathformer (Chen et al., 2024)](https://arxiv.org/abs/2402.05956) and related multi-granularity designs.
2. **Hierarchical temporal→spatial attention**: Stack temporal convolutions with multi-stage attention blocks to map temporal patches into node embeddings, enforce sparsity/entropy regularization, and expose interpretability hooks.
3. **Uncertainty-aware graph fusion**: Augment the dynamic/adaptive graph duo with stochastic structure learners that sample multiple adjacency hypotheses, estimate reliability, and support hypergraph priors, following ideas from [TAEGCN (Zhao et al., 2025)](https://arxiv.org/abs/2505.00302) and [STOIC (Kamarthi et al., 2024)](https://arxiv.org/abs/2407.02641).
4. **Probabilistic, multi-task decoding**: Retrofit the decoder with mixture-density or flow heads, add quantile/auxiliary tasks, and enable long-horizon patch outputs similar to [TimesFM (Sen & Zhou, 2024)](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).
5. **Rigorous validation & monitoring**: Build automated ablation suites, calibration diagnostics, and visualization tooling for scale usage, attention entropy, and graph drift.

## Step-by-Step Implementation Plan
1. **Design multi-scale patch composer**
   - Implement a `MultiScalePatchingComposer` that instantiates multiple patch sizes (short/medium/long horizons) and optional dilated temporal convolutions.
   - Add an attention-gated fusion layer to merge overlapping patches; regularize redundancy via KL/entropy penalties.
   - Integrate learnable rotary (time) and Fourier (frequency) positional encodings at the patch token level; expose config toggles.
   - *Validation*: run scale-ablation benchmarks, visualize receptive-field coverage, and audit spectrum density.
2. **Build hierarchical temporal→spatial mapper**
   - Prepend causal temporal convolutions before attention to capture local phase shifts.
   - Stack multi-head attention blocks that alternate between patch-level and node-level resolutions, concluding with cross-scale fusion.
   - Apply sparsity/diversity regularizers to attention weights and log alignment scores for explainability.
   - *Validation*: inspect attention entropy, perform perturbation tests on patch strides, and compare attribution heatmaps.
3. **Extend graph fusion framework**
   - Generalize `GatedGraphCombiner` into a meta-controller that scores base/adaptive/hypergraph proposals using contextual features.
   - Introduce a stochastic structure learner (STOIC-style) that samples adjacency hypotheses, tracks uncertainty, and feeds expectation/variance into the combiner.
   - Support hypergraph or community-aware priors plus curriculum toggles that warm-start before activating stochastic exploration.
   - *Validation*: calibrate edge probabilities, stress-test under structure shifts, and run causal masking interventions.
4. **Upgrade decoder for probabilistic multi-task outputs**
   - Add mixture-density or normalizing-flow heads alongside quantile and classification auxiliaries (e.g., anomaly flags).
   - Enable long-output patch generation (TimesFM-like) so one forward pass covers multiple horizons; optionally wrap in deep ensembles or MC dropout.
   - Enforce distributional calibration with CRPS/PIT losses and variance penalties.
   - *Validation*: evaluate CRPS, coverage, tail-risk backtests, and multi-horizon consistency.
5. **Integrate training curriculum & tooling**
   - Phase training: deterministic warm-up → activate stochastic graph + probabilistic decoder → fine-tune multi-scale regularizers.
   - Implement automated ablation toggles, reproducible config snapshots, and visualization dashboards for scale usage, attention drift, and graph evolution.
   - Establish regression suites spanning domain shifts, missing data, and noise injections; monitor via experiment tracker.
   - *Validation*: ensure statistical significance vs. baseline, document reproducibility artifacts, and archive diagnostic plots.

## Risk & Mitigation Checklist
- **Optimization complexity**: introduce switchable "baseline" mode, gradient clipping, and gradient checkpointing when memory spikes.
- **Graph instability**: throttle stochastic structure updates, apply spectral norm constraints, and monitor calibration metrics.
- **Overfitting to synthetic scales**: employ cross-domain validation, data augmentations, and sparsity penalties on attention/graph weights.
- **Interpretability regression**: require attention/structure logging and integrate explainability reviews into CI.

## Reference Shortlist
- Pathformer: Adaptive multi-scale transformer pathways for time-series (arXiv:2402.05956).
- Temporal Attention Evolutional GCN (TAEGCN): dynamic graph evolution with causal temporal modules (arXiv:2505.00302).
- STOIC: stochastic structure learning with calibrated forecasts (arXiv:2407.02641).
- TimesFM: decoder-only foundation forecasting with long patch outputs (Google Research Blog, Feb 2024; arXiv:2310.10688).
