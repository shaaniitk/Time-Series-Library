# Autoformer Framework Improvement Plan

This document analyzes shortcomings in the four Autoformer variants and provides step-by-step detailed changes for each `.py` file to build a unified framework. Changes focus on standardization, modularity, efficiency, and integration with `BaseModelConfig` from `configs/schemas.py`. These align with recommendations for shared components, uncertainty quantification, and diagnostics.

## 1. Autoformer.py (Base Autoformer)

### Shortcomings
1. **Configuration Handling:** Relies on raw `configs` without `BaseModelConfig` inheritance, leading to unvalidated parameters like `d_ff` and `n_heads`.
2. **Efficiency Issues:** Multiple decomposition calls and convolutions are computationally heavy for long sequences, lacking optimizations like caching.
3. **Modularity Gaps:** Core components (e.g., attention, decomposition) are not fully abstracted, causing duplication in variants.
4. **Uncertainty Integration:** Basic quantile support but no advanced methods like Bayesian sampling or conformal prediction.
5. **Backward Compatibility Overhead:** Unnecessary aliases and fallbacks clutter the code.

### Step-by-Step Changes
1. **Import BaseModelConfig:** At the top, add `from configs.schemas import BaseModelConfig`.
2. **Inherit from BaseModelConfig:** Modify `class EnhancedAutoformer(nn.Module)` to `class EnhancedAutoformer(nn.Module, BaseModelConfig)`.
3. **Add Validation Method:** Insert a `_validate_config` method after `__init__`: `def _validate_config(self, configs): try: BaseModelConfig(**configs.__dict__); except ValidationError as e: raise ValueError(f"Invalid config: {e}")` and call it in `__init__`.
4. **Abstract Decomposition:** Replace `self.decomp` with a call to a new shared `DecompFactory` (create in a new file `layers/modular/decomp_factory.py` if needed; for now, note to implement).
5. **Optimize Efficiency:** In `forward`, add caching for decompositions using `torch.lru_cache` on repeated calls.
6. **Enhance Uncertainty:** Add a `ConformalPredictor` wrapper in `forecast` method, integrating basic conformal prediction logic.
7. **Remove Clutter:** Delete unnecessary aliases at the end, ensuring compatibility via direct imports.
8. **Test Changes:** Add unit tests for validation and efficiency in a new `tests/modular/autoformer_test.py`.

## 2. BayesianEnhancedAutoformer.py

### Shortcomings
1. **Efficiency in Sampling:** Multiple loops for Bayesian/MC dropout are inefficient for large batches.
2. **Quantile Integration:** Limited handling of quantiles with Bayesian methods.
3. **KL Divergence Management:** Basic KL loss without adaptive weighting.
4. **Extensibility:** Hardcoded layer setups reduce reusability.
5. **Config Validation:** No inheritance from `BaseModelConfig`, risking invalid params.

### Step-by-Step Changes
1. **Import Dependencies:** Add `from configs.schemas import BaseModelConfig; from layers.BayesianLayers import BayesianLayerFactory`.
2. **Inherit from BaseModelConfig:** Change class to inherit from `nn.Module, BaseModelConfig`.
3. **Validate Config:** Add and call `_validate_config` similar to above.
4. **Use Shared Factory:** Replace dropout setup with `BayesianLayerFactory` calls for modular layers.
5. **Vectorize Sampling:** Rewrite sampling loops to use batched operations with `torch.vmap`.
6. **Enhance Quantile Support:** In forward passes, integrate quantile outputs with Bayesian variance.
7. **Adaptive KL:** Add a method to dynamically weight KL loss based on epoch.
8. **Add Diagnostics:** Integrate a diagnostics hook from a new `utils/diagnostics.py` for uncertainty metrics.

## 3. HierarchicalEnhancedAutoformer.py

### Shortcomings
1. **Complexity Overhead:** Multi-resolution processing increases computational load without dynamic scaling.
2. **Fusion Limitations:** Simple fusion lacks advanced cross-resolution attention.
3. **Quantile Handling:** Inconsistent with base Autoformer's quantile mode.
4. **Parameter Extraction:** `_get_params` is rigid and not validated.
5. **Modularity:** Duplicated wavelet code not shared with other variants.

### Step-by-Step Changes
1. **Import Modules:** Add `from configs.schemas import BaseModelConfig; from layers.modular import MultiScaleDecomposer`.
2. **Inherit and Validate:** Make class inherit `BaseModelConfig` and add validation.
3. **Modularize Decomposition:** Replace wavelet logic with `MultiScaleDecomposer` calls.
4. **Add Dynamic Scaling:** Implement level selection based on input length in `__init__`.
5. **Improve Fusion:** Add `CrossResolutionAttention` in encoder/decoder.
6. **Standardize Quantiles:** Align forecasting with Autoformer's quantile reshaping.
7. **Enhance Param Extraction:** Integrate with `BaseModelConfig` for validated params.
8. **Efficiency Profiling:** Add hooks for diagnostics toolkit to monitor hierarchy performance.

## 4. HybridAutoformer.py

### Shortcomings
1. **Integration Overload:** Too many mechanisms (Fourier, TCN) without modular switches.
2. **Efficiency Issues:** Multi-scale processing lacks optimizations for fusion.
3. **Fusion Simplicity:** Basic ensemble without adaptive weighting.
4. **Multi-Modality Limits:** Limited support beyond basic covariates.
5. **Config Gaps:** No unified validation, leading to param mismatches.

### Step-by-Step Changes
1. **Imports:** Add `from configs.schemas import BaseModelConfig; from layers.Attention import FourierAttention, TemporalConvNet`.
2. **Inheritance:** Extend with `BaseModelConfig` and add validation method.
3. **Modular Components:** Wrap attentions in a `FusionLayer` interface for easy switching.
4. **Optimize Fusion:** Add adaptive weighting via a gating mechanism in forward.
5. **Enhance Multi-Modality:** Extend embeddings for additional data types (e.g., images).
6. **Ablation Hooks:** Add config flags to disable components for testing.
7. **Uncertainty Addition:** Integrate `ConformalPredictor` for hybrid outputs.
8. **Diagnostics:** Add logging for component contributions using toolkit.

This plan ensures a unified, efficient Autoformer framework. Proceed by implementing changes file-by-file.

# Refactoring Plan Using ModularAutoformer

To enhance modularity across the four Autoformer models, we will leverage the existing `ModularAutoformer` class in `modular_autoformer.py`. This class provides a flexible, configuration-driven framework for assembling Autoformer variants from modular components in the `layers/modular` directory. The refactoring will involve subclassing `ModularAutoformer`, migrating model-specific logic, and removing duplicated code.

## Step-by-Step Refactoring Plan

1. **Preparation (Shared Across All Models)**:
   - Ensure all necessary modular components (e.g., encoders, decoders, attention mechanisms) are available in `layers/modular`.
   - Update `ModularAutoformerConfig` to include any model-specific parameters required for the four variants.
   - Create a new directory or module for refactored models if needed (e.g., `models/refactored`).

2. **Refactor Autoformer.py**:
   - Subclass `ModularAutoformer` in a new class, e.g., `RefactoredAutoformer`.
   - Migrate decomposition logic to use `LearnableDecomposition` from `layers/modular/decomposition`.
   - Replace custom encoder/decoder with `EnhancedEncoder` and `EnhancedDecoder` from `layers/modular/encoder` and `decoder`.
   - Configure attention using `FourierAttention` or similar from `layers/modular/attention`.
   - Remove duplicated embedding and projection layers, inheriting from the base class.
   - Update forward pass to use the modular backbone.

3. **Refactor BayesianEnhancedAutoformer.py**:
   - Subclass `ModularAutoformer` as `RefactoredBayesianAutoformer`.
   - Integrate Bayesian layers using `BayesianLayerFactory` (create if not existing in `layers/modular/layers`).
   - Configure uncertainty quantification (MC Dropout, KL divergence) via the config.
   - Reuse shared decomposition and attention components.
   - Adapt loss functions to use modular losses from `layers/modular/losses`.

4. **Refactor HierarchicalEnhancedAutoformer.py**:
   - Subclass as `RefactoredHierarchicalAutoformer`.
   - Use `HierarchicalEncoder` and `HierarchicalDecoder` from `layers/modular`.
   - Configure wavelet decomposition with `MultiResolutionDecomposer`.
   - Integrate Gated MoE FFN from `layers/modular/fusion`.
   - Handle multi-resolution processing through config-driven assembly.

5. **Refactor HybridAutoformer.py**:
   - Subclass as `RefactoredHybridAutoformer`.
   - Leverage existing imports from `layers` (e.g., `WaveletDecomposition`, `FourierAttention`, `TemporalConvNet`).
   - Configure hybrid attention and meta-learning via `ModularAutoformerConfig`.
   - Use `AdaptiveMixture` for experts and multi-scale processing.
   - Ensure task-specific heads are modularized in `layers/modular/output_heads`.

6. **Testing and Validation**:
   - For each refactored model, create unit tests comparing outputs with original implementations.
   - Validate performance on benchmark datasets.
   - Update documentation and any dependent scripts to use the refactored versions.

7. **Cleanup**:
   - Deprecate original files with warnings.
   - Monitor for 40-60% code reduction and improved maintenance.

This plan aligns with the existing schema and promotes a unified framework.