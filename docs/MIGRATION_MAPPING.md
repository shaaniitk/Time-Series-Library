# Modular Registry Migration Mapping

This document maps legacy per-family registry keys to the unified registry keys
now registered via `layers.modular.core.register_components`.

## Attention
Legacy Key | Unified Key | Notes
---------- | ----------- | -----
autocorrelation_layer | adaptive_autocorrelation_layer | Base layer variant now exposed as adaptive_autocorrelation_layer (EnhancedAutoCorrelation wraps logic)
adaptive_autocorrelation_layer | adaptive_autocorrelation_layer | Alias retained
enhanced_autocorrelation | autocorrelation | EnhancedAutoCorrelation registered under `autocorrelation` key with alias `enhanced_autocorrelation`
fourier_attention | fourier_attention | –
fourier_block | fourier_block | Added new registration (FourierBlock)
fourier_cross_attention | fourier_cross_attention | Added new registration (FourierCrossAttention)
wavelet_attention | wavelet_attention | –
wavelet_decomposition | wavelet_decomposition | WaveletDecomposition registered within ATTENTION family for compatibility
adaptive_wavelet_attention | adaptive_wavelet_attention | –
multi_scale_wavelet_attention | multi_scale_wavelet_attention | –
meta_learning_adapter | meta_learning_adapter | Alias: meta_adapter
adaptive_mixture | adaptive_mixture | Registered as adaptive_mixture (ComponentType.ADAPTIVE_MIXTURE_ATTN)
causal_convolution | causal_convolution | –
temporal_conv_net | temporal_conv_net | –
convolutional_attention | convolutional_attention | –
Bayesian / probabilistic keys | Same names | bayesian_attention, bayesian_multi_head_attention, variational_attention, bayesian_cross_attention
hierarchical_autocorrelation | hierarchical_autocorrelation | Added (hierarchical multi-resolution)
graph_attention_layer | graph_attention_layer | –
multi_graph_attention | multi_graph_attention | –

## Encoders
Legacy Key | Unified Key | Notes
standard | standard_encoder | ComponentType.STANDARD_ENCODER
enhanced | enhanced_encoder | Alias retained
stable | stable_encoder | Now explicitly registered
hierarchical | hierarchical_encoder | Alias: hier_encoder
graph | graph_encoder | –
hybrid_graph | hybrid_graph_encoder | Added
adaptive_graph | adaptive_graph_encoder | Added

## Decoders
Legacy Key | Unified Key | Notes
standard | standard_decoder | –
enhanced | enhanced_decoder | –
stable | stable_decoder | Added

## Decomposition
Legacy Key | Unified Key | Notes
moving_avg | moving_avg | SeriesDecomposition
learnable_decomposition / learnable_series_decomposition | learnable_decomp | LearnableSeriesDecomposition
stable_decomp | stable_decomp | StableSeriesDecomposition (odd kernel enforcement)
wavelet_decomp | wavelet_hierarchical_decomp | WaveletHierarchicalDecomposition (registered also under legacy key)
learnable_wavelet_decomp | advanced_wavelet_decomp | LearnableWaveletDecomposition (registered also under legacy key)

## Sampling
Legacy Key | Unified Key | Notes
deterministic | deterministic | –
bayesian | bayesian | –
monte_carlo_sampling | monte_carlo | MonteCarloSampling alias
mc_dropout / dropout | dropout | MC Dropout sampling

## Output Heads
Legacy Key | Unified Key | Notes
standard | standard | StandardOutputHead
quantile | quantile | QuantileOutputHead

## Losses
Legacy Key | Unified Key | Notes
mse | mse | Wrapper MSELoss
mae | mae | Wrapper MAELoss
quantile_loss | quantile_loss | QuantileLoss
bayesian_mse_loss | bayesian_mse_loss | BayesianMSELoss placeholder
bayesian_quantile_loss | bayesian_quantile_loss | BayesianQuantileLoss
adaptive_autoformer_loss | adaptive_autoformer_loss | –
frequency_aware_loss | frequency_aware_loss | –
uncertainty_calibration_loss | uncertainty_calibration_loss | –
mape_loss | mape_loss | MAPELoss
smape_loss | smape_loss | SMAPELoss
mase_loss | mase_loss | MASELoss
ps_loss | ps_loss | PSLoss requires pred_len
focal_loss | focal_loss | FocalLoss alias focal

## Fusion
Legacy Key | Unified Key | Notes
hierarchical_fusion | hierarchical_fusion | –

## Migration Guidance
quantile / pinball | quantile_loss | Legacy aliases captured
multi_quantile | quantile_loss | Alias to QuantileLoss multi-quantile mode
1. Replace imports of per-family registries with `from layers.modular.core import unified_registry, ComponentFamily`.
2. Create components via `unified_registry.create(ComponentFamily.ATTENTION, "fourier_attention", **kwargs)`.
3. Remove direct usage of legacy `AttentionRegistry.get/ create` after updating calls.
bayesian | bayesian_mse_loss | Generic bayesian alias mapped to BayesianMSELoss
4. For tests parametrized over legacy names, switch to unified names (see mappings above).
5. Default constructor args: consult `tools/unified_registry_selftest.py` for minimal instantiation patterns.

## Deprecation Timeline
- Phase 1 (current): Legacy registries emit DeprecationWarning and forward.
- Phase 2: Tests updated to unified API; legacy registries kept for external code grace period.
- Phase 3: Legacy registries removed; import stubs left raising clear exceptions.

## Open Items
- Confirm no external configs rely on removed aliases beyond those captured.
- Add any remaining decomposition or backbone components if later identified.
- Audit utils/implementations wrappers for redundancy and prune.
