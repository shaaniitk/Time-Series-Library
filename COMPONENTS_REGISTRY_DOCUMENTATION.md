# Modular Components Registry - Comprehensive Catalog

This document lists all components currently accessible via the unified modular registry (`unified_component_registry.unified_registry`) including their type, name, key features, and source.

Generated on: 2025-08-08

## How to use
- List all: `unified_registry.list_all_components()`
- Fetch class: `unified_registry.get_component(<type>, <name>)`
- Create: `unified_registry.create_component(<type>, <name>, config)`
- Metadata: `utils.modular_components.registry.get_global_registry().get_metadata(<type>, <name>)`

---

## Catalog

Below are grouped by component type.

### attention
- restored_fourier_attention
  - Features: complex_filtering, adaptive_selection, learnable_filters
  - Sophistication: high
  - Source: layers.modular.attention.fourier_attention
- restored_autocorrelation_attention
  - Features: learned_k_predictor, multi_scale, neural_network
  - Sophistication: very_high
  - Source: layers.EfficientAutoCorrelation
- restored_meta_learning_attention
  - Features: maml, fast_weights, gradient_adaptation, meta_learning
  - Sophistication: very_high
  - Source: layers.modular.attention.adaptive_components
- plus legacy-wrapped attentions registered by `utils.implementations.attention.layers_wrapped_attentions.register_layers_attentions()` (wavelet/bayesian/fourier-cross/convolutional/cross-resolution families)

### processor (decomposition)
- series_decomposition_processor
  - Features: moving_average, trend, seasonal
  - Source: layers.modular.decomposition.series_decomposition
- stable_series_decomposition_processor (deprecated)
  - Note: Use series_decomposition_processor with ensure_odd_kernel=True
- learnable_series_decomposition_processor
  - Features: learnable_weights, adaptive_kernel, feature_specific
  - Source: layers.modular.decomposition.learnable_decomposition
- wavelet_hierarchical_decomposition_processor
  - Features: dwt, multi_resolution, wavelet
  - Source: layers.modular.decomposition.wavelet_decomposition

### processor (encoder)
- encoder_standard_processor
  - Features: transformer, autoformer_layers
  - Source: layers.modular.encoder.standard_encoder
- encoder_enhanced_processor
  - Features: transformer, gated_ffn, scaled_attn
  - Source: layers.modular.encoder.enhanced_encoder
- encoder_stable_processor (deprecated)
  - Note: Use encoder_enhanced_processor unless fixed_weight variant is required.
- encoder_hierarchical_processor
  - Features: hierarchical, multi_resolution, cross_scale
  - Source: layers.modular.encoder.hierarchical_encoder

### processor (decoder)
- decoder_standard_processor
  - Features: transformer, autoformer_layers
  - Source: layers.modular.decoder.standard_decoder
- decoder_enhanced_processor
  - Features: transformer, gated_ffn, scaled_attn
  - Source: layers.modular.decoder.enhanced_decoder
- decoder_stable_processor (deprecated)
  - Note: Use decoder_enhanced_processor unless fixed-weight variant is required.

### processor (fusion)
- fusion_hierarchical_processor
  - Features: multi_resolution, weighted_concat, weighted_sum, attention_fusion
  - Source: layers.modular.fusion.hierarchical_fusion

### loss
- mse_loss
  - Features: differentiable, stable
  - Task Types: regression, forecasting
  - Source: utils.modular_components.implementations.losses
- mae_loss
  - Features: robust_to_outliers, differentiable
  - Task Types: regression, forecasting
  - Source: utils.modular_components.implementations.losses
- huber_loss
  - Features: robust_to_outliers, delta_param
  - Task Types: regression, forecasting
  - Source: utils.modular_components.implementations.losses
- cross_entropy_loss
  - Features: class_weights, ignore_index
  - Task Types: classification
  - Source: utils.modular_components.implementations.losses
- negative_log_likelihood_loss
  - Features: requires_uncertainty, gaussian
  - Task Types: probabilistic_forecasting, probabilistic_regression
  - Source: utils.modular_components.implementations.losses
- quantile_loss
  - Features: provides_uncertainty, multi_quantile
  - Task Types: quantile_regression, probabilistic_forecasting
  - Source: utils.modular_components.implementations.losses
- multitask_loss
  - Features: weighted_sum, composite
  - Task Types: multi_task
  - Source: utils.modular_components.implementations.losses

### output
- forecasting_head
  - Features: multistep, activation_optional
  - Task Types: forecasting
  - Source: utils.modular_components.implementations.outputs
- regression_head
  - Features: sequence_or_single, activation_optional
  - Task Types: regression
  - Source: utils.modular_components.implementations.outputs
- classification_head
  - Features: sequence_or_pooled, num_classes_param
  - Task Types: classification
  - Source: utils.modular_components.implementations.outputs
- probabilistic_forecasting_head
  - Features: mean_logvar, uncertainty
  - Task Types: probabilistic_forecasting
  - Source: utils.modular_components.implementations.outputs
- layers_standard_output_head
  - Features: linear_projection
  - Task Types: output
  - Source: layers.modular.output_heads.standard_output_head
- layers_quantile_output_head
  - Features: quantile_projection
  - Task Types: output
  - Source: layers.modular.output_heads.quantile_output_head

### embedding
- temporal_embedding
  - Features: positional, temporal_features
  - Source: utils.modular_components.implementations.embeddings
- value_embedding
  - Features: continuous_projection, optional_binning
  - Source: utils.modular_components.implementations.embeddings
- covariate_embedding
  - Features: categorical, numerical
  - Source: utils.modular_components.implementations.embeddings
- hybrid_embedding
  - Features: combine_temporal_value_covariate
  - Source: utils.modular_components.implementations.embeddings

### feedforward
- standard_ffn
  - Features: linear, dropout, activation
  - Source: utils.modular_components.implementations.feedforward
- gated_ffn
  - Features: gated, glu, dropout
  - Source: utils.modular_components.implementations.feedforward
- moe_ffn
  - Features: mixture_of_experts, experts, topk_gating
  - Source: utils.modular_components.implementations.feedforward
- conv_ffn
  - Features: conv1d, same_padding, dropout
  - Source: utils.modular_components.implementations.feedforward

### adapter
- covariate_adapter
  - Features: covariates, fusion: project|concat|add, tokenizer_aware
  - Source: utils.modular_components.implementations.adapters
- multiscale_adapter
  - Features: multi_scale, concat|add_aggregation
  - Source: utils.modular_components.implementations.adapters

---

## Notes
- All processors follow a consistent `process_sequence(x, context, target_length, **kwargs)` API.
- Encoders/decoders accept attention and decomposition components via `config.custom_params`.
- Decomposition processors can be injected into encoders/decoders; adapters bridge utils processors to legacy layer expectations.
- The unified facade auto-registers attentions, decompositions, encoders, decoders, and fusions.

## Quick checks
- Encoder smoke: `python quick_smoke_encoder.py`
- Decoder smoke: `python quick_smoke_decoder.py`
- Decomposition smoke: `python quick_smoke_decomposition.py`
- Fusion smoke: build a list of multi-res tensors and call the processor.

If a component is missing, ensure its registration function was called during `unified_registry` initialization.
