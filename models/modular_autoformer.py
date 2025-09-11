import torch
import torch.nn as nn
from argparse import Namespace
from typing import Union, Dict, Any

# GCLI Structured Configuration System
from configs.schemas import ModularAutoformerConfig, ComponentType
from configs.modular_components import (
    ModularAssembler, component_registry, AssembledAutoformer,
    register_all_components
)

# Legacy imports for backward compatibility
from layers.modular.decomposition import get_decomposition_component
from layers.modular.encoder import get_encoder_component
from layers.modular.decoder import get_decoder_component
from layers.modular.attention.registry import get_attention_component
import layers.modular.core.register_components  # noqa: F401  # ensure attention registry populated
from layers.modular.sampling import get_sampling_component
from layers.modular.output_heads import get_output_head_component
from layers.modular.loss import get_loss_component
from layers.Embed import DataEmbedding_wo_pos
from layers.modular.core.logger import logger

# Import unified base framework
from models.base_forecaster import BaseTimeSeriesForecaster, CustomFrameworkMixin

# Import modular dimension manager
from layers.modular.dimensions.modular_dimension_manager import create_modular_dimension_manager

class ModularAutoformer(BaseTimeSeriesForecaster, CustomFrameworkMixin):
    """
    A completely modular Autoformer that can be configured to replicate
    any of the 7 in-house Autoformer models and create new variants.
    
    Now implements GCLI recommendations with structured configuration
    and "dumb assembler" pattern.
    """
    def __init__(self, configs: Union[Namespace, ModularAutoformerConfig]):
        super(ModularAutoformer, self).__init__(configs)
        logger.info("Initializing ModularAutoformer with GCLI structured configuration")

        # Set framework identification
        self.framework_type = 'custom'
        self.model_type = 'modular_autoformer'

        # Convert legacy Namespace or dict to structured config if needed
        if isinstance(configs, Namespace):
            self.structured_config = self._convert_namespace_to_structured(configs)
            self.legacy_configs = configs
        elif isinstance(configs, dict):
            from types import SimpleNamespace
            legacy_ns = SimpleNamespace(**configs)
            self.structured_config = self._convert_namespace_to_structured(legacy_ns)
            self.legacy_configs = legacy_ns
        else:
            self.structured_config = configs
            self.legacy_configs = configs.to_namespace() if hasattr(configs, 'to_namespace') else None
        
        # Initialize component registry
        register_all_components()
        self.assembler = ModularAssembler(component_registry)
        
        # Check if we should use a modular backbone
        self.use_backbone_component = getattr(self.legacy_configs, 'use_backbone_component', False)
        self.backbone_type = getattr(self.legacy_configs, 'backbone_type', None)
        
        if self.use_backbone_component and self.backbone_type:
            logger.info(f"Using modular backbone: {self.backbone_type}")
            self._initialize_with_backbone()
        else:
            # Use "dumb assembler" pattern to build the model
            self._initialize_with_assembler()
            
    def _convert_namespace_to_structured(self, ns_config: Namespace) -> ModularAutoformerConfig:
        """Convert legacy Namespace configuration to structured Pydantic config"""
        # This is a compatibility layer - extract parameters from Namespace
        from configs.schemas import (
            AttentionConfig, DecompositionConfig, EncoderConfig, DecoderConfig,
            SamplingConfig, OutputHeadConfig, LossConfig, BayesianConfig,
            BackboneConfig
        )
        
        # Map string types to ComponentType enums
        type_mapping = {
            'multi_head': ComponentType.MULTI_HEAD,
            'autocorrelation': ComponentType.AUTOCORRELATION,
            'autocorrelation_layer': ComponentType.AUTOCORRELATION,
            'adaptive_autocorrelation_layer': ComponentType.ADAPTIVE_AUTOCORRELATION,
            'cross_resolution_attention': ComponentType.CROSS_RESOLUTION,
            'moving_avg': ComponentType.MOVING_AVG,
            'series_decomp': ComponentType.MOVING_AVG,
            'stable_decomp': ComponentType.MOVING_AVG,
            'learnable_decomp': ComponentType.LEARNABLE_DECOMP,
            'wavelet_decomp': ComponentType.WAVELET_DECOMP,
            'standard': ComponentType.STANDARD_ENCODER,  # Default for encoder
            'enhanced': ComponentType.ENHANCED_ENCODER,
            'hierarchical': ComponentType.HIERARCHICAL_ENCODER,
            'deterministic': ComponentType.DETERMINISTIC,
            'bayesian': ComponentType.BAYESIAN,
            'quantile': ComponentType.QUANTILE,
            'mse': ComponentType.MSE,
            'bayesian_mse': ComponentType.BAYESIAN_MSE,
            'bayesian_quantile': ComponentType.BAYESIAN_QUANTILE,
        }
        
        # Decoder type mapping
        decoder_type_mapping = {
            'standard': ComponentType.STANDARD_DECODER,
            'enhanced': ComponentType.ENHANCED_DECODER,
            'hierarchical': ComponentType.HIERARCHICAL_DECODER,
        }
        
        # Output head type mapping
        output_head_type_mapping = {
            'standard': ComponentType.STANDARD_HEAD,
            'quantile': ComponentType.QUANTILE,
        }
        
        # Extract basic parameters
        seq_len = getattr(ns_config, 'seq_len', 96)
        pred_len = getattr(ns_config, 'pred_len', 24)
        label_len = getattr(ns_config, 'label_len', 48)
        enc_in = getattr(ns_config, 'enc_in', 7)
        dec_in = getattr(ns_config, 'dec_in', 7)
        c_out = getattr(ns_config, 'c_out', 7)
        c_out_evaluation = getattr(ns_config, 'c_out_evaluation', 7)
        # If quantiles are requested, expand model c_out to targets * num_quantiles
        ns_quantile_levels = getattr(ns_config, 'quantile_levels', None)
        if ns_quantile_levels:
            try:
                num_quantiles = len(ns_quantile_levels)
                if num_quantiles > 0:
                    c_out = c_out_evaluation * num_quantiles
                    logger.info(
                        f"Quantile mode detected: expanding c_out to {c_out_evaluation} * {num_quantiles} = {c_out}"
                    )
            except Exception:
                # Fallback: leave c_out unchanged if quantile_levels is malformed
                pass
        d_model = getattr(ns_config, 'd_model', 512)
        
        # Extract component types with fallbacks
        attention_type = type_mapping.get(
            getattr(ns_config, 'attention_type', 'autocorrelation'),
            ComponentType.AUTOCORRELATION
        )
        decomp_type = type_mapping.get(
            getattr(ns_config, 'decomposition_type', 'moving_avg'),
            ComponentType.MOVING_AVG
        )
        # Loss function type mapping (explicit for supported variants)
        loss_type_mapping = {
            'mse': ComponentType.MSE,
            'mae': ComponentType.MAE,
            'huber': ComponentType.MSE,  # default to MSE until Huber is added
            'quantile_loss': ComponentType.QUANTILE_LOSS,
            'bayesian_mse': ComponentType.BAYESIAN_MSE,
            'bayesian_quantile': ComponentType.BAYESIAN_QUANTILE,
            'bayesian': ComponentType.BAYESIAN_MSE,  # alias
        }
        
        encoder_type = type_mapping.get(
            getattr(ns_config, 'encoder_type', 'standard'),
            ComponentType.STANDARD_ENCODER
        )
        decoder_type = decoder_type_mapping.get(
            getattr(ns_config, 'decoder_type', 'standard'),
            ComponentType.STANDARD_DECODER
        )
        sampling_type = type_mapping.get(
            getattr(ns_config, 'sampling_type', 'deterministic'),
            ComponentType.DETERMINISTIC
        )
        output_head_type = output_head_type_mapping.get(
            getattr(ns_config, 'output_head_type', 'standard'),
            ComponentType.STANDARD_HEAD
        )
        loss_type = loss_type_mapping.get(
            getattr(ns_config, 'loss_function_type', 'mse'),
            ComponentType.MSE
        )
        
        # Build structured configuration
        structured_config = ModularAutoformerConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            label_len=label_len,
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            c_out_evaluation=c_out_evaluation,
            d_model=d_model,
            
            attention=AttentionConfig(
                type=attention_type,
                d_model=d_model,
                n_heads=getattr(ns_config, 'n_heads', getattr(ns_config, 'num_heads', None)),
                num_heads=getattr(ns_config, 'num_heads', getattr(ns_config, 'n_heads', None)),
                # Some attention implementations read seq_len from config; allowed via extra fields
                seq_len=seq_len,
                dropout=getattr(ns_config, 'dropout', 0.1),
                factor=getattr(ns_config, 'factor', 1),
                output_attention=getattr(ns_config, 'output_attention', False)
            ),
            
            decomposition=DecompositionConfig(
                type=decomp_type,
                kernel_size=getattr(ns_config, 'moving_avg', 25)
            ),
            
            encoder=EncoderConfig(
                type=encoder_type,
                num_encoder_layers=getattr(ns_config, 'e_layers', 2),
                d_ff=getattr(ns_config, 'd_ff', 2048),
                dropout=getattr(ns_config, 'dropout', 0.1),
                activation=getattr(ns_config, 'activation', 'gelu'),
                d_model=d_model,
                n_heads=getattr(ns_config, 'n_heads', getattr(ns_config, 'num_heads', None)),
            ),
            
            decoder=DecoderConfig(
                type=decoder_type,
                num_decoder_layers=getattr(ns_config, 'd_layers', 1),
                d_ff=getattr(ns_config, 'd_ff', 2048),
                dropout=getattr(ns_config, 'dropout', 0.1),
                activation=getattr(ns_config, 'activation', 'gelu'),
                c_out=c_out,
                d_model=d_model,
                n_heads=getattr(ns_config, 'n_heads', getattr(ns_config, 'num_heads', None))
            ),
            
            sampling=SamplingConfig(
                type=sampling_type,
                n_samples=getattr(ns_config, 'n_samples', 50),
                quantile_levels=getattr(ns_config, 'quantile_levels', None)
            ),
            
            output_head=OutputHeadConfig(
                type=output_head_type,
                d_model=d_model,
                c_out=c_out_evaluation,  # Use evaluation targets for output head
                num_quantiles=len(getattr(ns_config, 'quantile_levels', [])) if getattr(ns_config, 'quantile_levels', None) else None
            ),
            
            loss=LossConfig(
                type=loss_type,
                quantiles=getattr(ns_config, 'quantile_levels', None)
            ),
            
            bayesian=BayesianConfig(
                enabled=len(getattr(ns_config, 'bayesian_layers', [])) > 0,
                layers_to_convert=getattr(ns_config, 'bayesian_layers', [])
            ),
            
            backbone=BackboneConfig(
                use_backbone=getattr(ns_config, 'use_backbone_component', False),
                type=type_mapping.get(getattr(ns_config, 'backbone_type', None), None) if getattr(ns_config, 'backbone_type', None) else None
            ),
            
            quantile_levels=getattr(ns_config, 'quantile_levels', None),
            embed=getattr(ns_config, 'embed', 'timeF'),
            freq=getattr(ns_config, 'freq', 'h'),
            dropout=getattr(ns_config, 'dropout', 0.1),
        )
        
        return structured_config
        
    def _initialize_with_assembler(self):
        """Initialize model using the GCLI "dumb assembler" pattern"""
        logger.info("Assembling model using structured configuration")
        
        # Use the assembler to build the model
        self.assembled_model = self.assembler.assemble_model(self.structured_config)
        
        # Set up embedding layers (still needed for time series data)
        self.enc_embedding = DataEmbedding_wo_pos(
            self.structured_config.enc_in, 
            self.structured_config.d_model, 
            self.structured_config.embed, 
            self.structured_config.freq,
            self.structured_config.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            self.structured_config.dec_in, 
            self.structured_config.d_model, 
            self.structured_config.embed, 
            self.structured_config.freq,
            self.structured_config.dropout
        )
        
        # Store component references for compatibility
        self.attention = self.assembled_model.attention
        self.decomposition = self.assembled_model.decomposition
        self.encoder = self.assembled_model.encoder
        self.decoder = self.assembled_model.decoder
        self.sampling = self.assembled_model.sampling
        self.output_head = self.assembled_model.output_head
        self.loss_component = self.assembled_model.loss_component
        
        logger.info("Model assembly complete using GCLI dumb assembler pattern")
        
    def _initialize_with_backbone(self):
        """Initialize with modular backbone component (e.g., ChronosX)"""
        logger.info("Using backbone component - legacy mode (will be migrated to GCLI)")
        
        # Legacy backbone initialization for backward compatibility
        # TODO: Migrate to GCLI component system
        
        # Traditional components (may be used differently with backbone)
        from layers.modular.loss import get_loss_component
        from layers.modular.sampling import get_sampling_component
        from layers.modular.output_heads import get_output_head_component
        
        self.loss_function, _ = get_loss_component(
            self.legacy_configs.loss_function_type, 
            **getattr(self.legacy_configs, 'loss_params', {})
        )
        
        # Embeddings (may not be needed with some backbones)
        if not self._backbone_handles_embedding():
            self.enc_embedding = DataEmbedding_wo_pos(
                self.legacy_configs.enc_in, 
                self.legacy_configs.d_model, 
                self.legacy_configs.embed, 
                self.legacy_configs.freq, 
                self.legacy_configs.dropout
            )
            self.dec_embedding = DataEmbedding_wo_pos(
                self.legacy_configs.dec_in, 
                self.legacy_configs.d_model, 
                self.legacy_configs.embed, 
                self.legacy_configs.freq, 
                self.legacy_configs.dropout
            )
        
        # Sampling and output head
        self.sampling = get_sampling_component(
            self.legacy_configs.sampling_type, 
            **getattr(self.legacy_configs, 'sampling_params', {})
        )
        self.output_head = get_output_head_component(
            self.legacy_configs.output_head_type, 
            **getattr(self.legacy_configs, 'output_head_params', {})
        )
    
    def _initialize_traditional(self):
        """Initialize with traditional encoder-decoder architecture"""
        
        # Create dimension manager for clean parameter handling
        self.dim_manager = create_modular_dimension_manager(self.configs)
        
        # Validate component compatibility
        is_valid, issues = self.dim_manager.validate_component_compatibility({})
        if not is_valid:
            logger.warning(f"Dimension compatibility issues: {issues}")
        
        # --- Component Assembly from Registries ---
        self.loss_function, _ = get_loss_component(self.configs.loss_function_type, **self.configs.loss_params)
        
        self.enc_embedding = DataEmbedding_wo_pos(self.configs.enc_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.configs.dec_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)

        # Get component-specific dimensions from dimension manager
        encoder_decomp_dims = self.dim_manager.get_component_dimensions('decomposition', 'encoder')
        decoder_decomp_dims = self.dim_manager.get_component_dimensions('decomposition', 'decoder')
        init_decomp_dims = self.dim_manager.get_component_dimensions('decomposition', 'init')
        
        # Merge with configured parameters
        encoder_decomp_params = {**encoder_decomp_dims, **self.configs.decomposition_params}
        decoder_decomp_params = {**decoder_decomp_dims, **self.configs.decomposition_params}
        init_decomp_params = {**init_decomp_dims, **self.configs.init_decomposition_params}

        # Create decomposition components - let the factory filter parameters
        self.configs.encoder_params['attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        self.configs.encoder_params['decomp_comp'] = get_decomposition_component(self.configs.decomposition_type, **encoder_decomp_params)
        self.encoder = get_encoder_component(self.configs.encoder_type, **self.configs.encoder_params)

        self.configs.decoder_params['self_attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        self.configs.decoder_params['cross_attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        self.configs.decoder_params['decomp_comp'] = get_decomposition_component(self.configs.decomposition_type, **decoder_decomp_params)
        self.decoder = get_decoder_component(self.configs.decoder_type, **self.configs.decoder_params)

        self.sampling = get_sampling_component(self.configs.sampling_type, **self.configs.sampling_params)
        self.output_head = get_output_head_component(self.configs.output_head_type, **self.configs.output_head_params)
        
        # Handle init_decomp - may use different decomposition type than encoder/decoder
        init_decomp_type = getattr(self.configs, 'init_decomposition_type', self.configs.decomposition_type)
        self.init_decomp = get_decomposition_component(init_decomp_type, **init_decomp_params)
    
    def _backbone_handles_embedding(self) -> bool:
        """Check if the backbone handles its own embedding"""
        if hasattr(self.backbone, 'handles_embedding'):
            return self.backbone.handles_embedding()
        
        # ChronosX and other HF models typically handle their own embedding
        backbone_type = getattr(self.configs, 'backbone_type', '')
        return 'chronos' in backbone_type.lower() or 'hf_' in backbone_type.lower()

    def _prepare_data(self, x_enc_full, x_dec_full):
        if hasattr(self.configs, 'encoder_input_type') and self.configs.encoder_input_type == 'targets_only':
            x_enc = x_enc_full[:, :, :self.configs.c_out_evaluation]
        else:
            x_enc = x_enc_full

        if hasattr(self.configs, 'decoder_input_type') and self.configs.decoder_input_type == 'covariates_only':
            x_dec = x_dec_full[:, :, self.configs.c_out_evaluation:]
        else:
            x_dec = x_dec_full
            
        return x_enc, x_dec

    def _model_forward_pass(self, x_enc_full, x_mark_enc, x_dec_full, x_mark_dec):
        """Model forward pass - handles both traditional and backbone approaches"""
        
        if self.use_backbone_component and hasattr(self, 'backbone'):
            return self._backbone_forward_pass(x_enc_full, x_mark_enc, x_dec_full, x_mark_dec)
        else:
            return self._traditional_forward_pass(x_enc_full, x_mark_enc, x_dec_full, x_mark_dec)
    
    def _backbone_forward_pass(self, x_enc_full, x_mark_enc, x_dec_full, x_mark_dec):
        """Forward pass using modular backbone (e.g., ChronosX)"""
        x_enc, x_dec = self._prepare_data(x_enc_full, x_dec_full)
        
        # Check if backbone supports uncertainty prediction
        if (hasattr(self.backbone, 'supports_uncertainty') and 
            self.backbone.supports_uncertainty() and 
            hasattr(self.backbone, 'predict_with_uncertainty')):
            
            # Use backbone's uncertainty prediction
            uncertainty_results = self.backbone.predict_with_uncertainty(
                context=x_enc, 
                prediction_length=self.pred_len
            )
            
            # Store uncertainty results for later access
            self.last_uncertainty_results = uncertainty_results
            
            # Return the mean prediction
            return uncertainty_results['prediction']
        
        else:
            # Standard backbone forward pass
            if not self._backbone_handles_embedding():
                # Apply embedding if backbone doesn't handle it
                x_enc = self.enc_embedding(x_enc, x_mark_enc)
            
            # Process through backbone
            backbone_output = self.backbone(x_enc)
            
            # Apply processor if available
            if hasattr(self, 'processor') and self.processor is not None:
                processed_output = self.processor.process_sequence(
                    embedded_input=x_enc,
                    backbone_output=backbone_output,
                    target_length=self.pred_len
                )
            else:
                processed_output = backbone_output
            
            # Apply output head
            final_output = self.output_head(processed_output)
            
            return final_output
    
    def _traditional_forward_pass(self, x_enc_full, x_mark_enc, x_dec_full, x_mark_dec):
        """Traditional autoformer forward pass"""
        x_enc, x_dec = self._prepare_data(x_enc_full, x_dec_full)

        # The initial trend for the decoder must be in the d_model space.
        trend_init = torch.zeros(x_dec_full.shape[0], self.label_len + self.pred_len, self.configs.d_model, device=x_enc_full.device)

        # The seasonal component is passed to the decoder embedding.
        seasonal_init, _ = self.init_decomp(x_dec)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim=1)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        
        decoder_output = trend_part + seasonal_part
        final_output = self.output_head(decoder_output)
        
        return final_output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass using either GCLI assembled model or legacy approach
        """
        if hasattr(self, 'assembled_model'):
            # Use GCLI assembled model
            return self._forward_with_assembled_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        else:
            # Use legacy approach
            return self._forward_legacy(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
    
    def _forward_with_assembled_model(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass using the GCLI assembled model"""
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Prepare embeddings
            x_enc_embedded = self.enc_embedding(x_enc, x_mark_enc)
            x_dec_embedded = self.dec_embedding(x_dec, x_mark_dec)
            
            # Use assembled model forward pass
            results = self.assembled_model.forward(
                x_enc_embedded, x_mark_enc, x_dec_embedded, x_mark_dec, mask
            )
            
            # Apply sampling if needed
            if self.structured_config.sampling.type != ComponentType.DETERMINISTIC:
                results = self.sampling(lambda: results)
                if isinstance(results, dict):
                    prediction = results.get('prediction')
                    self.last_sampling_results = results
                    return prediction[:, -self.pred_len:, :]
            
            return results[:, -self.pred_len:, :]
        
        logger.warning(f"Task '{self.task_name}' not fully implemented for GCLI ModularAutoformer yet.")
        return None
    
    def _forward_legacy(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Legacy forward pass for backward compatibility"""
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            results = self.sampling(self._model_forward_pass, x_enc, x_mark_enc, x_dec, x_mark_dec)
            if isinstance(results, dict):
                prediction = results.get('prediction')
                self.last_sampling_results = results
                return prediction[:, -self.pred_len:, :]
            else:
                return results[:, -self.pred_len:, :]
        
        logger.warning(f"Task '{self.task_name}' not fully implemented for ModularAutoformer yet.")
        return None
    
    def supports_uncertainty(self) -> bool:
        """Check if model supports uncertainty quantification."""
        if hasattr(self, 'backbone'):
            return getattr(self.backbone, 'supports_uncertainty', lambda: False)()
        return hasattr(self, 'sampling') and getattr(self.configs, 'sampling_type', '') == 'bayesian'
    
    def supports_quantiles(self) -> bool:
        """Check if model supports quantile predictions."""
        # Prefer structured configuration if available
        try:
            if hasattr(self, 'structured_config'):
                from configs.schemas import ComponentType
                sc = self.structured_config
                # Quantile support if quantile output head or quantile-aware loss is configured
                if sc.output_head.type == ComponentType.QUANTILE:
                    return True
                if sc.loss.type in {ComponentType.QUANTILE_LOSS, ComponentType.BAYESIAN_QUANTILE}:
                    return True
                # Fallback: explicit quantile_levels attribute
                if getattr(sc, 'quantile_levels', None):
                    return True
            # Legacy fallback uses original string attribute
            loss_type = getattr(self.configs, 'loss_function_type', '')
            return 'quantile' in str(loss_type)
        except Exception:
            loss_type = getattr(self.configs, 'loss_function_type', '')
            return 'quantile' in str(loss_type)
    
    def get_uncertainty_results(self):
        """Get uncertainty results from backbone if available"""
        if hasattr(self, 'last_uncertainty_results'):
            return self.last_uncertainty_results
        return None
    
    def get_backbone_info(self):
        """Get information about the current backbone"""
        if hasattr(self, 'backbone'):
            return {
                'backbone_type': self.backbone_type,
                'backbone_class': self.backbone.__class__.__name__,
                'supports_uncertainty': getattr(self.backbone, 'supports_uncertainty', lambda: False)(),
                'supports_seq2seq': getattr(self.backbone, 'supports_seq2seq', lambda: False)(),
                'model_info': getattr(self.backbone, 'get_model_info', lambda: {})()
            }
        return {
            'backbone_type': 'traditional',
            'backbone_class': 'encoder_decoder',
            'supports_uncertainty': False,
            'supports_seq2seq': True
        }
    
    def get_component_info(self):
        """Get comprehensive component information"""
        info = {
            'architecture': 'traditional' if not self.use_backbone_component else 'modular_backbone',
            'backbone': self.get_backbone_info(),
            'components': {}
        }
        
        # Add component information
        if hasattr(self, 'processor'):
            info['components']['processor'] = {
                'type': getattr(self.processor, 'get_processor_type', lambda: 'unknown')() if self.processor else None,
                'class': self.processor.__class__.__name__ if self.processor else None
            }
        
        if hasattr(self, 'sampling'):
            info['components']['sampling'] = self.configs.sampling_type
        
        if hasattr(self, 'output_head'):
            info['components']['output_head'] = self.configs.output_head_type
        
        return info
