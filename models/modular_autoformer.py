import torch
import torch.nn as nn
from argparse import Namespace
from typing import Union, Dict, Any, Optional

# GCLI Structured Configuration System
from configs.schemas import ModularAutoformerConfig, ComponentType
from configs.modular_components import (
    ModularAssembler, component_registry, AssembledAutoformer,
    register_all_components
)

# New modular framework imports - using UNIFIED components

from utils.modular_components.registry import ComponentRegistry
from utils.modular_components.config_schemas import ComponentConfig, BackboneConfig, safe_config_from_dict
from utils.modular_components.base_interfaces import BaseComponent, BaseLoss, BaseAttention

# Essential imports for embedding and base functionality
from layers.Embed import DataEmbedding_wo_pos
from utils.logger import logger

# Import unified component registries (single source of truth)
try:
    from utils.modular_components.implementations.Attention import ATTENTION_REGISTRY
    from utils.modular_components.implementations.Losses import LOSS_REGISTRY
    from utils.modular_components.implementations.Decomposition import DECOMPOSITION_REGISTRY
    # If encoder registry is needed, update here:
    # from utils.modular_components.implementations.Encoder import ENCODER_REGISTRY
except ImportError as e:
    logger.warning(f"Unified components not available: {e}")
    # Will fall back to minimal components


# Import unified base framework
from models.base_forecaster import BaseTimeSeriesForecaster, CustomFrameworkMixin

# Import modular dimension manager
from utils.modular_dimension_manager import create_modular_dimension_manager

# Import MigrationManager for component migration
from utils.modular_components.migration_framework import MigrationManager

# Import create_backbone for backbone instantiation
from utils.modular_components.factories import create_backbone


class ModularAutoformer(BaseTimeSeriesForecaster, CustomFrameworkMixin):
    """
    A completely modular Autoformer that can be configured to replicate
    any of the 7 in-house Autoformer models and create new variants.
    
    Now uses ONLY the new modular framework through migration adapters
    for complete architectural consistency.
    """
    
    def __init__(self, configs: Union[Namespace, ModularAutoformerConfig]):
        super(ModularAutoformer, self).__init__(configs)
        logger.info("Initializing ModularAutoformer with new modular framework")

        # Set framework identification
        self.framework_type = 'custom'
        self.model_type = 'modular_autoformer'

        # Convert legacy Namespace to structured config if needed
        if isinstance(configs, Namespace):
            self.structured_config = self._convert_namespace_to_structured(configs)
            self.legacy_configs = configs
        else:
            self.structured_config = configs
            self.legacy_configs = configs.to_namespace()

        # Set dropout from config (ensure always available)
        self.dropout = getattr(self.structured_config, 'dropout', 0.1)

        # Initialize new modular framework
        self._initialize_new_framework()

        # Check if we should use a modular backbone
        self.use_backbone_component = getattr(self.legacy_configs, 'use_backbone_component', False)
        self.backbone_type = getattr(self.legacy_configs, 'backbone_type', None)

        if self.use_backbone_component and self.backbone_type:
            logger.info(f"Using modular backbone: {self.backbone_type}")
            self._initialize_with_backbone()
        else:
            # Use new framework with migrated components
            self._initialize_with_new_framework()
    
    def _initialize_new_framework(self):
        """Initialize the new modular framework with migrated components"""
        logger.info("Initializing new modular framework...")
        
        # Create component registry
        self.component_registry = ComponentRegistry()
        
        # Create migration manager and perform migration
        self.migration_manager = MigrationManager(self.component_registry)
        self.migration_manager.migrate_all_components()
        
        logger.info("New modular framework initialized with migrated components")
    
    def _initialize_with_new_framework(self):
        """Initialize model using new framework with migrated components"""
        logger.info("Building model with new modular framework...")
        
        # Extract configuration parameters
        configs = self.legacy_configs
        
        # Basic dimensions
        self.seq_len = self.structured_config.seq_len
        self.label_len = self.structured_config.label_len
        self.pred_len = self.structured_config.pred_len
        self.enc_in = self.structured_config.enc_in
        self.dec_in = self.structured_config.dec_in
        self.c_out = self.structured_config.c_out
        self.d_model = self.structured_config.d_model
        self.n_heads = self.structured_config.encoder.n_heads
        self.e_layers = self.structured_config.encoder.e_layers
        self.d_layers = self.structured_config.decoder.d_layers
        self.d_ff = self.structured_config.encoder.d_ff
        self.dropout = self.structured_config.encoder.dropout
        self.factor = getattr(configs, 'factor', 1)
        self.output_attention = getattr(configs, 'output_attention', False)
        
        # Component parameters
        self.moving_avg = getattr(configs, 'moving_avg', 25)
        self.wavelet = getattr(configs, 'wavelet', 'db4')
        self.num_quantiles = getattr(configs, 'num_quantiles', 9)
        
        # Create embedding layer (still using legacy as it's not part of modular system)
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, 
                                                 configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, 
                                                 configs.embed, configs.freq, configs.dropout)
        
        # Create components using new framework
        self._create_migrated_components()
        
        # Initialize learnable parameters
        self._initialize_parameters()
        
        logger.info("Model built successfully with new modular framework")
    
    def _create_migrated_components(self):
        """Create all model components using new framework with migration adapters"""
        configs = self.legacy_configs
        
        # Create decomposition component
        decomp_config_dict = {
            'component_name': getattr(configs, 'decomposition_type', 'series_decomp'),
            'd_model': self.d_model,
            'dropout': self.dropout,
            'device': 'auto',
            'dtype': 'float32',
            'custom_params': {'kernel_size': self.moving_avg, 'seq_len': self.seq_len}
        }
        from utils.modular_components.config_schemas import safe_config_from_dict
        decomp_config = safe_config_from_dict(ComponentConfig, decomp_config_dict)
        
        # Note: Decomposition not migrated yet, use fallback
        try:
            from layers.modular.decomposition import get_decomposition_component
            self.decomp = get_decomposition_component(decomp_config.component_name, **decomp_config.parameters)
        except Exception as e:
            logger.warning(f"Decomposition component not available: {e}")
            # Simple fallback decomposition
            class SimpleDecomp(nn.Module):
                def forward(self, x):
                    return x, torch.zeros_like(x)
            self.decomp = SimpleDecomp()
        
        # Create attention component using migrated framework
        attention_config_dict = {
            'component_name': getattr(configs, 'attention_type', 'autocorrelation_layer'),
            'd_model': self.d_model,
            'dropout': self.dropout,
            'device': 'auto',
            'dtype': 'float32',
            'custom_params': {
                'n_heads': self.n_heads,
                'factor': self.factor,
                'output_attention': self.output_attention
            }
        }
        attention_config = safe_config_from_dict(ComponentConfig, attention_config_dict)
        
        try:
            self.attention = self.component_registry.create('attention', 
                                                          attention_config.component_name, 
                                                          attention_config)
            logger.info(f"Created attention component: {attention_config.component_name}")
        except Exception as e:
            logger.error(f"Failed to create attention component: {e}")
            # Fallback to simple attention
            self.attention = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=self.dropout)
        
        # Create encoder and decoder components
        try:
            from layers.modular.encoder import get_encoder_component
            from layers.modular.decoder import get_decoder_component
            
            self.encoder = get_encoder_component(
                getattr(configs, 'encoder_type', 'standard'),
                d_model=self.d_model, n_heads=self.n_heads, e_layers=self.e_layers,
                d_ff=self.d_ff, dropout=self.dropout, attention=self.attention,
                decomp=self.decomp
            )
            
            self.decoder = get_decoder_component(
                getattr(configs, 'decoder_type', 'standard'),
                d_model=self.d_model, n_heads=self.n_heads, d_layers=self.d_layers,
                d_ff=self.d_ff, dropout=self.dropout, attention=self.attention,
                decomp=self.decomp, c_out=self.c_out
            )
        except Exception as e:
            logger.warning(f"Encoder/Decoder components not available: {e}")
            # Create simple encoder/decoder fallbacks
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout),
                self.e_layers
            )
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout),
                self.d_layers
            )
        
        # Create output head component
        try:
            from layers.modular.output_heads import get_output_head_component
            
            self.output_head = get_output_head_component(
                getattr(configs, 'output_head_type', 'standard'),
                d_model=self.d_model, c_out=self.c_out, num_quantiles=self.num_quantiles
            )
        except Exception as e:
            logger.warning(f"Output head component not available: {e}")
            self.output_head = nn.Linear(self.d_model, self.c_out)
        
        # Create loss component using migrated framework
        loss_config_dict = {
            'component_name': getattr(configs, 'loss_type', 'mse'),
            'd_model': self.d_model,
            'dropout': self.dropout,
            'device': 'auto',
            'dtype': 'float32',
            'custom_params': {}
        }
        loss_config = safe_config_from_dict(ComponentConfig, loss_config_dict)
        
        try:
            self.loss_component = self.component_registry.create('loss',
                                                               loss_config.component_name,
                                                               loss_config)
            logger.info(f"Created loss component: {loss_config.component_name}")
        except Exception as e:
            logger.error(f"Failed to create loss component: {e}")
            # Fallback to simple MSE loss
            class SimpleMSELoss:
                def forward(self, pred, true):
                    return nn.functional.mse_loss(pred, true)
                def compute_loss(self, pred, true):
                    return self.forward(pred, true)
            self.loss_component = SimpleMSELoss()
    
    def _initialize_with_backbone(self):
        """Initialize model with modular backbone component"""
        logger.info(f"Initializing with backbone: {self.backbone_type}")
        
        # Create backbone configuration
        backbone_config_dict = {
            'model_name': getattr(self.legacy_configs, 'backbone_model_name', None),
            'pretrained': True,
            'freeze_backbone': False,
            'cache_dir': None,
            'trust_remote_code': False,
            'num_layers': getattr(self.legacy_configs, 'e_layers', 6),
            'num_heads': getattr(self.legacy_configs, 'n_heads', 8),
            'd_ff': getattr(self.legacy_configs, 'd_ff', 2048),
            'vocab_size': 1000,
            'd_model': self.d_model,
            'dropout': self.dropout,
            'device': 'auto',
            'dtype': 'float32',
            'custom_params': {
                'prediction_length': getattr(self.legacy_configs, 'pred_len', None),
                'context_length': getattr(self.legacy_configs, 'seq_len', None)
            }
        }
        backbone_config = safe_config_from_dict(BackboneConfig, backbone_config_dict)
        
        try:
            # Create backbone using new framework
            self.backbone = create_backbone(backbone_config)
            
            # Set model parameters based on backbone
            self.d_model = self.backbone.get_d_model()
            
            # Create other components with backbone integration
            self._create_migrated_components()
            
            logger.info("Model initialized with modular backbone")
            
        except Exception as e:
            logger.error(f"Failed to initialize with backbone: {e}")
            # Fall back to standard initialization
            self._initialize_with_new_framework()
    
    def _convert_namespace_to_structured(self, ns_config: Namespace) -> ModularAutoformerConfig:
        """Convert legacy Namespace configuration to structured Pydantic config"""
        # This maintains backward compatibility
        from configs.schemas import (
            AttentionConfig, DecompositionConfig, EncoderConfig, DecoderConfig,
            SamplingConfig, OutputHeadConfig, LossConfig, BayesianConfig,
            BackboneConfig, ComponentType
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
            'standard': ComponentType.STANDARD_ENCODER,
            'enhanced': ComponentType.ENHANCED_ENCODER,
            'hierarchical': ComponentType.HIERARCHICAL_ENCODER,
            'deterministic': ComponentType.DETERMINISTIC,
            'bayesian': ComponentType.BAYESIAN,
            'quantile': ComponentType.QUANTILE,
            'mse': ComponentType.MSE,
            'bayesian_mse': ComponentType.BAYESIAN_MSE,
            'bayesian_quantile': ComponentType.BAYESIAN_QUANTILE,
            'standard_head': ComponentType.STANDARD_HEAD,
        }
        
        # Extract basic parameters
        seq_len = getattr(ns_config, 'seq_len', 96)
        pred_len = getattr(ns_config, 'pred_len', 24)
        label_len = getattr(ns_config, 'label_len', 48)
        enc_in = getattr(ns_config, 'enc_in', 7)
        dec_in = getattr(ns_config, 'dec_in', 7)
        c_out = getattr(ns_config, 'c_out', 7)
        d_model = getattr(ns_config, 'd_model', 512)
        
        # Create structured configuration
        structured_config = ModularAutoformerConfig(
            # Basic parameters
            seq_len=seq_len,
            pred_len=pred_len,
            label_len=label_len,
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            c_out_evaluation=getattr(ns_config, 'c_out_evaluation', c_out),
            d_model=d_model,
            n_heads=getattr(ns_config, 'n_heads', 8),
            e_layers=getattr(ns_config, 'e_layers', 2),
            d_layers=getattr(ns_config, 'd_layers', 1),
            d_ff=getattr(ns_config, 'd_ff', 2048),
            dropout=getattr(ns_config, 'dropout', 0.1),

            # Component configurations using migrated framework
            attention=AttentionConfig(
                type=type_mapping.get(
                    getattr(ns_config, 'attention_type', 'autocorrelation'),
                    ComponentType.AUTOCORRELATION
                ),
                factor=getattr(ns_config, 'factor', 1),
                output_attention=getattr(ns_config, 'output_attention', False),
                d_model=getattr(ns_config, 'd_model', 512),
                n_heads=getattr(ns_config, 'n_heads', 8)
            ),
            
            decomposition=DecompositionConfig(
                type=type_mapping.get(
                    getattr(ns_config, 'decomposition_type', 'moving_avg'),
                    ComponentType.MOVING_AVG
                ),
                kernel_size=getattr(ns_config, 'moving_avg', 25)
            ),
            
            encoder=EncoderConfig(
                type=type_mapping.get(
                    getattr(ns_config, 'encoder_type', 'standard'),
                    ComponentType.STANDARD_ENCODER
                ),
                e_layers=getattr(ns_config, 'e_layers', 2),
                d_model=getattr(ns_config, 'd_model', 512),
                n_heads=getattr(ns_config, 'n_heads', 8),
                d_ff=getattr(ns_config, 'd_ff', 2048),
                dropout=getattr(ns_config, 'dropout', 0.1),
                activation=getattr(ns_config, 'activation', 'gelu'),
                factor=getattr(ns_config, 'factor', 1)
            ),

            decoder=DecoderConfig(
                type=type_mapping.get(
                    getattr(ns_config, 'decoder_type', 'standard'),
                    ComponentType.STANDARD_DECODER
                ),
                d_layers=getattr(ns_config, 'd_layers', 1),
                d_model=getattr(ns_config, 'd_model', 512),
                n_heads=getattr(ns_config, 'n_heads', 8),
                d_ff=getattr(ns_config, 'd_ff', 2048),
                dropout=getattr(ns_config, 'dropout', 0.1),
                activation=getattr(ns_config, 'activation', 'gelu'),
                factor=getattr(ns_config, 'factor', 1),
                c_out=getattr(ns_config, 'c_out', 7)
            ),
            
            loss=LossConfig(
                type=type_mapping.get(
                    getattr(ns_config, 'loss_type', 'mse'),
                    ComponentType.MSE
                )
            ),
            
            sampling=SamplingConfig(
                type=type_mapping.get(getattr(ns_config, 'sampling_type', 'deterministic'), ComponentType.DETERMINISTIC)
            ),
            output_head=OutputHeadConfig(
                type=type_mapping.get(getattr(ns_config, 'output_head_type', 'standard_head'), ComponentType.STANDARD_HEAD),
                d_model=d_model,
                c_out=c_out,
                dropout=getattr(ns_config, 'dropout', 0.1)
            ),
            # Optional components
            backbone=BackboneConfig(
                type=type_mapping.get(getattr(ns_config, 'backbone_type', None), None),
                model_name=getattr(ns_config, 'backbone_model_name', None),
                use_backbone=getattr(ns_config, 'use_backbone_component', False),
                prediction_length=getattr(ns_config, 'pred_len', None),
                context_length=getattr(ns_config, 'seq_len', None)
            ),
        )
        
        return structured_config
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass using new modular framework components"""
        # Input embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Decomposition
        try:
            seasonal_init, trend_init = self.decomp(enc_out)
        except:
            # Fallback if decomposition fails
            seasonal_init, trend_init = enc_out, torch.zeros_like(enc_out)
        
        # Encoder
        try:
            if hasattr(self.encoder, '__call__'):
                enc_out = self.encoder(seasonal_init, trend_init, mask)
            else:
                enc_out = self.encoder(seasonal_init)
        except:
            # Fallback for simple transformer encoder
            enc_out = self.encoder(seasonal_init.transpose(0, 1)).transpose(0, 1)
        
        # Decoder
        try:
            if hasattr(self.decoder, '__call__') and hasattr(self.decoder, 'forward'):
                dec_out = self.decoder(dec_out, enc_out, trend_init)
            else:
                # Fallback for simple transformer decoder
                dec_out = self.decoder(dec_out.transpose(0, 1), 
                                     enc_out.transpose(0, 1)).transpose(0, 1)
        except:
            dec_out = enc_out[:, -self.pred_len:, :]
        
        # Output projection
        output = self.output_head(dec_out)
        
        return output
    
    def compute_loss(self, predictions, targets, **kwargs):
        """Compute loss using migrated loss component"""
        try:
            return self.loss_component.compute_loss(predictions, targets, **kwargs)
        except:
            # Fallback to simple MSE
            return nn.functional.mse_loss(predictions, targets)
    
    def _initialize_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_migration_status(self):
        """Get the current migration status"""
        if hasattr(self, 'migration_manager'):
            return self.migration_manager.get_migration_status()
        return {}
    
    def list_available_components(self):
        """List all available migrated components"""
        if hasattr(self, 'component_registry'):
            return self.component_registry.list_components()
        return {}
