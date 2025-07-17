import torch
import torch.nn as nn
from argparse import Namespace

from layers.modular.decomposition import get_decomposition_component
from layers.modular.encoder import get_encoder_component
from layers.modular.decoder import get_decoder_component
from layers.modular.attention import get_attention_component
from layers.modular.sampling import get_sampling_component
from layers.modular.output_heads import get_output_head_component
from layers.modular.losses import get_loss_component
from layers.Embed import DataEmbedding_wo_pos
from utils.logger import logger

# Import modular component registry
from utils.modular_components.registry import ComponentRegistry, create_global_registry
from utils.modular_components.example_components import register_example_components
from utils.modular_components.config_schemas import ComponentConfig

class ModularAutoformer(nn.Module):
    """
    A completely modular Autoformer that can be configured to replicate
    any of the 7 in-house Autoformer models and create new variants.
    
    Now supports ChronosX and other backbone options through the component registry.
    """
    def __init__(self, configs):
        super(ModularAutoformer, self).__init__()
        logger.info("Initializing ModularAutoformer with modular backbone support")

        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Initialize component registry if not already done
        self.registry = create_global_registry()
        
        # Check if we should use a modular backbone
        self.use_backbone_component = getattr(configs, 'use_backbone_component', False)
        self.backbone_type = getattr(configs, 'backbone_type', None)
        
        if self.use_backbone_component and self.backbone_type:
            logger.info(f"Using modular backbone: {self.backbone_type}")
            self._initialize_with_backbone()
        else:
            logger.info("Using traditional encoder-decoder architecture")
            self._initialize_traditional()

    def _initialize_with_backbone(self):
        """Initialize with modular backbone component (e.g., ChronosX)"""
        
        # Create backbone configuration
        backbone_config = ComponentConfig()
        backbone_config.d_model = self.configs.d_model
        backbone_config.seq_len = self.configs.seq_len
        backbone_config.pred_len = self.configs.pred_len
        
        # Add backbone-specific parameters
        if hasattr(self.configs, 'backbone_params'):
            for key, value in self.configs.backbone_params.items():
                setattr(backbone_config, key, value)
        
        # Create backbone component
        backbone_class = self.registry.get('backbone', self.backbone_type)
        self.backbone = backbone_class(backbone_config)
        
        # Traditional components (may be used differently with backbone)
        self.loss_function, _ = get_loss_component(self.configs.loss_function_type, **self.configs.loss_params)
        
        # Embeddings (may not be needed with some backbones)
        if not self._backbone_handles_embedding():
            self.enc_embedding = DataEmbedding_wo_pos(self.configs.enc_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(self.configs.dec_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
        
        # Sampling and output head
        self.sampling = get_sampling_component(self.configs.sampling_type, **self.configs.sampling_params)
        self.output_head = get_output_head_component(self.configs.output_head_type, **self.configs.output_head_params)
        
        # Processor components (if using modular processing)
        if hasattr(self.configs, 'processor_type'):
            processor_config = ComponentConfig()
            processor_config.d_model = self.configs.d_model
            processor_config.pred_len = self.configs.pred_len
            
            processor_class = self.registry.get('processor', self.configs.processor_type)
            self.processor = processor_class(processor_config)
        else:
            self.processor = None
    
    def _initialize_traditional(self):
        """Initialize with traditional encoder-decoder architecture"""
        
        # --- Component Assembly from Registries ---
        self.loss_function, _ = get_loss_component(self.configs.loss_function_type, **self.configs.loss_params)
        
        self.enc_embedding = DataEmbedding_wo_pos(self.configs.enc_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.configs.dec_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)

        self.configs.encoder_params['attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        self.configs.encoder_params['decomp_comp'] = get_decomposition_component(self.configs.decomposition_type, **self.configs.encoder_params.get('decomp_params', self.configs.decomposition_params))
        self.encoder = get_encoder_component(self.configs.encoder_type, **self.configs.encoder_params)

        self.configs.decoder_params['self_attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        self.configs.decoder_params['cross_attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        self.configs.decoder_params['decomp_comp'] = get_decomposition_component(self.configs.decomposition_type, **self.configs.decoder_params.get('decomp_params', self.configs.decomposition_params))
        self.decoder = get_decoder_component(self.configs.decoder_type, **self.configs.decoder_params)

        self.sampling = get_sampling_component(self.configs.sampling_type, **self.configs.sampling_params)
        self.output_head = get_output_head_component(self.configs.output_head_type, **self.configs.output_head_params)
        self.init_decomp = get_decomposition_component(self.configs.decomposition_type, **self.configs.init_decomposition_params)
    
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
