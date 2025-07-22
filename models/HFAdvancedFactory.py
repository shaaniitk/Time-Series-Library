"""
Advanced HF Model Factory (Modularized)
"""

import torch.nn as nn
from typing import Dict, Any, Optional
from argparse import Namespace

from .HFEnhancedAutoformer import HFEnhancedAutoformer
from .HFBayesianExtensions import create_bayesian_extension, create_quantile_extension
from .HFHierarchicalExtensions import create_wavelet_processor, create_hierarchical_processor
from utils.modular_components.factories import create_loss
from utils.modular_components.config_schemas import LossConfig
from utils.logger import logger

class HFAdvancedModelWrapper(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = self._ensure_namespace(configs)
        self.hf_backbone = HFEnhancedAutoformer(self.configs)
        self.extensions = nn.ModuleDict()
        self._init_extensions()
        self._init_loss_manager()
        self.model_info = self._get_model_info()
        
    def _ensure_namespace(self, configs):
        return configs if isinstance(configs, Namespace) else Namespace(**configs)
            
    def _init_extensions(self):
        if getattr(self.configs, 'use_bayesian', False):
            self.extensions['bayesian'] = create_bayesian_extension(self.configs, self.hf_backbone)
        if getattr(self.configs, 'use_quantile', False):
            self.extensions['quantile'] = create_quantile_extension(self.configs)
        if getattr(self.configs, 'use_wavelet', False):
            self.extensions['wavelet'] = create_wavelet_processor(self.configs)
        if getattr(self.configs, 'use_hierarchical', False):
            self.extensions['hierarchical'] = create_hierarchical_processor(self.configs)
            
    def _init_loss_manager(self):
        loss_type = getattr(self.configs, 'loss_function', 'mse')
        loss_config = LossConfig(loss_type=loss_type, custom_params=vars(self.configs))
        self.loss_fn = create_loss(loss_config)
            
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        processed_input = self._apply_preprocessing_extensions(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if 'bayesian' in self.extensions:
            backbone_output = self.extensions['bayesian'](**processed_input, mask=mask)
        else:
            backbone_output = self.hf_backbone(**processed_input, mask=mask)
            
        return self._apply_postprocessing_extensions(backbone_output, processed_input)
        
    def _apply_preprocessing_extensions(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if 'wavelet' in self.extensions:
            x_enc = self.extensions['wavelet'](x_enc)
        if 'hierarchical' in self.extensions:
            x_enc = self.extensions['hierarchical'](x_enc)
        return {'x_enc': x_enc, 'x_mark_enc': x_mark_enc, 'x_dec': x_dec, 'x_mark_dec': x_mark_dec}
        
    def _apply_postprocessing_extensions(self, backbone_output, input_data):
        if 'quantile' in self.extensions:
            return self.extensions['quantile'](backbone_output, **input_data)
        return backbone_output
        
    def compute_loss(self, predictions, targets, x_enc=None):
        pred_tensor = predictions['prediction'] if isinstance(predictions, dict) else predictions
        base_loss = self.loss_fn(pred_tensor, targets)
        
        total_loss = base_loss
        loss_components = {'base_loss': base_loss}
        
        if 'bayesian' in self.extensions:
            kl_loss = self.extensions['bayesian'].get_kl_loss()
            total_loss += kl_loss
            loss_components['kl_loss'] = kl_loss
            
        return total_loss, loss_components
        
    def _get_model_info(self):
        return {
            'backbone': type(self.hf_backbone).__name__,
            'extensions': list(self.extensions.keys()),
            'parameters': {'total': sum(p.numel() for p in self.parameters())},
            'capabilities': {
                'uncertainty_quantification': 'bayesian' in self.extensions,
                'quantile_regression': 'quantile' in self.extensions,
                'multi_scale_processing': 'hierarchical' in self.extensions,
                'wavelet_analysis': 'wavelet' in self.extensions
            }
        }

def create_hf_model_from_config(configs, model_type='auto'):
    if model_type == 'auto':
        has_bayesian = getattr(configs, 'use_bayesian', False)
        has_hierarchical = getattr(configs, 'use_hierarchical', False)
        has_quantile = getattr(configs, 'use_quantile', False)
        
        if has_bayesian and has_hierarchical and has_quantile: model_type = 'full'
        elif has_bayesian: model_type = 'bayesian'
        elif has_hierarchical: model_type = 'hierarchical'
        elif has_quantile: model_type = 'quantile'
        else: model_type = 'standard'
            
    if model_type == 'standard': return HFEnhancedAutoformer(configs)
    
    model_configs = Namespace(**vars(configs))
    if model_type == 'bayesian': model_configs.use_bayesian = True
    elif model_type == 'hierarchical': model_configs.use_hierarchical = True
    elif model_type == 'quantile': model_configs.use_quantile = True
    elif model_type == 'full':
        model_configs.use_bayesian = model_configs.use_hierarchical = model_configs.use_quantile = True
    
    return HFAdvancedModelWrapper(model_configs)