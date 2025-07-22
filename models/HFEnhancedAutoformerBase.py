"""
HF Enhanced Autoformer Base - Foundation for Advanced HF Models (Modularized)
"""

import torch.nn as nn
from argparse import Namespace

from .HFEnhancedAutoformer import HFEnhancedAutoformer
from utils.modular_components.factories import create_loss
from utils.modular_components.config_schemas import LossConfig
from utils.logger import logger

class HFEnhancedAutoformerBase(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.configs = configs
        self.original_c_out = getattr(configs, 'c_out_evaluation', configs.c_out)
        
        hf_configs_dict = vars(configs).copy()
        hf_configs_dict['c_out'] = self.original_c_out
        self.hf_backbone = HFEnhancedAutoformer(Namespace(**hf_configs_dict))
        
        self.loss_manager = self._create_loss_manager(configs)
        self.feature_processors = nn.ModuleDict()
        
        logger.info(f"HFEnhancedAutoformerBase initialized successfully")
        
    def _create_loss_manager(self, configs):
        loss_type = getattr(configs, 'loss', 'mse').lower()
        loss_config = LossConfig(loss_type=loss_type, custom_params=vars(configs))
        
        if getattr(configs, 'uncertainty_method', None) in ['bayesian', 'dropout']:
            loss_config.loss_type = f"bayesian_{loss_type}"
        
        return create_loss(loss_config)
        
    def add_feature_processor(self, name, processor):
        self.feature_processors[name] = processor
        logger.info(f"Added feature processor: {name}")
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        base_output = self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        
        processed_output = base_output
        for name, processor in self.feature_processors.items():
            processed_output = processor(processed_output, x_enc, x_mark_enc, x_dec, x_mark_dec)
            
        return processed_output
        
    def compute_loss(self, predictions, targets, return_components=False):
        if hasattr(self.loss_manager, 'forward') and 'model' in self.loss_manager.forward.__code__.co_varnames:
            return self.loss_manager(model=self, pred_result=predictions, true=targets)
        else:
            pred_tensor = predictions['prediction'] if isinstance(predictions, dict) else predictions
            return self.loss_manager(pred_tensor, targets)

Model = HFEnhancedAutoformerBase