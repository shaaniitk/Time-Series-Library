"""
HF Enhanced Autoformer Advanced - Complete Implementation (Modularized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from argparse import Namespace

from .HFEnhancedAutoformer import HFEnhancedAutoformer
from layers.BayesianLayers import convert_to_bayesian, collect_kl_divergence, DropoutSampling
from utils.modular_components.factories import create_loss, create_processor
from utils.modular_components.config_schemas import LossConfig, ProcessorConfig
from utils.logger import logger

class HFEnhancedAutoformerAdvanced(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        logger.info("Initializing HFEnhancedAutoformerAdvanced")
        
        self.configs = configs
        self.uncertainty_method = getattr(configs, 'uncertainty_method', None)
        self.n_samples = getattr(configs, 'n_samples', 10)
        self.kl_weight = getattr(configs, 'kl_weight', 1e-5)
        self.use_wavelets = getattr(configs, 'use_wavelets', False)
        
        self.hf_backbone = HFEnhancedAutoformer(configs)
        
        self._setup_advanced_features(configs)
        self._create_loss_manager(configs)
        
    def _setup_advanced_features(self, configs):
        if self.uncertainty_method == 'bayesian':
            self.bayesian_layers = convert_to_bayesian(self.hf_backbone, getattr(configs, 'bayesian_layers', ['output_projection']))
        elif self.uncertainty_method == 'dropout':
            self.mc_dropout1 = DropoutSampling(p=0.1)
            self.mc_dropout3 = DropoutSampling(p=0.1)
            
        if self.use_wavelets:
            processor_config = ProcessorConfig(processor_type='wavelet', custom_params={'wavelet_type': getattr(configs, 'wavelet_type', 'db4'), 'levels': getattr(configs, 'n_levels', 3)})
            self.wavelet_processor = create_processor(processor_config)
            
    def _create_loss_manager(self, configs):
        loss_type = getattr(configs, 'loss', 'mse').lower()
        loss_config = LossConfig(loss_type=loss_type, custom_params=vars(configs))
        if self.uncertainty_method:
            loss_config.custom_params['kl_weight'] = self.kl_weight
            loss_config.loss_type = f"bayesian_{loss_type}"
        self.loss_manager = create_loss(loss_config)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_wavelets:
            x_enc = self.wavelet_processor(x_enc)
        
        if self.uncertainty_method:
            return self._forward_with_uncertainty(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        else:
            return {'prediction': self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)}
        
    def _forward_with_uncertainty(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if self.uncertainty_method == 'bayesian':
            predictions = [self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec) for _ in range(self.n_samples)]
        elif self.uncertainty_method == 'dropout':
            predictions = []
            self.train()
            with torch.no_grad():
                for _ in range(self.n_samples):
                    x_enc_dropped = self.mc_dropout1(x_enc)
                    pred = self.hf_backbone(x_enc_dropped, x_mark_enc, x_dec, x_mark_dec)
                    pred_dropped = self.mc_dropout3(pred)
                    predictions.append(pred_dropped)
            self.eval()
        
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack)
        
    def _compute_uncertainty_statistics(self, pred_stack):
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        confidence_intervals = {}
        for conf_level in [0.68, 0.95, 0.99]:
            alpha = 1 - conf_level
            lower_bound = torch.quantile(pred_stack, (alpha / 2), dim=0)
            upper_bound = torch.quantile(pred_stack, (1 - alpha / 2), dim=0)
            confidence_intervals[f'{int(conf_level * 100)}%'] = {'lower': lower_bound, 'upper': upper_bound}
            
        return {'prediction': mean_pred, 'uncertainty': total_std, 'variance': total_variance, 'confidence_intervals': confidence_intervals, 'predictions_samples': pred_stack}
        
    def compute_loss(self, predictions, targets, return_components=False):
        if hasattr(self.loss_manager, 'forward') and 'model' in self.loss_manager.forward.__code__.co_varnames:
            return self.loss_manager(model=self, pred_result=predictions, true=targets)
        else:
            pred_tensor = predictions['prediction'] if isinstance(predictions, dict) else predictions
            return self.loss_manager(pred_tensor, targets)
            
    def get_kl_loss(self, max_kl_value=1e6):
        if self.uncertainty_method != 'bayesian':
            return 0.0
        kl_div = collect_kl_divergence(self.hf_backbone)
        kl_div_clipped = torch.clamp(kl_div, max=max_kl_value)
        return kl_div_clipped * self.kl_weight

Model = HFEnhancedAutoformerAdvanced