"""
Bayesian Extension Module for HF Models (Modularized)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from argparse import Namespace

from layers.BayesianLayers import convert_to_bayesian, collect_kl_divergence, DropoutSampling
from utils.modular_components.factories import create_output
from utils.modular_components.config_schemas import OutputConfig
from utils.logger import logger

class BayesianExtension(nn.Module):
    def __init__(self, configs, hf_backbone):
        super().__init__()
        
        self.uncertainty_method = getattr(configs, 'uncertainty_method', 'dropout')
        self.n_samples = getattr(configs, 'n_samples', 10)
        self.kl_weight = getattr(configs, 'kl_weight', 1e-5)
        self.hf_backbone = hf_backbone
        
        logger.info(f"Initializing BayesianExtension with method: {self.uncertainty_method}")
        
        if self.uncertainty_method == 'bayesian':
            self._convert_to_bayesian(getattr(configs, 'bayesian_layers', ['output_projection']))
        elif self.uncertainty_method == 'dropout':
            self._setup_mc_dropout()
            
    def _convert_to_bayesian(self, layer_names):
        logger.info(f"Converting HF layers to Bayesian: {layer_names}")
        self.bayesian_layers_list = convert_to_bayesian(self.hf_backbone, layer_names)
        logger.info(f"Successfully converted {len(self.bayesian_layers_list)} HF layers to Bayesian")
        
    def _setup_mc_dropout(self):
        logger.info("Setting up Monte Carlo Dropout layers for HF model")
        self.mc_dropout1 = DropoutSampling(p=0.1)
        self.mc_dropout3 = DropoutSampling(p=0.1)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.uncertainty_method == 'bayesian':
            return self._bayesian_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.uncertainty_method == 'dropout':
            return self._dropout_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            return self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            
    def _bayesian_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        predictions = [self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec) for _ in range(self.n_samples)]
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack)
        
    def _dropout_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
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
            confidence_intervals[f'{int(conf_level * 100)}%'] = {'lower': lower_bound, 'upper': upper_bound, 'width': upper_bound - lower_bound}
            
        return {'prediction': mean_pred, 'uncertainty': total_std, 'variance': total_variance, 'confidence_intervals': confidence_intervals, 'predictions_samples': pred_stack}
        
    def get_kl_loss(self, max_kl_value=1e6):
        if self.uncertainty_method != 'bayesian':
            return torch.tensor(0.0, device=next(self.parameters()).device)
        kl_div = collect_kl_divergence(self.hf_backbone)
        kl_div_clipped = torch.clamp(kl_div, max=max_kl_value)
        if kl_div > kl_div_clipped:
            logger.warning(f"Clipped KL divergence from {kl_div:.4f} to {kl_div_clipped:.4f}")
        return kl_div_clipped * self.kl_weight

class QuantileExtension(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.quantiles = getattr(configs, 'quantile_levels', [0.1, 0.5, 0.9])
        self.num_quantiles = len(self.quantiles)
        self.original_c_out = getattr(configs, 'c_out_evaluation', configs.c_out)
        
        output_config = OutputConfig(
            output_type='quantile',
            d_model=self.original_c_out,
            output_dim=self.original_c_out,
            custom_params={'num_quantiles': self.num_quantiles}
        )
        self.quantile_expansion = create_output(output_config)
        
    def forward(self, base_output, **kwargs):
        if isinstance(base_output, dict) and 'prediction' in base_output:
            base_output['prediction'] = self.quantile_expansion(base_output['prediction'])
            if 'predictions_samples' in base_output and base_output['predictions_samples'] is not None:
                base_output['predictions_samples'] = self.quantile_expansion(base_output['predictions_samples'])
            return base_output
        else:
            return self.quantile_expansion(base_output)

def create_bayesian_extension(configs, hf_backbone):
    return BayesianExtension(configs, hf_backbone)

def create_quantile_extension(configs):
    return QuantileExtension(configs)