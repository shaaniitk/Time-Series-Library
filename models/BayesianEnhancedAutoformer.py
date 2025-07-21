"""
Bayesian Enhanced Autoformer for Uncertainty Quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from argparse import Namespace

from .EnhancedAutoformer import EnhancedAutoformer
from layers.BayesianLayers import convert_to_bayesian, collect_kl_divergence, DropoutSampling
from utils.logger import logger
from utils.losses import PinballLoss

class BayesianEnhancedAutoformer(nn.Module):
    """
    Bayesian version of Enhanced Autoformer with uncertainty quantification.
    """
    
    def __init__(self, configs, uncertainty_method='bayesian', n_samples=50, 
                 bayesian_layers=['projection'], kl_weight=1e-5, 
                 use_quantiles=None,
                 quantile_levels=None):
        super(BayesianEnhancedAutoformer, self).__init__()
        logger.info(f"Initializing BayesianEnhancedAutoformer with method={uncertainty_method}")
        
        self.configs_original_ref = configs
        self.uncertainty_method = uncertainty_method
        self.n_samples = n_samples
        self.kl_weight_param = kl_weight
        
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out

        passed_q_levels = quantile_levels
        if not passed_q_levels and hasattr(configs, 'quantile_levels'):
            passed_q_levels = configs.quantile_levels

        if passed_q_levels and isinstance(passed_q_levels, list) and len(passed_q_levels) > 0:
            self.is_quantile_mode = True
            self.quantiles = sorted(passed_q_levels)
            self.num_quantiles = len(self.quantiles)
            logger.info(f"BayesianEnhancedAutoformer: Quantile mode ON. Quantiles: {self.quantiles}")
        elif use_quantiles:
            self.is_quantile_mode = True
            self.quantiles = getattr(configs, 'quantiles_default_list', [0.1, 0.25, 0.5, 0.75, 0.9])
            self.num_quantiles = len(self.quantiles)
            logger.info(f"BayesianEnhancedAutoformer: Quantile mode ON (due to use_quantiles=True). Default quantiles: {self.quantiles}")
        else:
            self.is_quantile_mode = False
            self.quantiles = []
            self.num_quantiles = 1
            logger.info(f"BayesianEnhancedAutoformer: Quantile mode OFF.")
        
        self.original_c_out = getattr(configs, 'c_out_evaluation', configs.c_out)
        
        base_model_configs_dict = vars(configs).copy()
        if 'quantile_levels' in base_model_configs_dict:
            del base_model_configs_dict['quantile_levels']
        base_model_configs_dict['c_out'] = self.original_c_out
       
        base_model_configs_ns = Namespace(**base_model_configs_dict)
        
        self.base_model = EnhancedAutoformer(base_model_configs_ns, quantile_levels=None)
        
        self.quantile_expansion = None
        if self.is_quantile_mode and self.num_quantiles > 1:
            self.quantile_expansion = nn.Linear(self.original_c_out, self.c_out)
            logger.info(f"BayesianEnhancedAutoformer: Added quantile expansion layer {self.original_c_out} -> {self.c_out}")
        
        if uncertainty_method == 'bayesian':
            logger.info(f"Converting layers to Bayesian: {bayesian_layers}")
            self.bayesian_layers_list = convert_to_bayesian(self.base_model, bayesian_layers)
            logger.info(f"Converted {len(self.bayesian_layers_list)} layers to Bayesian.")
        elif uncertainty_method == 'dropout':
            self._setup_dropout_layers()

    def _setup_dropout_layers(self):
        logger.info("Setting up Monte Carlo Dropout layers")
        self.mc_dropout1 = DropoutSampling(p=0.1)
        self.mc_dropout2 = DropoutSampling(p=0.15)
        self.mc_dropout3 = DropoutSampling(p=0.1)

    def _single_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.uncertainty_method == 'dropout':
            x_enc = self.mc_dropout1(x_enc)
            
        output = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.uncertainty_method == 'dropout':
            output = self.mc_dropout3(output)
        
        if self.quantile_expansion is not None:
            output = self.quantile_expansion(output)
            
        return output
    
    def _bayesian_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, detailed=False):
        logger.debug(f"Computing Bayesian uncertainty with {self.n_samples} samples")
        predictions = []
        grad_context = torch.enable_grad if self.training else torch.no_grad

        for i in range(self.n_samples):
            with torch.enable_grad() if i == 0 else grad_context():
                pred = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred.detach() if i > 0 and not self.training else pred)
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack, detailed)
    
    def _dropout_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, detailed=False):
        logger.debug(f"Computing MC Dropout uncertainty with {self.n_samples} samples")
        predictions = []
        self.train()
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)
        self.eval()
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack, detailed)
    
    def _compute_uncertainty_statistics(self, pred_stack, detailed=False):
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        confidence_intervals = self._compute_confidence_intervals(pred_stack)
        result = {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'confidence_intervals': confidence_intervals,
            'predictions_samples': pred_stack if detailed else None
        }
        if self.is_quantile_mode and self.quantiles:
            result.update(self._compute_quantile_uncertainty(pred_stack, self.quantiles))
        return result
    
    def _compute_confidence_intervals(self, pred_stack, confidence_levels=[0.68, 0.95, 0.99]):
        intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_bound = torch.quantile(pred_stack, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(pred_stack, upper_percentile / 100, dim=0)
            intervals[f'{int(conf_level * 100)}%'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        return intervals
    
    def _compute_quantile_uncertainty(self, pred_stack, current_quantiles_list):
        if not current_quantiles_list:
            return {}
        n_quantiles = len(self.quantiles)
        total_features = pred_stack.shape[3]
        if total_features % n_quantiles != 0:
            raise ValueError(f"Total features ({total_features}) must be divisible by the number of quantiles ({n_quantiles})")
        features_per_quantile = total_features // n_quantiles
        quantile_results = {}
        for i, q_level in enumerate(current_quantiles_list):
            start_idx = i * features_per_quantile
            end_idx = (i + 1) * features_per_quantile
            q_predictions = pred_stack[:, :, :, start_idx:end_idx]
            q_mean = torch.mean(q_predictions, dim=0)
            q_std = torch.std(q_predictions, dim=0)
            q_intervals = {}
            for conf in [0.68, 0.95]:
                alpha = 1 - conf
                q_lower = torch.quantile(q_predictions, alpha/2, dim=0)
                q_upper = torch.quantile(q_predictions, 1-alpha/2, dim=0)
                q_intervals[f'{int(conf*100)}%'] = {'lower': q_lower, 'upper': q_upper, 'width': q_upper - q_lower}
            quantile_results[f'quantile_{q_level}'] = {
                'prediction': q_mean, 'uncertainty': q_std,
                'confidence_intervals': q_intervals,
                'certainty_score': self._compute_quantile_certainty(q_predictions, q_level)
            }
        return {'quantile_specific': quantile_results}
    
    def _compute_quantile_certainty(self, q_predictions, q_level):
        mean_pred = torch.mean(q_predictions, dim=0)
        std_pred = torch.std(q_predictions, dim=0)
        cv = std_pred / (torch.abs(mean_pred) + 1e-8)
        return 1.0 / (1.0 + cv)
    
    def get_kl_loss(self, max_kl_value=1e6):
        if self.uncertainty_method != 'bayesian':
            return 0.0
        kl_div = collect_kl_divergence(self)
        kl_div_clipped = torch.clamp(kl_div, max=max_kl_value)
        if kl_div > kl_div_clipped:
            logger.warning(f"Clipped KL divergence from {kl_div:.4f} to {kl_div_clipped:.4f}")
        return kl_div_clipped * self.kl_weight_param

    def compute_loss(self, predictions, targets, base_criterion, return_components=False):
        data_loss = base_criterion(predictions, targets)
        kl_loss_weighted = self.get_kl_loss()
        total_loss = data_loss + kl_loss_weighted
        if return_components:
            return {'data_loss': data_loss, 'kl_contribution': kl_loss_weighted, 'total_loss': total_loss}
        return total_loss

    def configure_optimizer_loss(self, base_criterion, verbose=False):
        self._verbose_loss = verbose
        self._loss_history = []
        
        def enhanced_bayesian_loss(predictions, targets):
            loss_components = self.compute_loss(predictions, targets, base_criterion, return_components=True)
            if self._verbose_loss:
                loss_history_entry = {k: v.item() for k, v in loss_components.items()}
                self._loss_history.append(loss_history_entry)
                if len(self._loss_history) % 10 == 0:
                    recent = self._loss_history[-1]
                    logger.info(f"Bayesian Loss - Data: {recent['data_loss']:.6f}, KL: {recent['kl_contribution']:.6f}, Total: {recent['total_loss']:.6f}")
            return loss_components['total_loss']
        
        if isinstance(base_criterion, PinballLoss):
            enhanced_bayesian_loss._is_pinball_loss_based = True
            logger.info("BayesianEnhancedAutoformer: Wrapped loss is PinballLoss based.")
            
        return enhanced_bayesian_loss

Model = BayesianEnhancedAutoformer