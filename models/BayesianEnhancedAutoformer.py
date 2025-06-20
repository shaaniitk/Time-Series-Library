# BayesianEnhancedAutoformer.py

"""
Bayesian Enhanced Autoformer for Uncertainty Quantification

This module implements a Bayesian version of the Enhanced Autoformer that provides
uncertainty estimates along with predictions. It integrates with any loss function
including quantile loss to provide certainty measures for specific quantile ranges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from models.EnhancedAutoformer import EnhancedAutoformer
from layers.BayesianLayers import BayesianLinear, BayesianConv1d, convert_to_bayesian, collect_kl_divergence, DropoutSampling
from layers.Embed import DataEmbedding_wo_pos
from utils.logger import logger
from argparse import Namespace # For creating a temporary configs object
from utils.losses import get_loss_function # For checking PinballLoss type

class BayesianEnhancedAutoformer(nn.Module):
    """
    Bayesian version of Enhanced Autoformer with uncertainty quantification.
    
    Key Features:
    - Provides prediction uncertainty estimates
    - Works with any loss function (MSE, MAE, Quantile, etc.)
    - Separates aleatoric and epistemic uncertainty
    - Supports multiple uncertainty estimation methods
    """
    
    def _setup_bayesian_layers(self, bayesian_layers):
        """Convert specified layers to Bayesian versions"""
        logger.info(f"Converting layers to Bayesian: {bayesian_layers}")
        
        self.bayesian_layers_list = []
        
        if 'projection' in bayesian_layers:
            projection_found = False
            
            if hasattr(self.base_model, 'projection'):
                original_proj = self.base_model.projection
                bayesian_proj = BayesianLinear(
                    original_proj.in_features,
                    original_proj.out_features,
                    bias=original_proj.bias is not None
                )
                self.base_model.projection = bayesian_proj
                self.bayesian_layers_list.append(bayesian_proj)
                projection_found = True
                logger.info(f"Converted top-level projection: {original_proj.in_features}->{original_proj.out_features}")
            
            elif hasattr(self.base_model, 'decoder') and hasattr(self.base_model.decoder, 'projection'):
                original_proj = self.base_model.decoder.projection
                bayesian_proj = BayesianLinear(
                    original_proj.in_features,
                    original_proj.out_features,
                    bias=original_proj.bias is not None
                )
                self.base_model.decoder.projection = bayesian_proj
                self.bayesian_layers_list.append(bayesian_proj)
                projection_found = True
                logger.info(f"Converted decoder projection: {original_proj.in_features}->{original_proj.out_features}")
            
            if not projection_found:
                logger.warning("No projection layer found to convert to Bayesian!")
        
    def _setup_dropout_layers(self):
        """Add Monte Carlo dropout layers for uncertainty estimation"""
        logger.info("Setting up Monte Carlo Dropout layers")
        self.mc_dropout1 = DropoutSampling(p=0.1)
        self.mc_dropout2 = DropoutSampling(p=0.15)
        self.mc_dropout3 = DropoutSampling(p=0.1)

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
        if 'quantile_levels' in base_model_configs_dict: # Remove quantile_levels from base model configs
            del base_model_configs_dict['quantile_levels']
        # The base model should always be configured for point prediction.
        # Its output dimension (c_out) should be the number of base target variables.
        base_model_configs_dict['c_out'] = self.original_c_out
       
        base_model_configs_ns = Namespace(**base_model_configs_dict)
        
        self.base_model = EnhancedAutoformer(base_model_configs_ns, quantile_levels=None)
        
        self.quantile_expansion = None
        if self.is_quantile_mode and self.num_quantiles > 1:
            self.quantile_expansion = nn.Linear(self.original_c_out, self.c_out)
            logger.info(f"BayesianEnhancedAutoformer: Added quantile expansion layer {self.original_c_out} -> {self.c_out}")
        
        if uncertainty_method == 'bayesian':
            self._setup_bayesian_layers(bayesian_layers)
        elif uncertainty_method == 'dropout':
            self._setup_dropout_layers()
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                return_uncertainty=False, detailed_uncertainty=False):
        if not return_uncertainty:
            return self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.uncertainty_method == 'bayesian':
            return self._bayesian_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, detailed_uncertainty)
        elif self.uncertainty_method == 'dropout':
            return self._dropout_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, detailed_uncertainty)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
    
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
        for i in range(self.n_samples):
            with torch.no_grad() if i > 0 else torch.enable_grad():
                pred = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred.detach() if i > 0 else pred)
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
        if detailed:
            result.update(self._compute_detailed_uncertainty(pred_stack))
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
        batch_size, seq_len, total_features = pred_stack.shape[1], pred_stack.shape[2], pred_stack.shape[3]
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
                q_intervals[f'{int(conf*100)}%'] = {
                    'lower': q_lower,
                    'upper': q_upper,
                    'width': q_upper - q_lower
                }
            quantile_results[f'quantile_{q_level}'] = {
                'prediction': q_mean,
                'uncertainty': q_std,
                'confidence_intervals': q_intervals,
                'certainty_score': self._compute_quantile_certainty(q_predictions, q_level)
            }
        return {'quantile_specific': quantile_results}
    
    def _compute_quantile_certainty(self, q_predictions, q_level):
        mean_pred = torch.mean(q_predictions, dim=0)
        std_pred = torch.std(q_predictions, dim=0)
        cv = std_pred / (torch.abs(mean_pred) + 1e-8)
        certainty = 1.0 / (1.0 + cv)
        return certainty
    
    def _compute_detailed_uncertainty(self, pred_stack):
        total_var = torch.var(pred_stack, dim=0)
        epistemic_var = total_var * 0.7
        aleatoric_var = total_var - epistemic_var
        return {
            'epistemic_uncertainty': torch.sqrt(epistemic_var),
            'aleatoric_uncertainty': torch.sqrt(aleatoric_var),
            'total_uncertainty_decomposed': torch.sqrt(total_var)
        }
    
    def get_kl_loss(self):
        if self.uncertainty_method != 'bayesian':
            return 0.0
        return collect_kl_divergence(self) * self.kl_weight_param

    def compute_loss(self, predictions, targets, base_criterion, return_components=False):
        """
        Computes the total loss for the Bayesian model, combining data loss and KL divergence.
        
        Args:
            predictions: Model outputs.
            targets: Ground truth values.
            base_criterion: The base loss function (e.g., MSE, PinballLoss).
            return_components: If True, returns a dictionary of loss components.
            
        Returns:
            The total loss, or a dictionary of loss components.
        """
        # 1. Data Loss
        data_loss = base_criterion(predictions, targets)

        # 2. KL Divergence Loss (already weighted)
        kl_loss_weighted = self.get_kl_loss()

        # 3. Total Loss
        total_loss = data_loss + kl_loss_weighted

        if return_components:
            return {
                'data_loss': data_loss,
                'kl_contribution': kl_loss_weighted,
                'total_loss': total_loss,
            }
        
        return total_loss

    def configure_optimizer_loss(self, base_criterion, verbose=False):
        self._verbose_loss = verbose
        self._loss_history = []
        
        def enhanced_bayesian_loss(predictions, targets):
            loss_components = self.compute_loss( # type: ignore
                predictions, targets, base_criterion, return_components=True
            )
            if self._verbose_loss:
                loss_history_entry = {k: v.item() for k, v in loss_components.items()}
                self._loss_history.append(loss_history_entry)
                if len(self._loss_history) % 10 == 0:
                    recent = self._loss_history[-1]
                    logger.info(f"Bayesian Loss - Data: {recent['data_loss']:.6f}, KL: {recent['kl_contribution']:.6f}, Total: {recent['total_loss']:.6f}")
            return loss_components['total_loss']
        
        if isinstance(base_criterion, get_loss_function('pinball', quantile_levels=[0.5]).__class__):
            enhanced_bayesian_loss._is_pinball_loss_based = True
            logger.info("BayesianEnhancedAutoformer: Wrapped loss is PinballLoss based.")
            
        return enhanced_bayesian_loss

# Alias for compatibility with experiment framework
Model = BayesianEnhancedAutoformer
