"""
HFBayesianAutoformer: Production-Ready Bayesian Uncertainty with Covariate Support (Modularized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, List, Union, NamedTuple
from argparse import Namespace

from .HFEnhancedAutoformer import HFEnhancedAutoformer
from layers.BayesianLayers import BayesianLinear, convert_to_bayesian, collect_kl_divergence
from utils.modular_components.factories import create_loss
from utils.modular_components.implementations.losses import LossConfig, BayesianLossConfig

logger = logging.getLogger(__name__)

class UncertaintyResult(NamedTuple):
    prediction: torch.Tensor
    uncertainty: torch.Tensor
    variance: torch.Tensor
    confidence_intervals: Dict[str, Dict[str, torch.Tensor]]
    quantiles: Dict[str, torch.Tensor]
    predictions_samples: Optional[torch.Tensor] = None
    quantile_specific: Optional[Dict] = None
    covariate_impact: Optional[Dict] = None
    epistemic_uncertainty: Optional[torch.Tensor] = None
    aleatoric_uncertainty: Optional[torch.Tensor] = None
    kl_divergence: Optional[torch.Tensor] = None
    loss_components: Optional[Dict] = None
    covariate_loss: Optional[torch.Tensor] = None
    target_loss: Optional[torch.Tensor] = None

class HFBayesianAutoformerProduction(nn.Module):
    def __init__(self, configs):
        super(HFBayesianAutoformerProduction, self).__init__()
        
        self.configs = configs
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 24)
        self.d_model = getattr(configs, 'd_model', 512)
        self.c_out = getattr(configs, 'c_out', 1)
        self.dropout_prob = getattr(configs, 'dropout', 0.1)
        
        self.mc_samples = getattr(configs, 'mc_samples', 10)
        self.uncertainty_method = getattr(configs, 'uncertainty_method', 'mc_dropout')
        
        self.use_bayesian_layers = getattr(configs, 'use_bayesian_layers', True)
        self.bayesian_kl_weight = getattr(configs, 'bayesian_kl_weight', 1e-5)
        self.uncertainty_decomposition = getattr(configs, 'uncertainty_decomposition', True)
        
        self.loss_type = getattr(configs, 'loss_type', 'adaptive')
        self.covariate_loss_mode = getattr(configs, 'covariate_loss_mode', 'combined')
        self.multi_component_loss = getattr(configs, 'multi_component_loss', True)
        
        self.is_quantile_mode = getattr(configs, 'quantile_mode', False)
        self.quantiles = getattr(configs, 'quantile_levels', [0.1, 0.25, 0.5, 0.75, 0.9])
        
        logger.info("Initializing ENHANCED Production HFBayesianAutoformer")
        
        self.base_model = HFEnhancedAutoformer(configs)
        
        self._init_covariate_components()
        self._init_bayesian_components()
        self._init_uncertainty_components()
        self._init_loss_ecosystem()
        self._init_monte_carlo_components()
        
        logger.info("✅ ENHANCED Production HFBayesianAutoformer initialized successfully")

    def _init_covariate_components(self):
        from layers.Embed import TemporalEmbedding, TimeFeatureEmbedding
        if self.configs.embed == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=self.base_model.d_model, embed_type=self.configs.embed, freq=self.configs.freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=self.base_model.d_model)

    def _init_bayesian_components(self):
        if not self.use_bayesian_layers:
            return
        base_d_model = self.base_model.d_model
        self.bayesian_projection = BayesianLinear(base_d_model, base_d_model)
        self.bayesian_uncertainty_head = BayesianLinear(base_d_model, self.c_out)
        if hasattr(self.base_model, 'projection'):
            self.base_model.projection = convert_to_bayesian(self.base_model.projection)
        self.bayesian_variance_head = BayesianLinear(base_d_model, self.c_out)
        self.variance_activation = nn.Softplus()

    def _init_loss_ecosystem(self):
        loss_config = BayesianLossConfig(
            loss_type=self.loss_type,
            kl_weight=self.bayesian_kl_weight,
            uncertainty_weight=0.1,
            quantiles=self.quantiles
        )
        self.primary_loss_fn = create_loss(loss_config)
        self.mse_loss = nn.MSELoss()

    def _init_uncertainty_components(self):
        base_d_model = self.base_model.d_model
        self.uncertainty_head = nn.Sequential(
            nn.Linear(base_d_model, base_d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(base_d_model // 2, self.c_out),
            nn.Softplus()
        )
        if self.is_quantile_mode:
            self.quantile_heads = nn.ModuleDict({
                f'q{int(q*100)}': nn.Linear(base_d_model, self.c_out) for q in self.quantiles
            })

    def _init_monte_carlo_components(self):
        if self.uncertainty_method == 'mc_dropout':
            self.mc_dropout1 = nn.Dropout(self.dropout_prob)
            self.mc_dropout3 = nn.Dropout(self.dropout_prob * 0.8)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_uncertainty=False, detailed_uncertainty=False, analyze_covariate_impact=False):
        if not return_uncertainty:
            return self._single_forward_with_covariates(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return self._uncertainty_forward_with_covariates(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, detailed_uncertainty, analyze_covariate_impact)

    def _single_forward_with_covariates(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.training and self.uncertainty_method == 'mc_dropout':
            x_enc = self.mc_dropout1(x_enc)
        
        if x_mark_enc is not None and x_mark_enc.size(-1) > 0:
            enc_temporal_embed = self.temporal_embedding(x_mark_enc)
            if enc_temporal_embed.size(1) == x_enc.size(1):
                x_enc = x_enc + enc_temporal_embed[:, :, :x_enc.size(-1)]
        
        base_output = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        
        if self.training and self.uncertainty_method == 'mc_dropout':
            base_output = self.mc_dropout3(base_output)
        
        return base_output

    def _uncertainty_forward_with_covariates(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, detailed, analyze_covariate_impact):
        predictions = []
        original_training = self.training
        if self.uncertainty_method == 'mc_dropout':
            self.train()
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self._single_forward_with_covariates(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
                predictions.append(pred.clone())
        
        self.train(original_training)
        
        pred_stack = torch.stack(predictions)
        return self._compute_enhanced_uncertainty_statistics(pred_stack, detailed)

    def _compute_enhanced_uncertainty_statistics(self, pred_stack, detailed):
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance + 1e-8)
        
        confidence_intervals = self._compute_confidence_intervals_robust(pred_stack)
        
        quantiles_dict = {}
        if self.is_quantile_mode:
            for q_level in self.quantiles:
                quantiles_dict[f'q{int(q_level*100)}'] = torch.quantile(pred_stack, q_level, dim=0)

        return UncertaintyResult(
            prediction=mean_pred,
            uncertainty=total_std,
            variance=total_variance,
            confidence_intervals=confidence_intervals,
            quantiles=quantiles_dict,
            predictions_samples=pred_stack if detailed else None
        )

    def _compute_confidence_intervals_robust(self, pred_stack, confidence_levels=[0.68, 0.95]):
        intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = alpha / 2
            upper_percentile = 1 - alpha / 2
            lower_bound = torch.quantile(pred_stack, lower_percentile, dim=0)
            upper_bound = torch.quantile(pred_stack, upper_percentile, dim=0)
            intervals[f'{int(conf_level * 100)}%'] = {'lower': lower_bound, 'upper': upper_bound, 'width': upper_bound - lower_bound}
        return intervals

    def collect_kl_divergence(self):
        if not self.use_bayesian_layers:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return collect_kl_divergence(self)

    def compute_loss(self, pred_result, true, **kwargs):
        if not self.multi_component_loss:
            pred = pred_result['prediction'] if isinstance(pred_result, dict) else pred_result
            return self.mse_loss(pred, true)
        
        return self.primary_loss_fn(self, pred_result, true, **kwargs)

Model = HFBayesianAutoformerProduction