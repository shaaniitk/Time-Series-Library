"""
Hugging Face-based Autoformer Suite (Modularized)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from argparse import Namespace
import logging

from models.base_forecaster import BaseTimeSeriesForecaster, HFFrameworkMixin
from utils.modular_components.factories import create_backbone, create_embedding, create_output
from utils.modular_components.config_schemas import BackboneConfig, EmbeddingConfig, OutputConfig

logger = logging.getLogger(__name__)

class UncertaintyResult:
    def __init__(self, prediction, uncertainty=None, confidence_intervals=None, quantiles=None):
        self.prediction = prediction
        self.uncertainty = uncertainty
        self.confidence_intervals = confidence_intervals or {}
        self.quantiles = quantiles or {}

class HuggingFaceEnhancedAutoformer(BaseTimeSeriesForecaster, HFFrameworkMixin):
    def __init__(self, configs):
        super().__init__(configs)
        self.framework_type = 'hf'
        self.model_type = 'hf_enhanced_autoformer'
        
        backbone_config = BackboneConfig(backbone_type='robust_hf', model_name="amazon/chronos-t5-tiny", d_model=getattr(configs, 'd_model', 64), dropout=configs.dropout)
        self.backbone = create_backbone(backbone_config)
        self.backbone_name = "amazon/chronos-t5-tiny"
        
        self.input_projection = nn.Linear(self.enc_in, self.backbone.get_output_dim())
        self.projection = nn.Linear(self.backbone.get_output_dim(), self.c_out)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        projected_input = self.input_projection(x_enc)
        projected_input = self.dropout(projected_input)
        
        decoder_input = torch.zeros(x_enc.shape[0], self.configs.pred_len, self.backbone.get_output_dim()).to(x_enc.device)
        
        outputs = self.backbone(inputs_embeds=projected_input, decoder_inputs_embeds=decoder_input)
        hidden_state = outputs.last_hidden_state
        
        return self.projection(hidden_state)

class HuggingFaceBayesianAutoformer(HuggingFaceEnhancedAutoformer):
    def __init__(self, configs, uncertainty_method='bayesian', n_samples=50, quantile_levels=None, use_quantiles=None):
        super().__init__(configs)
        self.uncertainty_method = uncertainty_method
        self.n_samples = n_samples
        self.is_quantile_mode = bool(quantile_levels) or bool(use_quantiles)
        self.quantiles = sorted(quantile_levels) if quantile_levels else ([0.1, 0.25, 0.5, 0.75, 0.9] if use_quantiles else [])
        
        if self.is_quantile_mode:
            self.quantile_heads = nn.ModuleList([nn.Linear(self.backbone.get_output_dim(), configs.c_out) for _ in self.quantiles])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_uncertainty=False, detailed_uncertainty=False):
        if not return_uncertainty:
            return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        predictions = [super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec).detach() for _ in range(self.n_samples)]
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack, detailed_uncertainty)

    def _compute_uncertainty_statistics(self, pred_stack, detailed):
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        result = {'prediction': mean_pred, 'uncertainty': total_std, 'variance': total_variance}
        if detailed:
            result['predictions_samples'] = pred_stack
        
        if self.is_quantile_mode:
            result['quantiles'] = {f'q{int(q*100)}': torch.quantile(pred_stack, q, dim=0) for q in self.quantiles}
            
        return UncertaintyResult(**result)

class HuggingFaceHierarchicalAutoformer(HuggingFaceEnhancedAutoformer):
    def __init__(self, configs, hierarchy_levels=3):
        super().__init__(configs)
        self.hierarchy_levels = hierarchy_levels
        self.resolution_models = nn.ModuleList([HuggingFaceEnhancedAutoformer(configs) for _ in range(hierarchy_levels)])
        self.downsample_layers = nn.ModuleList([nn.AvgPool1d(kernel_size=2**i, stride=2**i) if i > 0 else nn.Identity() for i in range(hierarchy_levels)])
        self.fusion_layer = nn.Linear(configs.c_out * hierarchy_levels, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        resolution_outputs = []
        for level, model in enumerate(self.resolution_models):
            x_down = self.downsample_layers[level](x_enc.transpose(1, 2)).transpose(1, 2)
            mark_down = self.downsample_layers[level](x_mark_enc.transpose(1, 2)).transpose(1, 2) if x_mark_enc is not None else None
            output = model(x_down, mark_down, x_dec, x_mark_dec)
            output = F.interpolate(output.transpose(1, 2), size=self.configs.pred_len, mode='linear', align_corners=False).transpose(1, 2)
            resolution_outputs.append(output)
            
        combined = torch.cat(resolution_outputs, dim=-1)
        return self.fusion_layer(combined)

class HuggingFaceQuantileAutoformer(HuggingFaceBayesianAutoformer):
    def __init__(self, configs, quantiles=[0.1, 0.5, 0.9], kl_weight=0.3):
        super().__init__(configs, quantile_levels=quantiles)
        self.kl_weight = kl_weight

def create_hf_autoformer(model_type: str, configs, **kwargs):
    if model_type.lower() == "bayesian":
        return HuggingFaceBayesianAutoformer(configs, **kwargs)
    elif model_type.lower() == "hierarchical":
        return HuggingFaceHierarchicalAutoformer(configs, **kwargs)
    elif model_type.lower() == "quantile":
        return HuggingFaceQuantileAutoformer(configs, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

Model = HuggingFaceEnhancedAutoformer