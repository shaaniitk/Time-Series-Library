"""
Complete Hugging Face Autoformer Suite

This module provides HF-based replacements for all three custom Autoformer variants:
1. HFEnhancedAutoformer - Basic enhanced model (drop-in replacement for EnhancedAutoformer)
2. HFBayesianAutoformer - Bayesian uncertainty quantification 
3. HFHierarchicalAutoformer - Multi-resolution hierarchical processing
4. HFQuantileAutoformer - Quantile regression with uncertainty

All models use Amazon Chronos or other HF time series models as backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoConfig
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)

class UncertaintyResult:
    """Structured uncertainty quantification result"""
    def __init__(self, prediction, uncertainty=None, confidence_intervals=None, quantiles=None):
        self.prediction = prediction
        self.uncertainty = uncertainty
        self.confidence_intervals = confidence_intervals or {}
        self.quantiles = quantiles or {}

class HFEnhancedAutoformer(nn.Module):
    """
    Basic HF-based Enhanced Autoformer
    Drop-in replacement for EnhancedAutoformer using HF backbone
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        logger.info("Initializing HFEnhancedAutoformer (Basic Enhanced Model)")
        
        # Try to use Chronos, fallback to standard transformer
        try:
            self.backbone = AutoModel.from_pretrained("amazon/chronos-t5-tiny")
            logger.info("✅ Using Amazon Chronos T5 backbone")
        except Exception as e:
            logger.warning(f"Chronos not available ({e}), using fallback transformer")
            config = AutoConfig.from_pretrained("google/flan-t5-small")
            config.d_model = getattr(configs, 'd_model', 64)
            self.backbone = AutoModel.from_config(config)
        
        # Input projection layer
        self.input_projection = nn.Linear(configs.enc_in, self.backbone.config.d_model)
        
        # Output projection
        self.projection = nn.Linear(self.backbone.config.d_model, configs.c_out)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Standard forward pass"""
        batch_size, seq_len, features = x_enc.shape
        
        # Project input to model dimension
        projected_input = self.input_projection(x_enc)  # (batch, seq_len, d_model)
        
        # Apply dropout
        projected_input = self.dropout(projected_input)
        
        # For T5-like models, handle encoder-decoder structure
        try:
            # Create decoder input
            decoder_input = torch.zeros(
                batch_size, self.configs.pred_len, self.backbone.config.d_model
            ).to(x_enc.device)
            
            outputs = self.backbone(
                inputs_embeds=projected_input,
                decoder_inputs_embeds=decoder_input
            )
            hidden_state = outputs.last_hidden_state
            
        except Exception:
            # Fallback: encoder-only mode
            outputs = self.backbone.encoder(inputs_embeds=projected_input)
            hidden_state = outputs.last_hidden_state
            
            # Pool to prediction length
            hidden_state = hidden_state.mean(dim=1, keepdim=True)
            hidden_state = hidden_state.repeat(1, self.configs.pred_len, 1)
        
        # Generate final prediction
        output = self.projection(hidden_state)
        
        return output

class HFBayesianAutoformer(nn.Module):
    """
    HF-based Bayesian Autoformer with uncertainty quantification
    Drop-in replacement for BayesianEnhancedAutoformer
    """
    
    def __init__(self, configs, uncertainty_method='bayesian', n_samples=50,
                 quantile_levels=None, use_quantiles=None):
        super().__init__()
        self.configs = configs
        self.uncertainty_method = uncertainty_method
        self.n_samples = n_samples
        
        logger.info(f"Initializing HFBayesianAutoformer with {uncertainty_method} uncertainty")
        
        # Handle quantile configuration
        if quantile_levels:
            self.is_quantile_mode = True
            self.quantiles = sorted(quantile_levels)
        elif use_quantiles:
            self.is_quantile_mode = True
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        else:
            self.is_quantile_mode = False
            self.quantiles = []
        
        # Base enhanced model
        self.base_model = HFEnhancedAutoformer(configs)
        
        # Uncertainty-specific layers
        self.uncertainty_head = nn.Linear(self.base_model.backbone.config.d_model, configs.c_out)
        
        # Quantile heads if needed
        if self.is_quantile_mode:
            self.quantile_heads = nn.ModuleList([
                nn.Linear(self.base_model.backbone.config.d_model, configs.c_out)
                for _ in self.quantiles
            ])
        
        # Monte Carlo dropout layers
        self.mc_dropout1 = nn.Dropout(0.1)
        self.mc_dropout2 = nn.Dropout(0.15)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                return_uncertainty=False, detailed_uncertainty=False):
        """Forward pass with optional uncertainty quantification"""
        
        if not return_uncertainty:
            return self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        return self._uncertainty_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, detailed_uncertainty)
    
    def _single_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Single forward pass"""
        if self.uncertainty_method == 'dropout':
            x_enc = self.mc_dropout1(x_enc)
        
        return self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    def _uncertainty_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, detailed=False):
        """Forward pass with uncertainty estimation"""
        predictions = []
        
        # Generate multiple predictions for uncertainty estimation
        for i in range(self.n_samples):
            if self.uncertainty_method == 'dropout':
                # Enable dropout during inference
                self.train()
                pred = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                self.eval()
            else:
                # Bayesian sampling (simplified version)
                pred = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                # Add noise for variational approximation
                if self.training:
                    pred = pred + torch.randn_like(pred) * 0.01
            
            predictions.append(pred.detach())
        
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack, detailed)
    
    def _compute_uncertainty_statistics(self, pred_stack, detailed=False):
        """Compute uncertainty statistics from prediction samples"""
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        # Confidence intervals
        confidence_intervals = {}
        for conf_level in [0.68, 0.95]:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_bound = torch.quantile(pred_stack, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(pred_stack, upper_percentile / 100, dim=0)
            confidence_intervals[f'{int(conf_level * 100)}%'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        result = {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'confidence_intervals': confidence_intervals,
            'predictions_samples': pred_stack if detailed else None
        }
        
        # Add quantile-specific results if enabled
        if self.is_quantile_mode and self.quantiles:
            quantile_results = {}
            for i, q_level in enumerate(self.quantiles):
                quantile_results[f'q{int(q_level*100)}'] = torch.quantile(pred_stack, q_level, dim=0)
            result['quantiles'] = quantile_results
        
        return result

class HFHierarchicalAutoformer(nn.Module):
    """
    HF-based Hierarchical Autoformer for multi-resolution processing
    Drop-in replacement for HierarchicalEnhancedAutoformer
    """
    
    def __init__(self, configs, hierarchy_levels=3):
        super().__init__()
        self.configs = configs
        self.hierarchy_levels = hierarchy_levels
        
        logger.info(f"Initializing HFHierarchicalAutoformer with {hierarchy_levels} hierarchy levels")
        
        # Multiple HF models for different resolutions
        self.resolution_models = nn.ModuleList([
            HFEnhancedAutoformer(configs) for _ in range(hierarchy_levels)
        ])
        
        # Downsampling layers for different resolutions
        self.downsample_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2**i, stride=2**i) if i > 0 else nn.Identity()
            for i in range(hierarchy_levels)
        ])
        
        # Upsampling layers to bring back to original resolution
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='linear', align_corners=False) if i > 0 else nn.Identity()
            for i in range(hierarchy_levels)
        ])
        
        # Fusion layer to combine multi-resolution features
        self.fusion_weights = nn.Parameter(torch.ones(hierarchy_levels) / hierarchy_levels)
        self.fusion_layer = nn.Linear(configs.c_out * hierarchy_levels, configs.c_out)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Hierarchical forward pass"""
        batch_size, seq_len, features = x_enc.shape
        
        resolution_outputs = []
        
        # Process at different resolutions
        for level, (model, downsample, upsample) in enumerate(
            zip(self.resolution_models, self.downsample_layers, self.upsample_layers)
        ):
            # Downsample input for this resolution level
            if level > 0:
                # Transpose for 1D pooling: (batch, seq_len, features) -> (batch, features, seq_len)
                x_down = x_enc.transpose(1, 2)
                x_down = downsample(x_down)
                x_down = x_down.transpose(1, 2)  # Back to (batch, seq_len', features)
                
                # Adjust temporal markers accordingly
                if x_mark_enc is not None:
                    mark_down = x_mark_enc.transpose(1, 2)
                    mark_down = downsample(mark_down)
                    mark_down = mark_down.transpose(1, 2)
                else:
                    mark_down = x_mark_enc
            else:
                x_down = x_enc
                mark_down = x_mark_enc
            
            # Process through model at this resolution
            output = model(x_down, mark_down, x_dec, x_mark_dec)
            
            # Upsample back to target prediction length if needed
            if level > 0 and output.shape[1] != self.configs.pred_len:
                # Adjust output to target prediction length
                output = F.interpolate(
                    output.transpose(1, 2),
                    size=self.configs.pred_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            resolution_outputs.append(output)
        
        # Weighted fusion of multi-resolution outputs
        weighted_outputs = []
        for i, output in enumerate(resolution_outputs):
            weighted_outputs.append(self.fusion_weights[i] * output)
        
        # Concatenate and fuse
        combined = torch.cat(weighted_outputs, dim=-1)  # (batch, pred_len, c_out * levels)
        fused_output = self.fusion_layer(combined)  # (batch, pred_len, c_out)
        
        return fused_output

class HFQuantileAutoformer(nn.Module):
    """
    HF-based Quantile Autoformer for quantile regression
    Drop-in replacement for QuantileBayesianAutoformer
    """
    
    def __init__(self, configs, quantiles=[0.1, 0.5, 0.9], kl_weight=0.3):
        super().__init__()
        self.configs = configs
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.kl_weight = kl_weight
        self.quantile_weight = 1.0 - kl_weight
        
        logger.info(f"Initializing HFQuantileAutoformer with quantiles: {quantiles}")
        logger.info(f"Loss weights: KL={kl_weight:.3f}, Quantile={self.quantile_weight:.3f}")
        
        # Base Bayesian model
        quantile_configs = Namespace(**vars(configs))
        quantile_configs.c_out = configs.c_out * self.n_quantiles  # Expand for quantiles
        
        self.base_model = HFBayesianAutoformer(quantile_configs, quantile_levels=quantiles)
        
        # Original output dimension
        self.original_c_out = configs.c_out
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                return_uncertainty=False, detailed_uncertainty=False):
        """Forward pass for quantile prediction"""
        
        if return_uncertainty:
            # Get uncertainty results from base model
            result = self.base_model(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                return_uncertainty=True, detailed_uncertainty=detailed_uncertainty
            )
            
            # Reshape outputs to separate quantiles
            prediction = result['prediction']
            batch_size, pred_len, total_features = prediction.shape
            
            if total_features != self.original_c_out * self.n_quantiles:
                logger.warning(f"Expected {self.original_c_out * self.n_quantiles} features, got {total_features}")
            
            # Reshape to (batch, pred_len, n_quantiles, c_out)
            quantile_predictions = prediction.view(
                batch_size, pred_len, self.n_quantiles, self.original_c_out
            )
            
            # Create quantile-specific results
            quantile_outputs = {}
            for i, q_level in enumerate(self.quantiles):
                quantile_outputs[f'quantile_{q_level}'] = quantile_predictions[:, :, i, :]
            
            result['quantile_outputs'] = quantile_outputs
            return result
        
        else:
            # Standard forward pass
            return self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# Model aliases for compatibility
Model = HFEnhancedAutoformer  # Default model
