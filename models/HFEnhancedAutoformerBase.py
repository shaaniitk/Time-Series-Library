"""
HF Enhanced Autoformer Base - Foundation for Advanced HF Models

This module provides the base class for HF-enhanced models that can handle
advanced features like Bayesian uncertainty, quantile regression, and 
hierarchical processing while leveraging existing loss infrastructure.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union
from argparse import Namespace

from models.HFEnhancedAutoformer import HFEnhancedAutoformer
from utils.logger import logger


class HFEnhancedAutoformerBase(nn.Module):
    """
    Base class for HF Enhanced models with advanced feature support.
    
    This class follows the defensive copying pattern from BayesianEnhancedAutoformer
    and integrates with existing loss infrastructure from losses.py and bayesian_losses.py
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Store original configs safely (defensive copy pattern)
        self.configs_original = configs
        self.original_c_out = getattr(configs, 'c_out_evaluation', configs.c_out)
        
        # Initialize HF backbone with defensive config copying
        hf_configs_dict = vars(configs).copy()
        
        # Clean up quantile-specific configs for base model (following BayesianEnhancedAutoformer pattern)
        if 'quantile_levels' in hf_configs_dict:
            del hf_configs_dict['quantile_levels']
        if 'uncertainty_method' in hf_configs_dict:
            del hf_configs_dict['uncertainty_method']
        if 'bayesian_layers' in hf_configs_dict:
            del hf_configs_dict['bayesian_layers']
            
        # Base model should only do point predictions
        hf_configs_dict['c_out'] = self.original_c_out
        
        logger.info(f"HFEnhancedAutoformerBase: Creating HF backbone with c_out={self.original_c_out}")
        self.hf_backbone = HFEnhancedAutoformer(Namespace(**hf_configs_dict))
        
        # Initialize loss manager using existing infrastructure
        self.loss_manager = self._create_loss_manager(configs)
        
        # Feature extension points (to be populated by subclasses)
        self.feature_processors = nn.ModuleDict()
        
        logger.info(f"HFEnhancedAutoformerBase initialized successfully")
        
    def _create_loss_manager(self, configs):
        """Create loss manager using existing loss infrastructure"""
        from utils.losses import get_loss_function
        from utils.bayesian_losses import create_bayesian_loss
        
        loss_type = getattr(configs, 'loss', 'mse').lower()
        
        # Check if this should use Bayesian loss wrapper
        if (hasattr(configs, 'uncertainty_method') and 
            getattr(configs, 'uncertainty_method', None) in ['bayesian', 'dropout']):
            
            logger.info(f"Creating Bayesian loss wrapper for loss_type: {loss_type}")
            
            # Use Bayesian loss wrapper from existing bayesian_losses.py
            return create_bayesian_loss(
                loss_type=loss_type,
                kl_weight=getattr(configs, 'kl_weight', 1e-5),
                uncertainty_weight=getattr(configs, 'uncertainty_weight', 0.1),
                quantiles=getattr(configs, 'quantile_levels', None)
            )
        else:
            logger.info(f"Creating standard loss function: {loss_type}")
            
            # Use standard loss from existing losses.py
            return get_loss_function(
                loss_type,
                **self._extract_loss_kwargs(configs, loss_type)
            )
    
    def _extract_loss_kwargs(self, configs, loss_type):
        """Extract appropriate kwargs for loss function from existing patterns"""
        kwargs = {}
        
        if loss_type == 'pinball':
            kwargs['quantile_levels'] = getattr(configs, 'quantile_levels', [0.1, 0.5, 0.9])
        elif loss_type == 'ps_loss':
            kwargs.update({
                'pred_len': configs.pred_len,
                'mse_weight': getattr(configs, 'ps_mse_weight', 0.5),
                'w_corr': getattr(configs, 'ps_w_corr', 1.0),
                'w_var': getattr(configs, 'ps_w_var', 1.0),
                'w_mean': getattr(configs, 'ps_w_mean', 1.0)
            })
        elif loss_type == 'huber':
            kwargs['delta'] = getattr(configs, 'huber_delta', 1.0)
        elif loss_type == 'quantile':
            kwargs['quantile'] = getattr(configs, 'quantile', 0.5)
        elif loss_type == 'seasonal':
            kwargs.update({
                'season_length': getattr(configs, 'season_length', 24),
                'seasonal_weight': getattr(configs, 'seasonal_weight', 1.0)
            })
        elif loss_type == 'trend_aware':
            kwargs.update({
                'trend_weight': getattr(configs, 'trend_weight', 1.0),
                'noise_weight': getattr(configs, 'noise_weight', 0.5)
            })
            
        return kwargs
        
    def add_feature_processor(self, name, processor):
        """Hook for adding feature processors (Bayesian, Hierarchical, etc.)"""
        self.feature_processors[name] = processor
        logger.info(f"Added feature processor: {name}")
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Base forward pass that can be extended by feature processors
        
        Subclasses should override this to add their specific processing
        """
        # Base HF forward
        base_output = self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        
        # Apply feature processors in order
        processed_output = base_output
        for name, processor in self.feature_processors.items():
            logger.debug(f"Applying feature processor: {name}")
            processed_output = processor(processed_output, x_enc, x_mark_enc, x_dec, x_mark_dec)
            
        return processed_output
        
    def compute_loss(self, predictions, targets, return_components=False):
        """Compute loss using existing loss infrastructure"""
        
        if hasattr(self.loss_manager, 'forward'):
            # This is a BayesianLoss from bayesian_losses.py
            logger.debug("Using Bayesian loss computation")
            return self.loss_manager.forward(
                model=self,
                pred_result=predictions,
                true=targets
            )
        else:
            # Standard loss from losses.py
            logger.debug("Using standard loss computation")
            if isinstance(predictions, dict):
                pred_tensor = predictions['prediction']
            else:
                pred_tensor = predictions
                
            return self.loss_manager(pred_tensor, targets)
            
    def get_model_info(self):
        """Get model information for debugging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': self.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone_type': 'HF_Enhanced',
            'feature_processors': list(self.feature_processors.keys()),
            'loss_manager_type': type(self.loss_manager).__name__
        }


# Factory function for creating base models
def create_hf_enhanced_base(configs):
    """Factory function to create HF Enhanced base model"""
    logger.info("Creating HF Enhanced base model")
    return HFEnhancedAutoformerBase(configs)


# Export for compatibility
Model = HFEnhancedAutoformerBase
