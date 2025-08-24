"""
Advanced HF Model Factory

This module provides factory functions for creating advanced HF models
with various combinations of features:

1. Standard HF models (existing)
2. Bayesian HF models (with uncertainty quantification)
3. Hierarchical HF models (with multi-scale processing)
4. Quantile HF models (with quantile regression)
5. Combined models (Bayesian + Hierarchical + Quantile)

Following the strategy of external extensions while keeping HF backbone stable.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from argparse import Namespace

# Import existing HF models
from models.HFEnhancedAutoformer import HFEnhancedAutoformer

# Import extension modules
from models.HFBayesianExtensions import (
    BayesianExtension, 
    QuantileExtension,
    create_bayesian_extension,
    create_quantile_extension
)
from models.HFHierarchicalExtensions import (
    WaveletProcessor,
    HierarchicalProcessor, 
    create_wavelet_processor,
    create_hierarchical_processor
)

# Import loss integration
from layers.modular.losses.registry import get_loss_component
from layers.modular.losses.adaptive_bayesian_losses import create_bayesian_loss
from utils.logger import logger


class HFAdvancedModelWrapper(nn.Module):
    """
    Wrapper that combines HF backbone with advanced extensions.
    
    This is the core implementation of our strategy:
    - Keep HF backbone unchanged and stable
    - Add advanced features through external extensions
    - Leverage existing loss infrastructure
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Store configuration
        self.configs = self._ensure_namespace(configs)
        
        # Create HF backbone (unchanged)
        self.hf_backbone = HFEnhancedAutoformer(self.configs)
        logger.info(f"Created HF backbone: {type(self.hf_backbone).__name__}")
        
        # Initialize extensions based on config
        self.extensions = nn.ModuleDict()
        self._init_extensions()
        
        # Loss manager (leverages existing infrastructure)
        self._init_loss_manager()
        
        # Metadata for tracking
        self.model_info = self._get_model_info()
        
        logger.info(f"Created HFAdvancedModelWrapper with {len(self.extensions)} extensions")
        
    def _ensure_namespace(self, configs):
        """Ensure configs is a Namespace object"""
        if isinstance(configs, dict):
            return Namespace(**configs)
        elif hasattr(configs, '__dict__'):
            return configs
        else:
            raise ValueError(f"Configs must be dict or Namespace, got {type(configs)}")
            
    def _init_extensions(self):
        """Initialize extensions based on configuration"""
        
        # Bayesian extension
        if getattr(self.configs, 'use_bayesian', False):
            self.extensions['bayesian'] = create_bayesian_extension(self.configs, self.hf_backbone)
            logger.info("Added Bayesian extension")
            
        # Quantile extension
        if getattr(self.configs, 'use_quantile', False):
            self.extensions['quantile'] = create_quantile_extension(self.configs)
            logger.info("Added Quantile extension")
            
        # Wavelet extension
        if getattr(self.configs, 'use_wavelet', False):
            self.extensions['wavelet'] = create_wavelet_processor(self.configs)
            logger.info("Added Wavelet extension")
            
        # Hierarchical extension
        if getattr(self.configs, 'use_hierarchical', False):
            self.extensions['hierarchical'] = create_hierarchical_processor(self.configs)
            logger.info("Added Hierarchical extension")
            
    def _init_loss_manager(self):
        """Initialize loss manager using existing infrastructure"""
        
        # Get base loss function from existing infrastructure
        loss_type = getattr(self.configs, 'loss_function', 'mse')
        self.base_loss_fn, _ = get_loss_component(loss_type, **vars(self.configs))
        
        # Wrap with Bayesian loss if needed
        if 'bayesian' in self.extensions:
            self.loss_fn = create_bayesian_loss(self.base_loss_fn, self.configs)
            logger.info(f"Using Bayesian loss wrapper for {loss_type}")
        else:
            self.loss_fn = self.base_loss_fn
            logger.info(f"Using standard loss: {loss_type}")
            
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass through backbone and extensions.
        
        Processing order:
        1. Preprocessing extensions (wavelet, hierarchical)
        2. HF backbone
        3. Postprocessing extensions (quantile, bayesian)
        """
        
        # Preprocessing: Apply wavelet/hierarchical processing to input
        processed_input = self._apply_preprocessing_extensions(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Core HF backbone processing
        if 'bayesian' in self.extensions:
            # Use Bayesian extension for uncertainty
            backbone_output = self.extensions['bayesian'](
                processed_input['x_enc'], processed_input['x_mark_enc'],
                processed_input['x_dec'], processed_input['x_mark_dec'], mask
            )
        else:
            # Standard HF backbone
            backbone_output = self.hf_backbone(
                processed_input['x_enc'], processed_input['x_mark_enc'],
                processed_input['x_dec'], processed_input['x_mark_dec'], mask
            )
            
        # Postprocessing: Apply quantile expansion
        final_output = self._apply_postprocessing_extensions(backbone_output, processed_input)
        
        return final_output
        
    def _apply_preprocessing_extensions(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Apply preprocessing extensions (wavelet, hierarchical)"""
        
        processed_x_enc = x_enc
        
        # Apply wavelet processing
        if 'wavelet' in self.extensions:
            processed_x_enc = self.extensions['wavelet'](processed_x_enc)
            logger.debug("Applied wavelet preprocessing")
            
        # Apply hierarchical processing
        if 'hierarchical' in self.extensions:
            processed_x_enc = self.extensions['hierarchical'](processed_x_enc)
            logger.debug("Applied hierarchical preprocessing")
            
        return {
            'x_enc': processed_x_enc,
            'x_mark_enc': x_mark_enc,
            'x_dec': x_dec,
            'x_mark_dec': x_mark_dec
        }
        
    def _apply_postprocessing_extensions(self, backbone_output, input_data):
        """Apply postprocessing extensions (quantile)"""
        
        output = backbone_output
        
        # Apply quantile expansion
        if 'quantile' in self.extensions:
            output = self.extensions['quantile'](output, **input_data)
            logger.debug("Applied quantile postprocessing")
            
        return output
        
    def compute_loss(self, predictions, targets, x_enc=None):
        """
        Compute loss using existing infrastructure.
        
        This leverages the complete loss ecosystem from losses.py and bayesian_losses.py
        """
        
        # Standard loss computation
        if isinstance(predictions, dict) and 'prediction' in predictions:
            # Handle uncertainty output from Bayesian extension
            pred_tensor = predictions['prediction']
        else:
            pred_tensor = predictions
            
        # Compute base loss
        base_loss = self.loss_fn(pred_tensor, targets)
        
        total_loss = base_loss
        loss_components = {'base_loss': base_loss}
        
        # Add KL loss if Bayesian
        if 'bayesian' in self.extensions:
            kl_loss = self.extensions['bayesian'].get_kl_loss()
            total_loss = total_loss + kl_loss
            loss_components['kl_loss'] = kl_loss
            
        # Add regularization losses if any
        reg_loss = self._compute_regularization_loss()
        if reg_loss > 0:
            total_loss = total_loss + reg_loss
            loss_components['regularization_loss'] = reg_loss
            
        return total_loss, loss_components
        
    def _compute_regularization_loss(self):
        """Compute regularization losses from extensions"""
        reg_loss = 0.0
        
        # Wavelet regularization
        if 'wavelet' in self.extensions and hasattr(self.extensions['wavelet'], 'reconstruction_weights'):
            # L2 regularization on reconstruction weights
            weights = self.extensions['wavelet'].reconstruction_weights
            reg_loss += 1e-4 * torch.sum(weights ** 2)
            
        # Hierarchical regularization
        if 'hierarchical' in self.extensions and hasattr(self.extensions['hierarchical'], 'level_weights'):
            # Encourage balanced level weights
            weights = self.extensions['hierarchical'].level_weights
            target_weight = 1.0 / len(weights)
            reg_loss += 1e-3 * torch.sum((weights - target_weight) ** 2)
            
        return reg_loss
        
    def get_model_info(self):
        """Get comprehensive model information"""
        return self.model_info
        
    def _get_model_info(self):
        """Collect model information"""
        info = {
            'backbone': type(self.hf_backbone).__name__,
            'extensions': list(self.extensions.keys()),
            'parameters': {
                'total': sum(p.numel() for p in self.parameters()),
                'backbone': sum(p.numel() for p in self.hf_backbone.parameters()),
                'extensions': sum(p.numel() for ext in self.extensions.values() for p in ext.parameters())
            },
            'capabilities': {
                'uncertainty_quantification': 'bayesian' in self.extensions,
                'quantile_regression': 'quantile' in self.extensions,
                'multi_scale_processing': 'hierarchical' in self.extensions,
                'wavelet_analysis': 'wavelet' in self.extensions,
                'advanced_losses': hasattr(self, 'loss_fn')
            }
        }
        
        # Extension-specific info
        for name, extension in self.extensions.items():
            if hasattr(extension, f'get_{name}_info'):
                info[f'{name}_info'] = getattr(extension, f'get_{name}_info')()
            elif hasattr(extension, 'get_uncertainty_info'):
                info[f'{name}_info'] = extension.get_uncertainty_info()
            elif hasattr(extension, 'get_wavelet_info'):
                info[f'{name}_info'] = extension.get_wavelet_info()
            elif hasattr(extension, 'get_hierarchical_info'):
                info[f'{name}_info'] = extension.get_hierarchical_info()
                
        return info


# Factory functions for different model configurations

def create_hf_bayesian_model(configs):
    """
    Create HF model with Bayesian uncertainty quantification.
    
    Features:
    - HF backbone (unchanged)
    - Bayesian uncertainty estimation
    - Advanced loss functions from existing infrastructure
    """
    
    # Configure for Bayesian
    if isinstance(configs, dict):
        configs = configs.copy()
    else:
        configs = Namespace(**vars(configs))
        
    configs.use_bayesian = True
    configs.uncertainty_method = getattr(configs, 'uncertainty_method', 'bayesian')
    configs.n_samples = getattr(configs, 'n_samples', 10)
    
    model = HFAdvancedModelWrapper(configs)
    logger.info(f"Created HF Bayesian model with {model.model_info['parameters']['total']:,} parameters")
    
    return model


def create_hf_hierarchical_model(configs):
    """
    Create HF model with hierarchical multi-scale processing.
    
    Features:
    - HF backbone (unchanged)  
    - Multi-scale temporal modeling
    - Wavelet decomposition
    """
    
    # Configure for Hierarchical
    if isinstance(configs, dict):
        configs = configs.copy()
    else:
        configs = Namespace(**vars(configs))
        
    configs.use_hierarchical = True
    configs.use_wavelet = getattr(configs, 'use_wavelet', True)
    configs.hierarchy_levels = getattr(configs, 'hierarchy_levels', [1, 2, 4])
    
    model = HFAdvancedModelWrapper(configs)
    logger.info(f"Created HF Hierarchical model with {model.model_info['parameters']['total']:,} parameters")
    
    return model


def create_hf_quantile_model(configs):
    """
    Create HF model with quantile regression.
    
    Features:
    - HF backbone (unchanged)
    - Quantile regression
    - Pinball loss from existing infrastructure
    """
    
    # Configure for Quantile
    if isinstance(configs, dict):
        configs = configs.copy()
    else:
        configs = Namespace(**vars(configs))
        
    configs.use_quantile = True
    configs.quantile_levels = getattr(configs, 'quantile_levels', [0.1, 0.5, 0.9])
    configs.loss_function = getattr(configs, 'loss_function', 'pinball')
    
    model = HFAdvancedModelWrapper(configs)
    logger.info(f"Created HF Quantile model with {model.model_info['parameters']['total']:,} parameters")
    
    return model


def create_hf_full_model(configs):
    """
    Create HF model with all advanced features.
    
    Features:
    - HF backbone (unchanged)
    - Bayesian uncertainty quantification
    - Hierarchical multi-scale processing
    - Quantile regression
    - Complete loss ecosystem integration
    """
    
    # Configure for all features
    if isinstance(configs, dict):
        configs = configs.copy()
    else:
        configs = Namespace(**vars(configs))
        
    configs.use_bayesian = True
    configs.use_hierarchical = True
    configs.use_wavelet = True
    configs.use_quantile = True
    
    # Bayesian settings
    configs.uncertainty_method = getattr(configs, 'uncertainty_method', 'bayesian')
    configs.n_samples = getattr(configs, 'n_samples', 10)
    
    # Hierarchical settings
    configs.hierarchy_levels = getattr(configs, 'hierarchy_levels', [1, 2, 4])
    configs.wavelet_type = getattr(configs, 'wavelet_type', 'db4')
    
    # Quantile settings
    configs.quantile_levels = getattr(configs, 'quantile_levels', [0.1, 0.5, 0.9])
    
    # Loss function
    configs.loss_function = getattr(configs, 'loss_function', 'pinball')
    
    model = HFAdvancedModelWrapper(configs)
    logger.info(f"Created HF Full model with {model.model_info['parameters']['total']:,} parameters")
    logger.info(f"Capabilities: {list(model.model_info['capabilities'].keys())}")
    
    return model


def create_hf_model_from_config(configs, model_type='auto'):
    """
    Create HF model based on configuration and type.
    
    Args:
        configs: Model configuration
        model_type: 'auto', 'standard', 'bayesian', 'hierarchical', 'quantile', 'full'
        
    Returns:
        Configured HF model
    """
    
    if model_type == 'auto':
        # Auto-detect based on config
        has_bayesian = getattr(configs, 'use_bayesian', False) or getattr(configs, 'uncertainty_method', None)
        has_hierarchical = getattr(configs, 'use_hierarchical', False) or getattr(configs, 'hierarchy_levels', None)
        has_quantile = getattr(configs, 'use_quantile', False) or getattr(configs, 'quantile_levels', None)
        
        if has_bayesian and has_hierarchical and has_quantile:
            model_type = 'full'
        elif has_bayesian:
            model_type = 'bayesian'
        elif has_hierarchical:
            model_type = 'hierarchical'
        elif has_quantile:
            model_type = 'quantile'
        else:
            model_type = 'standard'
            
        logger.info(f"Auto-detected model type: {model_type}")
        
    if model_type == 'standard':
        return HFEnhancedAutoformer(configs)
    elif model_type == 'bayesian':
        return create_hf_bayesian_model(configs)
    elif model_type == 'hierarchical':
        return create_hf_hierarchical_model(configs)
    elif model_type == 'quantile':
        return create_hf_quantile_model(configs)
    elif model_type == 'full':
        return create_hf_full_model(configs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Convenience aliases following existing naming patterns
def HFBayesianEnhancedAutoformer(configs):
    """Alias for Bayesian HF model following existing naming"""
    return create_hf_bayesian_model(configs)


def HFHierarchicalEnhancedAutoformer(configs):
    """Alias for Hierarchical HF model following existing naming"""
    return create_hf_hierarchical_model(configs)


def HFQuantileEnhancedAutoformer(configs):
    """Alias for Quantile HF model following existing naming"""
    return create_hf_quantile_model(configs)


def HFFullEnhancedAutoformer(configs):
    """Alias for Full HF model following existing naming"""
    return create_hf_full_model(configs)
