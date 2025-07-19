"""
ChronosX Configuration Module

This module provides configuration generators for ChronosX-based models,
enabling easy integration with the unified framework.
"""

from argparse import Namespace
from configs.autoformer.base_config import get_base_config


def get_chronosx_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for ChronosX Autoformer model.
    
    Args:
        num_targets: Number of target variables
        num_covariates: Number of covariate variables
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration namespace for ChronosX model
    """
    config = get_base_config(num_targets, num_covariates, **kwargs)
    
    # ChronosX specific configuration
    config.use_chronosx_backbone = True
    config.framework_type = 'chronosx'
    config.model_type = 'chronosx_autoformer'
    
    # ChronosX model settings
    config.chronosx_model_size = kwargs.get('chronosx_model_size', 'base')  # tiny, mini, small, base, large
    config.uncertainty_enabled = kwargs.get('uncertainty_enabled', True)
    config.num_samples = kwargs.get('num_samples', 20)
    config.device = kwargs.get('device', 'auto')
    
    # Input handling for multivariate data
    config.chronosx_input_channel = kwargs.get('chronosx_input_channel', 0)  # Which channel to use for ChronosX
    config.chronosx_aggregate_method = kwargs.get('chronosx_aggregate_method', 'first')  # first, mean, sum
    
    # Override some base settings for ChronosX compatibility
    config.task_name = 'long_term_forecast'
    config.embed = 'timeF'
    config.freq = kwargs.get('freq', 'h')
    
    return config


def get_chronosx_uncertainty_config(num_targets, num_covariates, **kwargs):
    """
    Returns configuration for ChronosX model with enhanced uncertainty quantification.
    
    Args:
        num_targets: Number of target variables
        num_covariates: Number of covariate variables
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration namespace optimized for uncertainty quantification
    """
    config = get_chronosx_autoformer_config(num_targets, num_covariates, **kwargs)
    
    # Enhanced uncertainty settings
    config.uncertainty_enabled = True
    config.num_samples = kwargs.get('num_samples', 50)  # More samples for better uncertainty
    config.chronosx_model_size = kwargs.get('chronosx_model_size', 'base')  # Larger model for uncertainty
    
    # Quantile support
    config.quantile_levels = kwargs.get('quantile_levels', [0.1, 0.25, 0.5, 0.75, 0.9])
    
    return config


def get_chronosx_multivariate_config(num_targets, num_covariates, **kwargs):
    """
    Returns configuration for ChronosX model optimized for multivariate time series.
    
    Args:
        num_targets: Number of target variables
        num_covariates: Number of covariate variables
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration namespace for multivariate forecasting
    """
    config = get_chronosx_autoformer_config(num_targets, num_covariates, **kwargs)
    
    # Multivariate specific settings
    if num_targets > 1 or num_covariates > 0:
        # Configure how to handle multiple variables with ChronosX
        config.chronosx_multivariate_strategy = kwargs.get('multivariate_strategy', 'channel_wise')
        config.chronosx_input_channels = kwargs.get('input_channels', list(range(num_targets)))
        
        # Output adaptation
        config.output_adaptation = kwargs.get('output_adaptation', 'neural_projection')
    
    return config


def get_chronosx_hf_hybrid_config(num_targets, num_covariates, **kwargs):
    """
    Returns configuration for hybrid ChronosX + HF model.
    
    This configuration enables both ChronosX backbone and HF integration features,
    providing the best of both frameworks.
    
    Args:
        num_targets: Number of target variables
        num_covariates: Number of covariate variables
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration namespace for hybrid model
    """
    config = get_chronosx_autoformer_config(num_targets, num_covariates, **kwargs)
    
    # Enable both frameworks
    config.use_chronosx_backbone = True
    config.use_hf_integration = True
    config.framework_type = 'hybrid'
    
    # HF integration settings
    config.hf_tokenizer = kwargs.get('hf_tokenizer', 'google/flan-t5-small')
    config.hf_adaptation_layers = kwargs.get('hf_adaptation_layers', True)
    
    # Advanced features
    config.cross_attention_enabled = kwargs.get('cross_attention_enabled', False)
    config.feature_fusion_method = kwargs.get('feature_fusion_method', 'concatenation')
    
    return config


def get_unified_framework_config(num_targets, num_covariates, framework_type='auto', **kwargs):
    """
    Returns configuration for the unified framework with automatic framework selection.
    
    Args:
        num_targets: Number of target variables
        num_covariates: Number of covariate variables
        framework_type: Framework type ('auto', 'custom', 'hf', 'chronosx', 'hybrid')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration namespace for the specified framework
    """
    # Base configuration
    config = get_base_config(num_targets, num_covariates, **kwargs)
    
    # Framework selection
    config.framework_type = framework_type
    
    if framework_type == 'auto':
        # Auto-select based on requirements
        if kwargs.get('uncertainty_required', False) or kwargs.get('num_samples', 0) > 1:
            framework_type = 'chronosx'
        elif kwargs.get('use_hf_models', False):
            framework_type = 'hf'
        else:
            framework_type = 'custom'
        config.framework_type = framework_type
    
    # Configure based on selected framework
    if framework_type == 'chronosx':
        chronosx_config = get_chronosx_autoformer_config(num_targets, num_covariates, **kwargs)
        config.__dict__.update(chronosx_config.__dict__)
    elif framework_type == 'hf':
        config.use_hf_backbone = True
        config.hf_model_name = kwargs.get('hf_model_name', 'amazon/chronos-t5-tiny')
    elif framework_type == 'hybrid':
        hybrid_config = get_chronosx_hf_hybrid_config(num_targets, num_covariates, **kwargs)
        config.__dict__.update(hybrid_config.__dict__)
    elif framework_type == 'custom':
        # Use modular framework
        config.use_backbone_component = kwargs.get('use_backbone_component', False)
        config.backbone_type = kwargs.get('backbone_type', None)
    
    return config


# Convenience function for backward compatibility
def create_chronosx_config(seq_len=96, pred_len=24, enc_in=1, c_out=1, **kwargs):
    """
    Create a simple ChronosX configuration for testing and quick setup.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction length
        enc_in: Number of input features
        c_out: Number of output features
        **kwargs: Additional parameters
        
    Returns:
        ChronosX configuration namespace
    """
    # Calculate targets and covariates
    num_targets = c_out
    num_covariates = max(0, enc_in - num_targets)
    
    return get_chronosx_autoformer_config(
        num_targets=num_targets,
        num_covariates=num_covariates,
        seq_len=seq_len,
        pred_len=pred_len,
        **kwargs
    )
