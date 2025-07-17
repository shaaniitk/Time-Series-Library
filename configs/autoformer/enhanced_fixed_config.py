
from .base_config import get_base_config

def get_enhanced_fixed_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the EnhancedAutoformer_Fixed model.
    """
    kwargs['decomposition_type'] = 'stable_decomp'
    kwargs['encoder_type'] = 'enhanced'
    kwargs['decoder_type'] = 'enhanced'
    kwargs['attention_type'] = 'adaptive_autocorrelation_layer'

    config = get_base_config(num_targets, num_covariates, **kwargs)

    config.decomposition_params.update({'kernel_size': 25})
    config.init_decomposition_params.update({'kernel_size': 25})

    return config


    return config
