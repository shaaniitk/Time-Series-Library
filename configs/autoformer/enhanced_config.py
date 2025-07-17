
from .base_config import get_base_config

def get_enhanced_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the EnhancedAutoformer model.
    """
    # Set the specific components for this model
    kwargs['attention_type'] = 'adaptive_autocorrelation_layer'
    kwargs['decomposition_type'] = 'learnable_decomp'
    kwargs['encoder_type'] = 'enhanced'
    kwargs['decoder_type'] = 'enhanced'
    
    # Get the base configuration with these components
    config = get_base_config(num_targets, num_covariates, **kwargs)

    # Set specific parameters for the components
    config.decomposition_params.update({'input_dim': config.d_model})
    config.init_decomposition_params.update({'input_dim': config.dec_in})

    return config
