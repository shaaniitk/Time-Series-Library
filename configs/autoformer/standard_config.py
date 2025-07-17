
from .base_config import get_base_config

def get_standard_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the standard Autoformer model.
    """
    # Start with the base configuration
    config = get_base_config(num_targets, num_covariates, **kwargs)

    # Override components for the standard Autoformer
    config.attention_type = 'autocorrelation_layer'
    config.decomposition_type = 'series_decomp'
    config.encoder_type = 'standard'
    config.decoder_type = 'standard'
    config.sampling_type = 'deterministic'
    config.output_head_type = 'standard'
    
    # Specific parameters for standard Autoformer
    config.attention_params = {'d_model': config.d_model, 'n_heads': config.n_heads, 'dropout': config.dropout, 'factor': 1, 'output_attention': False}
    config.decomposition_params = {'kernel_size': 25} # Default moving average
    config.init_decomposition_params = {'kernel_size': 25}

    return config
