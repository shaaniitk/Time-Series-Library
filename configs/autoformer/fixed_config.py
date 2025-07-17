
from .base_config import get_base_config

def get_fixed_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the Autoformer_Fixed model.
    """
    config = get_base_config(num_targets, num_covariates, **kwargs)

    # Use stable decomposition
    config.decomposition_type = 'stable_decomp'
    
    # Other components are standard
    config.attention_type = 'autocorrelation_layer'
    config.encoder_type = 'standard'
    config.decoder_type = 'standard'
    config.sampling_type = 'deterministic'
    config.output_head_type = 'standard'
    
    config.attention_params = {'d_model': config.d_model, 'n_heads': config.n_heads, 'dropout': config.dropout, 'factor': 1, 'output_attention': False}
    config.decomposition_params = {'kernel_size': 25}
    config.init_decomposition_params = {'kernel_size': 25}

    return config
