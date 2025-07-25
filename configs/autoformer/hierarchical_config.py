
from .base_config import get_base_config

def get_hierarchical_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the HierarchicalEnhancedAutoformer model.
    """
    kwargs['attention_type'] = 'adaptive_autocorrelation_layer'
    kwargs['decomposition_type'] = 'wavelet_decomp'  # For encoder/decoder (embedded space)
    kwargs['encoder_type'] = 'hierarchical'
    kwargs['decoder_type'] = 'enhanced'

    config = get_base_config(num_targets, num_covariates, **kwargs)

    config.decomposition_params.update({
        'wavelet_type': 'db4', 'levels': 3
    })
    
    # Use different decomposition for init_decomp (raw input space)
    config.init_decomposition_type = 'learnable_decomp'
    config.init_decomposition_params.update({
        'input_dim': config.dec_in,  # Raw input dimension
        'init_kernel_size': 25
    })
    config.attention_type = 'cross_resolution_attention'
    config.attention_params.update({
        'n_levels': 3,
    })
    config.encoder_params.update({
        'e_layers': config.e_layers,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'd_ff': config.d_ff,
        'dropout': config.dropout,
        'activation': config.activation,
        'n_levels': 3, 
        'decomp_type': 'wavelet_decomp',
        'attention_type': 'adaptive_autocorrelation_layer',
        'decomp_params': {
            'seq_len': config.seq_len,
            'd_model': config.d_model,
            **config.decomposition_params  # Include wavelet_type, levels, etc.
        }
    })
    config.decoder_params.update({
        'd_model': config.d_model,
        'c_out': config.c_out_evaluation,
        'n_heads': config.n_heads,
        'd_ff': config.d_ff,
        'dropout': config.dropout,
        'activation': config.activation,
    })
    config.fusion_type = 'hierarchical_fusion'
    config.fusion_params = {
        'd_model': config.d_model,
        'n_levels': 3,
    }

    return config


