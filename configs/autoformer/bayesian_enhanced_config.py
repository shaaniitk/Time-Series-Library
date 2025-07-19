
from .base_config import get_base_config

def get_bayesian_enhanced_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the BayesianEnhancedAutoformer model.
    """
    config = get_base_config(num_targets, num_covariates, **kwargs)

    # Inherits from Enhanced
    config.attention_type = 'adaptive_autocorrelation_layer'
    config.decomposition_type = 'learnable_decomp'
    config.encoder_type = 'enhanced'
    config.decoder_type = 'enhanced'
    
    # Key difference: Bayesian sampling with proper loss function
    config.sampling_type = 'bayesian'
    config.output_head_type = 'standard'
    
    # CRITICAL: Bayesian models MUST use Bayesian loss functions
    # This enforces proper uncertainty quantification training
    config.loss_function_type = 'bayesian'  # Changed from 'mse' to 'bayesian'
    
    # Parameters
    config.attention_params = {
        'd_model': config.d_model, 'n_heads': config.n_heads, 'dropout': config.dropout, 
        'factor': 1, 'output_attention': False
    }
    config.decomposition_params = {'input_dim': config.d_model, 'init_kernel_size': 25}
    config.init_decomposition_params = {'input_dim': config.dec_in, 'init_kernel_size': 25}
    config.sampling_params = {'n_samples': kwargs.get('n_samples', 50), 'quantile_levels': config.quantile_levels if hasattr(config, 'quantile_levels') else []}

    # This config would also require converting model layers to Bayesian,
    # which would be handled by a helper function during model construction.
    config.bayesian_layers = ['projection'] # Example

    return config
