
from .base_config import get_base_config

def get_quantile_bayesian_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the QuantileBayesianAutoformer model.
    Combines Bayesian uncertainty quantification with quantile regression.
    """
    # Set required parameters for quantile bayesian model
    kwargs['loss_function_type'] = 'bayesian_quantile'  # Use proper Bayesian quantile loss
    kwargs['quantile_levels'] = kwargs.get('quantile_levels', [0.1, 0.5, 0.9])
    kwargs['loss_params'] = {'quantiles': kwargs['quantile_levels']}
    kwargs['output_head_type'] = 'quantile'  # Required for quantile outputs
    kwargs['sampling_type'] = 'bayesian'     # Required for uncertainty quantification
    kwargs['attention_type'] = 'adaptive_autocorrelation_layer'
    kwargs['decomposition_type'] = 'learnable_decomp'
    kwargs['encoder_type'] = 'enhanced'
    kwargs['decoder_type'] = 'enhanced'

    config = get_base_config(num_targets, num_covariates, **kwargs)

    # Update component parameters
    config.decomposition_params.update({'input_dim': config.d_model})
    config.init_decomposition_params.update({'input_dim': config.dec_in})
    # Fix: Pass base number of targets to quantile output head, not pre-multiplied value
    config.output_head_params.update({
        'c_out': config.c_out_evaluation,  # Use base targets (7), not multiplied (35)
        'num_quantiles': len(config.quantile_levels)
    })
    config.sampling_params.update({
        'n_samples': kwargs.get('n_samples', 50), 
        'quantile_levels': config.quantile_levels
    })
    
    # Bayesian layer configuration
    config.bayesian_layers = ['projection']
    
    return config

    return config
