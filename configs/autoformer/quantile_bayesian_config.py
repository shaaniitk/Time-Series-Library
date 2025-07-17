
from .base_config import get_base_config

def get_quantile_bayesian_autoformer_config(num_targets, num_covariates, **kwargs):
    """
    Returns the configuration for the QuantileBayesianAutoformer model.
    """
    kwargs['loss_function_type'] = 'quantile'
    kwargs['loss_params'] = {'quantiles': kwargs.get('quantile_levels', [0.1, 0.5, 0.9])}
    kwargs['output_head_type'] = 'quantile'
    kwargs['sampling_type'] = 'bayesian'
    kwargs['attention_type'] = 'adaptive_autocorrelation_layer'
    kwargs['decomposition_type'] = 'learnable_decomp'
    kwargs['encoder_type'] = 'enhanced'
    kwargs['decoder_type'] = 'enhanced'

    config = get_base_config(num_targets, num_covariates, **kwargs)

    config.decomposition_params.update({'input_dim': config.d_model})
    config.init_decomposition_params.update({'input_dim': config.dec_in})
    config.sampling_params.update({'n_samples': kwargs.get('n_samples', 50)})
    
    kwargs['loss_params'] = {'quantiles': kwargs.get('quantile_levels', [0.1, 0.5, 0.9])}
    kwargs['quantile_levels'] = kwargs['loss_params']['quantiles']
    kwargs['loss_function_type'] = 'quantile'
    kwargs['output_head_type'] = 'quantile'
    kwargs['sampling_type'] = 'bayesian'
    kwargs['attention_type'] = 'adaptive_autocorrelation_layer'
    kwargs['decomposition_type'] = 'learnable_decomp' 
    kwargs['encoder_type'] = 'enhanced'
    kwargs['decoder_type'] = 'enhanced'

    config = get_base_config(num_targets, num_covariates, **kwargs)

    config.decomposition_params.update({'input_dim': config.d_model})
    config.init_decomposition_params.update({'input_dim': config.dec_in})
    config.output_head_params.update({'num_quantiles': len(config.quantile_levels)})
    config.sampling_params.update({'n_samples': kwargs.get('n_samples', 50), 'quantile_levels': kwargs['loss_params']['quantiles']})
    
    config.bayesian_layers = ['projection']

    return config
