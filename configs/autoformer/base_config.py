from argparse import Namespace
from layers.modular.losses import get_loss_component

class DimensionManager:
    """
    Manages the dynamic dimensions of the model based on inputs and data flow.
    """
    def __init__(self, num_targets, num_covariates, output_dim_multiplier,
                 encoder_input_type='targets_and_covariates', 
                 decoder_input_type='targets_and_covariates'):
        
        self.num_targets = num_targets
        self.num_covariates = num_covariates
        self.output_dim_multiplier = output_dim_multiplier
        self.encoder_input_type = encoder_input_type
        self.decoder_input_type = decoder_input_type

    @property
    def enc_in(self):
        if self.encoder_input_type == 'targets_and_covariates':
            return self.num_targets + self.num_covariates
        elif self.encoder_input_type == 'targets_only':
            return self.num_targets
        else:
            raise ValueError(f"Unknown encoder_input_type: {self.encoder_input_type}")

    @property
    def dec_in(self):
        if self.decoder_input_type == 'targets_and_covariates':
            return self.num_targets + self.num_covariates
        elif self.decoder_input_type == 'covariates_only':
            return self.num_covariates
        else:
            raise ValueError(f"Unknown decoder_input_type: {self.decoder_input_type}")

    @property
    def c_out(self):
        """The final output dimension of the model."""
        return self.num_targets * self.output_dim_multiplier
    
    @property
    def c_out_evaluation(self):
        """The number of base target variables for evaluation."""
        return self.num_targets

def get_base_config(num_targets, num_covariates, **kwargs):
    """
    Creates a base configuration Namespace with managed dimensions.
    """
    loss_type = kwargs.get('loss_function_type', 'mse')
    loss_params = kwargs.get('loss_params', {})
    _, output_dim_multiplier = get_loss_component(loss_type, **loss_params)
    
    dim_manager = DimensionManager(num_targets, num_covariates, output_dim_multiplier,
                                 kwargs.get('encoder_input_type', 'targets_and_covariates'),
                                 kwargs.get('decoder_input_type', 'targets_and_covariates'))

    config = {
        'task_name': 'long_term_forecast',
        'seq_len': 96, 'label_len': 48, 'pred_len': 24,
        'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 2048,
        'dropout': 0.1, 'embed': 'timeF', 'freq': 'h', 'activation': 'gelu',
        'num_class': 1, 'norm_type': 'LayerNorm', 'factor': 1,

        'enc_in': dim_manager.enc_in,
        'dec_in': dim_manager.dec_in,
        'c_out': dim_manager.c_out,
        'c_out_evaluation': dim_manager.c_out_evaluation,
        
        'loss_function_type': loss_type,
        'attention_type': 'autocorrelation_layer',
        'decomposition_type': 'series_decomp',
        'encoder_type': 'standard',
        'decoder_type': 'standard',
        'sampling_type': 'deterministic',
        'output_head_type': 'standard',

        'attention_params': {}, 'decomposition_params': {}, 'encoder_params': {},
        'decoder_params': {}, 'sampling_params': {}, 'output_head_params': {},
        'init_decomposition_params': {}, 'loss_params': {},
    }
    
    config.update(kwargs)

    # Set dependent parameters.
    # The init_decomp often needs different parameters (like input_dim=dec_in)
    # than the decomp inside the model layers (input_dim=d_model).
    # We copy the base decomp params and let specific configs override.
    config['init_decomposition_params'] = config['decomposition_params'].copy()
    
    config['attention_params'].update({
        'd_model': config['d_model'], 'n_heads': config['n_heads'], 
        'dropout': config['dropout'], 'factor': config['factor'], 'output_attention': False
    })
    
    config['encoder_params'].update({
        'e_layers': config['e_layers'], 'd_model': config['d_model'], 'n_heads': config['n_heads'],
        'd_ff': config['d_ff'], 'dropout': config['dropout'], 'activation': config['activation']
    })
    config['decoder_params'].update({
        'd_layers': config['d_layers'], 'd_model': config['d_model'], 'c_out': config['c_out_evaluation'], 
        'n_heads': config['n_heads'], 'd_ff': config['d_ff'], 'dropout': config['dropout'], 'activation': config['activation']
    })
    config['output_head_params'].update({
        'd_model': config['d_model'], 'c_out': config['c_out']
    })

    return Namespace(**config)
