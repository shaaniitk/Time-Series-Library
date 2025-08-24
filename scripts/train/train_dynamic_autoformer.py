import os, sys, torch
import torch.nn as nn
from torch import optim
import numpy as np
import time
import warnings
import traceback
from types import SimpleNamespace

# Add the project root to the Python path
# This allows the script to find modules like `data_provider` and `exp`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Debug info to help diagnose import issues
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Project root:", project_root)
    
# Set up detailed logging
import logging
logging.basicConfig(
    level=logging.DEBUG, # Use DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Ensure project root is in path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    log.info(f"Added project root to path: {project_root}")

# --- Import our new, robust data factory and experiment class ---
from data_provider.data_factory import setup_financial_forecasting_data
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric
from utils.logger import logger

warnings.filterwarnings('ignore')


def _configure_modular_autoformer_args(args) -> None:
    """Populate args with modular component selections and params for ModularAutoformer.

    This keeps legacy CLI/YAML compatibility and lets ModularAutoformer consume a Namespace
    via its internal converter to ModularAutoformerConfig.
    """
    # Sensible defaults
    e_layers = getattr(args, 'e_layers', 2)
    d_layers = getattr(args, 'd_layers', 1)
    n_heads = getattr(args, 'n_heads', 8)
    d_model = getattr(args, 'd_model', 512)
    d_ff = getattr(args, 'd_ff', 2048)
    dropout = getattr(args, 'dropout', 0.1)
    activation = getattr(args, 'activation', 'gelu')
    factor = getattr(args, 'factor', 1)
    output_attention = getattr(args, 'output_attention', False)

    # Base component types (enhanced variant)
    args.attention_type = 'adaptive_autocorrelation_layer'
    args.decomposition_type = 'learnable_decomp'
    args.encoder_type = 'enhanced'
    args.decoder_type = 'enhanced'

    # Sampling / head / loss selections derived from model_type and quantiles
    model_type = getattr(args, 'model_type', 'enhanced')
    quantile_levels = getattr(args, 'quantile_levels', []) or []
    is_quantile = isinstance(quantile_levels, list) and len(quantile_levels) > 0

    if model_type == 'bayesian' and is_quantile:
        args.sampling_type = 'bayesian'
        args.output_head_type = 'quantile'
        args.loss_function_type = 'bayesian_quantile'
        args.loss = 'pinball'  # Exp selects PinballLoss for multi-quantile
    elif model_type == 'bayesian':
        args.sampling_type = 'bayesian'
        args.output_head_type = 'standard'
        args.loss_function_type = 'bayesian_mse'
        args.loss = 'mse'
    elif model_type == 'hierarchical':
        # Keep sampling deterministic by default; swap decomposition/encoder higher up if desired
        args.sampling_type = 'deterministic'
        args.output_head_type = 'standard'
        args.loss_function_type = 'mse'
        args.loss = 'mse'
        # Optional: enable a hierarchical flavor when requested
        args.decomposition_type = getattr(args, 'decomposition_type', 'wavelet_decomp')
        args.encoder_type = getattr(args, 'encoder_type', 'hierarchical')
    else:
        # enhanced baseline
        args.sampling_type = 'deterministic'
        args.output_head_type = 'quantile' if is_quantile else 'standard'
        args.loss_function_type = 'quantile_loss' if is_quantile else 'mse'
        args.loss = 'pinball' if is_quantile else 'mse'

    # Component params expected by ModularAutoformer._convert_namespace_to_structured
    args.attention_params = {
        'd_model': d_model,
        'n_heads': n_heads,
        'dropout': dropout,
        'factor': factor,
        'output_attention': output_attention,
        'seq_len': getattr(args, 'seq_len', 96),
    }

    # For learnable/wavelet decompositions
    args.decomposition_params = {
        'kernel_size': getattr(args, 'moving_avg', 25),
        'input_dim': d_model,
        # wavelet extras tolerated by schema via extra fields
        'levels': getattr(args, 'wavelet_levels', 3),
        'wavelet_type': getattr(args, 'wavelet_type', 'db4'),
    }

    args.encoder_params = {
        'e_layers': e_layers,
        'd_model': d_model,
        'n_heads': n_heads,
        'd_ff': d_ff,
        'dropout': dropout,
        'activation': activation,
    }

    # c_out is required by some decoder variants metadata; use evaluation targets
    args.decoder_params = {
        'd_layers': d_layers,
        'd_model': d_model,
        'n_heads': n_heads,
        'd_ff': d_ff,
        'dropout': dropout,
        'activation': activation,
        'c_out': getattr(args, 'c_out_evaluation', getattr(args, 'c_out', 1)),
    }

    args.sampling_params = {
        'n_samples': getattr(args, 'n_samples', 50),
        'quantile_levels': quantile_levels if is_quantile else None,
        'dropout_rate': getattr(args, 'bayesian_dropout', 0.1),
        'temperature': getattr(args, 'bayesian_temperature', 1.0),
    }

    # Output head params depend on quantiles
    args.output_head_params = {
        'd_model': d_model,
        'c_out': getattr(args, 'c_out_evaluation', getattr(args, 'c_out', 1)),
        'num_quantiles': len(quantile_levels) if is_quantile else None,
    }

    args.loss_params = {
        'quantiles': quantile_levels if is_quantile else None,
        'kl_weight': getattr(args, 'kl_weight', 1e-5),
        'prior_scale': getattr(args, 'prior_scale', 1.0),
    }

    # Bayesian layers to convert (projection by default)
    if model_type == 'bayesian':
        args.bayesian_layers = getattr(args, 'bayesian_layers', ['projection'])
    else:
        args.bayesian_layers = getattr(args, 'bayesian_layers', [])

    # Tell the experiment which model to build
    args.model = 'ModularAutoformer'

    # Debug summary of resolved modular configuration
    try:
        log.debug(
            "Modular config resolved | model=%s | model_type=%s | loss=%s | head=%s | sampling=%s | quantiles=%s | c_out_model=%s | c_out_eval=%s",
            args.model,
            model_type,
            getattr(args, 'loss', None),
            getattr(args, 'output_head_type', None),
            getattr(args, 'sampling_type', None),
            quantile_levels if is_quantile else [],
            getattr(args, 'c_out', None),
            getattr(args, 'c_out_evaluation', None),
        )
    except Exception:  # Logging must not break config
        pass


# Main function to run the script
def main():
    import argparse
    
    log.info("Starting script execution")
    parser = argparse.ArgumentParser(description='Train Modular Autoformer (dynamic)')
    parser.add_argument('--config', type=str, default='configs/modular_enhanced_light.yaml', help='Path to the config file')
    parser.add_argument('--model_type', type=str, default='enhanced', help='Type of model to run [enhanced, bayesian, hierarchical]')
    
    # Parse command-line arguments first
    cmd_args = parser.parse_args()
    
    log.info(f"Loading config from {cmd_args.config}")
    try:
        # Load PyYAML lazily to avoid hard import dependency at module load time
        import importlib
        yaml = importlib.import_module('yaml')
        with open(cmd_args.config, 'r') as f:
            config = yaml.safe_load(f)
            log.info(f"Config loaded successfully: {config}")

        # Convert config to SimpleNamespace for compatibility
        args = SimpleNamespace(**config)  # This 'args' now contains config file parameters

        # Merge command-line arguments into the 'args' object
        args.model_type = cmd_args.model_type
        args.config = cmd_args.config  # preserve config path

        # Add any missing default attributes
        if not hasattr(args, 'use_amp'):
            args.use_amp = False
        if not hasattr(args, 'augmentation_ratio'):
            args.augmentation_ratio = 0
        if not hasattr(args, 'use_multi_gpu'):
            args.use_multi_gpu = False
        if not hasattr(args, 'devices'):
            args.devices = '0'
        if not hasattr(args, 'quantile_levels'):
            args.quantile_levels = []
        if not hasattr(args, 'embed'):
            args.embed = 'timeF'
        if not hasattr(args, 'activation'):
            args.activation = 'gelu'
        if not hasattr(args, 'output_attention'):
            args.output_attention = False
        if not hasattr(args, 'scale'):
            args.scale = True

        # Common architectural/task defaults
        if not hasattr(args, 'task_name'):
            args.task_name = 'long_term_forecast'
        if not hasattr(args, 'seq_len'):
            args.seq_len = 96
        if not hasattr(args, 'label_len'):
            args.label_len = 48
        if not hasattr(args, 'pred_len'):
            args.pred_len = 24
        if not hasattr(args, 'd_model'):
            args.d_model = 512
        if not hasattr(args, 'freq'):
            args.freq = 'h'
        if not hasattr(args, 'dropout'):
            args.dropout = 0.1
        if not hasattr(args, 'factor'):
            args.factor = 1
        if not hasattr(args, 'n_heads'):
            args.n_heads = 8
        if not hasattr(args, 'd_ff'):
            args.d_ff = 2048
        # Learning rate schedule defaults
        if not hasattr(args, 'lradj'):
            # default policy used by adjust_learning_rate in utils.tools
            args.lradj = 'type1'
        if not hasattr(args, 'model_id'):
            args.model_id = 'modular'
        if not hasattr(args, 'features'):
            args.features = 'M'

        # Dynamically set use_gpu if not specified in config
        if not hasattr(args, 'use_gpu'):
            if torch.cuda.is_available():
                args.use_gpu = True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                args.use_gpu = True
            else:
                args.use_gpu = False
            log.info(f"Dynamically set use_gpu to {args.use_gpu} based on hardware detection.")

        # Dynamically set gpu_type if not specified in config
        if not hasattr(args, 'gpu_type'):
            if torch.cuda.is_available():
                args.gpu_type = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                args.gpu_type = 'mps'
            else:
                args.gpu_type = 'cpu'
            log.info(f"Dynamically set gpu_type to {args.gpu_type} based on hardware detection.")

        # Validation/test lengths >= seq_len + pred_len
        min_len_for_dataset = args.seq_len + args.pred_len
        if not hasattr(args, 'validation_length'):
            args.validation_length = max(150, min_len_for_dataset)
        else:
            args.validation_length = max(args.validation_length, min_len_for_dataset)
        if not hasattr(args, 'test_length'):
            args.test_length = max(120, min_len_for_dataset)
        else:
            args.test_length = max(args.test_length, min_len_for_dataset)

        log.info(f"Converted config to args: {args.__dict__}")

        # STEP 1: Data pipeline
        log.info("Setting up data pipeline...")
        train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(args)
        log.info("Data pipeline setup complete.")

        # Pass managers/loaders
        args.train_loader = train_loader
        args.vali_loader = vali_loader
        args.test_loader = test_loader
        args.scaler_manager = scaler_manager
        args.dim_manager = dim_manager

        # Configure ModularAutoformer
        args.enc_in = getattr(args, 'enc_in', dim_manager.enc_in)
        args.dec_in = getattr(args, 'dec_in', dim_manager.dec_in)
        # Always take model c_out from DimensionManager (includes quantile expansion when present)
        args.c_out = dim_manager.c_out_model
        args.c_out_evaluation = getattr(args, 'c_out_evaluation', dim_manager.c_out_evaluation)
        _configure_modular_autoformer_args(args)

        # Extra debug on dimension manager vs args
        log.debug(
            "DimensionManager -> enc_in=%s dec_in=%s c_out_model=%s c_out_eval=%s | args.c_out=%s args.c_out_eval=%s | quantiles=%s",
            dim_manager.enc_in,
            dim_manager.dec_in,
            dim_manager.c_out_model,
            dim_manager.c_out_evaluation,
            args.c_out,
            args.c_out_evaluation,
            getattr(args, 'quantile_levels', []),
        )

        # Instantiate and run
        exp = Exp_Long_Term_Forecast(args)
        log.info("Experiment initialized successfully")
        log.debug(f"Exp_Long_Term_Forecast instance has scaler_manager: {exp.scaler_manager is not None}")

        log.info("Starting training")
        setting = f"{args.model_id}_{args.features}_{args.seq_len}_{args.pred_len}"
        exp.train(setting)

        log.info("Starting testing")
        exp.test(setting, test=1)

        log.info("Script execution completed successfully")

    except Exception as e:
        log.error(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()