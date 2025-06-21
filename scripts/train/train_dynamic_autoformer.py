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


# Main function to run the script
def main():
    import argparse
    import yaml
    
    log.info("Starting script execution")
    parser = argparse.ArgumentParser(description='Train DLinear model')
    parser.add_argument('--config', type=str, default='configs/config_DLinear_M_light.yaml', help='Path to the config file')
    parser.add_argument('--model_type', type=str, default='enhanced', help='Type of model to run [enhanced, bayesian, hierarchical]')
    
    # Parse command-line arguments first
    cmd_args = parser.parse_args()
    
    log.info(f"Loading config from {cmd_args.config}")
    try:
        with open(cmd_args.config, 'r') as f:
            config = yaml.safe_load(f)
            log.info(f"Config loaded successfully: {config}")
        
        # Convert config to SimpleNamespace for compatibility
        args = SimpleNamespace(**config) # This 'args' now contains config file parameters
        
        # Merge command-line arguments into the 'args' object
        # This ensures model_type (and other cmd-line specific args) are preserved
        args.model_type = cmd_args.model_type
        args.config = cmd_args.config # Also preserve the config path if needed later
        
        # Override model from config if model_type is specified
        if args.model_type == 'bayesian':
            args.model = 'BayesianEnhancedAutoformer'
        
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
            args.embed = 'timeF'  # Default embedding type
        if not hasattr(args, 'activation'):
            args.activation = 'gelu'  # Default activation
        if not hasattr(args, 'output_attention'):
            args.output_attention = False  # Default output_attention
        
        # Add common missing architectural/task defaults if not present in config
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
        
        # Dynamically set use_gpu if not specified in config
        if not hasattr(args, 'use_gpu'):
            if torch.cuda.is_available():
                args.use_gpu = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.use_gpu = True
            else:
                args.use_gpu = False
            log.info(f"Dynamically set use_gpu to {args.use_gpu} based on hardware detection.") # This line already exists

        # Dynamically set gpu_type if not specified in config
        if not hasattr(args, 'gpu_type'):
            if torch.cuda.is_available():
                args.gpu_type = 'cuda'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.gpu_type = 'mps'
            else:
                args.gpu_type = 'cpu'
            log.info(f"Dynamically set gpu_type to {args.gpu_type} based on hardware detection.")
        
        # Add default validation and test lengths if not present in config
        # Ensure validation_length and test_length are at least seq_len + pred_len
        min_len_for_dataset = args.seq_len + args.pred_len
        if not hasattr(args, 'validation_length'):
            args.validation_length = max(150, min_len_for_dataset) # Default value, adjust as needed
        else: args.validation_length = max(args.validation_length, min_len_for_dataset)
        if not hasattr(args, 'test_length'):
            args.test_length = max(120, min_len_for_dataset) # Default value, adjust as needed
        else: args.test_length = max(args.test_length, min_len_for_dataset)
        
        log.info(f"Converted config to args: {args.__dict__}")
        
        # --- STEP 1: SETUP THE ENTIRE DATA PIPELINE WITH ONE CALL ---
        log.info("Setting up data pipeline...")
        train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(args)
        log.info("Data pipeline setup complete.")

        # Initialize and run experiment
        log.info("Initializing experiment")
        # Pass the managers and loaders to the experiment class via the args object
        args.train_loader = train_loader
        args.vali_loader = vali_loader
        args.test_loader = test_loader
        args.scaler_manager = scaler_manager
        args.dim_manager = dim_manager

        # Instantiate Exp_Long_Term_Forecast with the updated args
        exp = Exp_Long_Term_Forecast(args)
        log.info("Experiment initialized successfully")
        
        log.info("Starting training")
        setting = f"{args.model_id}_{args.features}_{args.seq_len}_{args.pred_len}"
        exp.train(setting)
        
        log.info("Starting testing")
        exp.test(setting, test=1)
        
        log.info("Script execution completed successfully")
            
    except Exception as e:
        log.error(f"Error loading config: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()