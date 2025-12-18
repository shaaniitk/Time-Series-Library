#!/usr/bin/env python3
"""
Training Script for Celestial Enhanced PGAT
Revolutionary Astrological AI for Financial Time Series Forecasting
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import argparse
import time
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import logging
import warnings
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')


from types import SimpleNamespace
from layers.modular.dimensions.dimension_manager import DimensionManager
from utils.scaler_manager import ScalerManager
from data_provider.data_factory import setup_financial_forecasting_data # <-- Add this import

def main():
    parser = argparse.ArgumentParser(description='Train Celestial Enhanced PGAT')
    parser.add_argument('--config', type=str, default='configs/celestial_enhanced_pgat.yaml',
                       help='Path to configuration file')
    parser.add_argument('--collect_diagnostics', action='store_true',
                       help='Enable deep diagnostic logging (gradients, activations, anomalies)')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Merge config_dict into args Namespace
    # Merge config_dict into args Namespace, handling nested dictionaries (like 'data')
    for key, value in config_dict.items():
        if isinstance(value, dict):
             # For nested dicts like 'data', we flatten them into the main args
             # This handles args.data.root_path -> args.root_path
             for sub_key, sub_value in value.items():
                 setattr(args, sub_key, sub_value)
             # Also keep the original dict for compatibility if needed (some code might check args.data)
             setattr(args, key, value) 
        else:
             setattr(args, key, value)

    # HACK: Force CPU if CUDA is not available or compiled with torch
    if not torch.cuda.is_available(): 
        args.use_gpu = False
        print("⚠️  CUDA not available or Torch not compiled with CUDA. Forcing CPU mode.")

    # Set a default task_name for forecasting, as expected by data_provider
    args.task_name = getattr(args, 'task_name', 'long_term_forecast')
    # Set default num_workers if not specified in config
    args.num_workers = getattr(args, 'num_workers', 0)
    # Set default scale flag if not specified in config
    args.scale = getattr(args, 'scale', True)
    # Set default gpu_type if not specified in config
    args.gpu_type = getattr(args, 'gpu_type', 'cuda')
    # Set default use_multi_gpu flag if not specified in config
    args.use_multi_gpu = getattr(args, 'use_multi_gpu', False)
    # Set default gpu index if not specified in config
    args.gpu = getattr(args, 'gpu', 0)
    # Set default checkpoints path if not specified in config
    args.checkpoints = getattr(args, 'checkpoints', './checkpoints')
    # Set default use_amp flag if not specified in config
    args.use_amp = getattr(args, 'use_amp', False)
    # Set default loss function if not specified in config
    args.loss = getattr(args, 'loss', 'mse')
    # Set default use_future_celestial_conditioning flag
    args.use_future_celestial_conditioning = getattr(args, 'use_future_celestial_conditioning', False)

    # Setup device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        args.device = torch.device('cpu')
        print("💻 Using CPU")

    print(f"Args in main before experiment init: {args}")

    # --- NEW: Use robust data pipeline for financial forecasting ---
    print("Step 1: Setting up robust financial forecasting data pipeline...")
    train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(args)

    # Attach managers and loaders to args
    args.train_loader = train_loader
    args.vali_loader = vali_loader
    args.test_loader = test_loader
    args.scaler_manager = scaler_manager
    args.dim_manager = dim_manager

    print(f"Step 2: Initialized DimensionManager with {dim_manager.enc_in} input features and {dim_manager.num_targets} targets.")
    print("Step 3: ScalerManager setup and fitted.")
    print("Step 4: Dataloaders for train, validation, and test prepared.")

    # Initialize experiment
    exp = Exp_Long_Term_Forecast(args)

    print("DEBUG: Experiment initialized. Constructing setting string...", flush=True)

    print("🌌 Celestial Enhanced PGAT Training with Standard Exp_Long_Term_Forecast")
    print("=" * 60)

    # Construct a base setting string that does *not* include the unique ID placeholder yet
    try:
        base_setting_string = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
            getattr(args, 'model', 'unknown_model'),
            getattr(args, 'model', 'unknown_model'),
            # Fix: args.data is a dict which causes formatting issues later. Use data_path or dataset.
            getattr(args, 'data_path', 'data'),
            getattr(args, 'features', 'S'),
            getattr(args, 'seq_len', 0),
            getattr(args, 'label_len', 0),
            getattr(args, 'pred_len', 0),
            getattr(args, 'd_model', 0),
            getattr(args, 'n_heads', 0),
            getattr(args, 'e_layers', 0),
            getattr(args, 'd_layers', 0),
            getattr(args, 'd_ff', 0),
            getattr(args, 'factor', 0),
            getattr(args, 'embed', 'FIX'),
            getattr(args, 'distil', True),
            getattr(args, 'des', 'Exp')) # A default description for the experiment
        print(f"DEBUG: Base setting string constructed: {base_setting_string}", flush=True)
    except Exception as e:
        print(f"DEBUG: Error constructing setting string: {e}")
        base_setting_string = 'fallback_setting'

    # Unique run ID for checkpoints
    ii = 0
    # Construct full path for existence check. Include a placeholder for ii.
    # The format call here will only operate on the _ii{} part.
    checkpoint_path_base = os.path.join(getattr(args, 'checkpoints', './checkpoints'), base_setting_string)
    checkpoint_full_path_template = checkpoint_path_base + '_ii{}' # Add the unique ID placeholder here
    
    print(f"DEBUG: Checking checkpoints at {checkpoint_path_base}...", flush=True)

    loop_count = 0
    while os.path.exists(checkpoint_full_path_template.format(ii)):
        ii += 1
        loop_count += 1
        if loop_count % 1000 == 0:
             print(f"DEBUG: Checked {loop_count} checkpoint paths...", flush=True)
    
    print(f"DEBUG: Found unique ID: {ii}", flush=True)

    # Final setting string for Exp_Long_Term_Forecast
    setting = checkpoint_full_path_template.format(ii) # Apply unique run ID to the final setting string

    print(f"Using setting: {setting}", flush=True)
    
    # Call the train method of the experiment
    print(f"DEBUG: Calling exp.train({setting})...", flush=True)
    exp.train(setting)
    
    # Call the test method of the experiment
    exp.test(setting)

    print("\n" + "=" * 60)
    print("🎉 Celestial Enhanced PGAT Training (via Exp_Long_Term_Forecast) Complete!")
    return 0


if __name__ == "__main__":
    exit(main())