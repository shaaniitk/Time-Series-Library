#!/usr/bin/env python3
"""
Debug Script: Overfit One Batch
Tests if the model can memorize a single batch perfectly.
If Train Loss drops but Validation Loss (on SAME batch) rises, we have a code bug.
If both drop, the model is fine, and the issue is Distributional Shift.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# --- MONKEY PATCH DIAGNOSTICS TO AVOID PERMISSION ERROR ---
# Must be done BEFORE importing any modules that use ModelDiagnostics
import sys
import types

# define dummy class
class DummyDiagnostics:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: {} # Return empty dict or None for any method call


# Create a mock module
mock_diag_module = types.ModuleType("models.celestial_modules.diagnostics")
mock_diag_module.ModelDiagnostics = DummyDiagnostics
# Inject into sys.modules
sys.modules["models.celestial_modules.diagnostics"] = mock_diag_module

# --- MONKEY PATCH CUDA SYNC ---
# REMOVED: GPU Activation Requested
# ------------------------------


# ----------------------------------------------------------

from data_provider.data_factory import setup_financial_forecasting_data
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleBatchLoader:
    def __init__(self, batch):
        self.batch = batch
        self.dataset = type('DummyDataset', (), {'__len__': lambda s: 1})() # Mock dataset for len() calls

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1

def main():
    # 1. Load Configuration (Use the fixed one)
    config_path = 'configs/celestial_enhanced_pgat.yaml'
    print(f"Loading config from {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 2. Create Args
    args = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
             for sub_key, sub_value in value.items():
                 setattr(args, sub_key, sub_value)
             setattr(args, key, value)
        else:
             setattr(args, key, value)
    
    # Force defaults for missing args
    args.task_name = 'long_term_forecast'
    args.use_gpu = True and torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.checkpoints = './checkpoints_debug'
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    args.model_id = 'DEBUG_ONE_BATCH'
    args.des = 'Debug'
    args.use_amp = False # Disable AMP for raw debugging
    args.num_workers = 0
    args.collect_diagnostics = False # Disable diagnostics to avoid PermissionError

    # Device
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    args.device = device
    print(f"Using device: {device}")

    # 3. Load Real Data (to get standard shapes/scalers)
    print("Setting up data pipeline...")
    train_loader_real, _, _, scaler_manager, dim_manager = setup_financial_forecasting_data(args)
    
    # 4. Extract ONE Batch
    print("Extracting one batch...")
    one_batch = next(iter(train_loader_real))
    
    # 5. Create Dummy Loaders
    dummy_loader = SingleBatchLoader(one_batch)
    
    # 6. Override Loaders in Args
    args.train_loader = dummy_loader
    args.vali_loader = dummy_loader # Validation sees EXACTLY the same data as Train
    args.test_loader = dummy_loader
    args.scaler_manager = scaler_manager
    args.dim_manager = dim_manager

    # 7. Initialize Experiment
    print("Initializing Experiment...")
    exp = Exp_Long_Term_Forecast(args)
    
    # 8. Train Loop (Shortened)
    print("\n" + "="*40)
    print("STARTING OVERFIT TEST")
    print("Goal: Train Loss and Val Loss must BOTH drop to near zero.")
    print("="*40 + "\n")
    
    # We manually run a simplified loop or call exp.train()?
    # Calling exp.train() is better to test the EXACT logic including .eval() calls
    args.train_epochs = 20 # Enough to memorize
    args.patience = 20 # Disable early stopping
    args.lradj = 'type1' # Simple decay
    args.learning_rate = 0.001 # Aggressive LR for memorization
    
    try:
        exp.train('DEBUG_SETTING')
    except Exception as e:
        logger.exception("Training failed")

if __name__ == '__main__':
    main()
