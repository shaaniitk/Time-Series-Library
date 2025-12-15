#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import yaml
import argparse
from pathlib import Path

def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_args_from_config(config_path):
    """Create args namespace from config file"""
    config = load_config(config_path)
    
    # Convert config to args namespace
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Add required attributes that might be missing
    if not hasattr(args, 'data'):
        args.data = 'custom'
    if not hasattr(args, 'embed'):
        args.embed = 'timeF'
    if not hasattr(args, 'validation_length'):
        args.validation_length = 0.1
    if not hasattr(args, 'test_length'):
        args.test_length = 0.1
    if not hasattr(args, 'dynamic_cov_path'):
        args.dynamic_cov_path = None
    if not hasattr(args, 'static_cov_path'):
        args.static_cov_path = None
        
    return args

def debug_training_steps():
    """Debug why Enhanced PGAT has fewer training steps"""
    
    print("=== Debugging Training Steps ===")
    
    # Enhanced PGAT config
    enhanced_config = "configs/enhanced_sota_pgat_synthetic.yaml"
    print(f"\n1. Enhanced PGAT Config: {enhanced_config}")
    
    if os.path.exists(enhanced_config):
        enhanced_args = create_args_from_config(enhanced_config)
        print(f"   Model: {enhanced_args.model}")
        print(f"   Batch size: {enhanced_args.batch_size}")
        print(f"   Train epochs: {enhanced_args.train_epochs}")
        print(f"   Data path: {enhanced_args.data_path}")
        print(f"   Seq len: {enhanced_args.seq_len}")
        print(f"   Pred len: {enhanced_args.pred_len}")
    else:
        print(f"   Config file not found!")
    
    # Base PGAT config (let's check what was used)
    base_config = "configs/enhanced_pgat_full_test.yaml"
    print(f"\n2. Base PGAT Config: {base_config}")
    
    if os.path.exists(base_config):
        base_args = create_args_from_config(base_config)
        print(f"   Model: {getattr(base_args, 'model', 'Not specified')}")
        print(f"   Batch size: {base_args.batch_size}")
        print(f"   Train epochs: {base_args.train_epochs}")
        print(f"   Data path: {base_args.data_path}")
        print(f"   Seq len: {base_args.seq_len}")
        print(f"   Pred len: {base_args.pred_len}")
    else:
        print(f"   Config file not found!")
    
    # Check dataset
    dataset_path = "data/synthetic_multi_wave.csv"
    print(f"\n3. Dataset Analysis: {dataset_path}")
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
        print(f"   Total rows: {len(lines)}")
        print(f"   Header: {lines[0].strip()}")
        
        # Calculate expected samples
        seq_len = 96
        total_samples = len(lines) - 1 - seq_len  # -1 for header, -seq_len for windowing
        print(f"   Expected samples (with seq_len={seq_len}): {total_samples}")
        
        # Calculate expected steps per epoch
        batch_size = 8
        steps_per_epoch = total_samples // batch_size
        print(f"   Expected steps per epoch (batch_size={batch_size}): {steps_per_epoch}")
        
    else:
        print(f"   Dataset file not found!")

if __name__ == "__main__":
    debug_training_steps()