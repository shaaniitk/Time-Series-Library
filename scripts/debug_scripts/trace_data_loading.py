#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import yaml
import argparse
import torch
from pathlib import Path

def create_args_from_config(config_path):
    """Create args namespace from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Add required attributes
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
    if not hasattr(args, 'device'):
        args.device = 'cpu'
    if not hasattr(args, 'use_gpu'):
        args.use_gpu = False
    if not hasattr(args, 'gpu'):
        args.gpu = 0
    if not hasattr(args, 'use_multi_gpu'):
        args.use_multi_gpu = False
    if not hasattr(args, 'devices'):
        args.devices = '0'
        
    return args

def trace_data_loading():
    """Trace the data loading process for both models"""
    
    print("=== Tracing Data Loading Process ===")
    
    # Test Enhanced PGAT
    print("\n1. Enhanced PGAT Data Loading:")
    enhanced_config = "configs/enhanced_sota_pgat_synthetic.yaml"
    
    try:
        enhanced_args = create_args_from_config(enhanced_config)
        
        # Import the data setup function
        from data_provider.data_factory import setup_financial_forecasting_data
        
        print(f"   Config loaded: {enhanced_args.model}")
        print(f"   Validation length: {enhanced_args.validation_length}")
        print(f"   Test length: {enhanced_args.test_length}")
        
        # Try to load data
        result = setup_financial_forecasting_data(enhanced_args)
        train_loader, vali_loader, test_loader, dim_manager, scaler_manager = result
        
        print(f"   Train loader steps: {len(train_loader)}")
        print(f"   Validation loader steps: {len(vali_loader)}")
        print(f"   Test loader steps: {len(test_loader)}")
        
        # Check first batch
        for i, batch in enumerate(train_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            print(f"   First batch shapes: x={batch_x.shape}, y={batch_y.shape}")
            break
            
    except Exception as e:
        print(f"   Enhanced PGAT data loading error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Base PGAT (using the enhanced_pgat_full_test config)
    print("\n2. Base PGAT Data Loading:")
    base_config = "configs/enhanced_pgat_full_test.yaml"
    
    try:
        base_args = create_args_from_config(base_config)
        
        print(f"   Config loaded")
        print(f"   Validation length: {base_args.validation_length}")
        print(f"   Test length: {base_args.test_length}")
        
        # Try to load data
        result = setup_financial_forecasting_data(base_args)
        train_loader, vali_loader, test_loader, dim_manager, scaler_manager = result
        
        print(f"   Train loader steps: {len(train_loader)}")
        print(f"   Validation loader steps: {len(vali_loader)}")
        print(f"   Test loader steps: {len(test_loader)}")
        
        # Check first batch
        for i, batch in enumerate(train_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            print(f"   First batch shapes: x={batch_x.shape}, y={batch_y.shape}")
            break
            
    except Exception as e:
        print(f"   Base PGAT data loading error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    trace_data_loading()