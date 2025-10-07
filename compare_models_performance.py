#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import torch
import yaml
import argparse
import time
from pathlib import Path

def create_args_from_config(config_path):
    """Create args namespace from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Add required attributes
    required_attrs = {
        'data': 'custom',
        'embed': 'timeF',
        'validation_length': 0.1,
        'test_length': 0.1,
        'dynamic_cov_path': None,
        'static_cov_path': None,
        'device': 'cpu',
        'use_gpu': False,
        'gpu': 0,
        'use_multi_gpu': False,
        'devices': '0',
        'task_name': 'long_term_forecast',
        'is_training': 1,
        'model_id': 'test',
        'checkpoints': './checkpoints/',
        'des': 'test'
    }
    
    for attr, default_value in required_attrs.items():
        if not hasattr(args, attr):
            setattr(args, attr, default_value)
        
    return args

def benchmark_model(model_class, config_path, model_name, num_runs=20):
    """Benchmark a model's forward pass performance"""
    
    print(f"\n=== Benchmarking {model_name} ===")
    
    try:
        # Load config
        args = create_args_from_config(config_path)
        
        # Create model
        model = model_class(args)
        model.eval()
        
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create test input - use the correct feature dimensions for each model
        batch_size = 8  # Match training batch size
        seq_len = args.seq_len
        
        if model_name == "Enhanced SOTA PGAT":
            # Enhanced model handles feature separation internally, so pass all features
            total_features = 12  # From dataset
            wave_window = torch.randn(batch_size, seq_len, total_features)
            target_window = torch.randn(batch_size, args.label_len + args.pred_len, total_features)
        else:
            # Base model expects only enc_in features
            enc_in = getattr(args, 'enc_in', 7)
            wave_window = torch.randn(batch_size, seq_len, enc_in)
            target_window = torch.randn(batch_size, args.label_len + args.pred_len, enc_in)
        
        print(f"   Input shapes: wave={wave_window.shape}, target={target_window.shape}")
        
        # Warmup runs
        for _ in range(3):
            with torch.no_grad():
                _ = model(wave_window, target_window)
        
        # Benchmark runs
        times = []
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(wave_window, target_window)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   Average forward time: {avg_time:.4f}s")
        print(f"   Min/Max forward time: {min_time:.4f}s / {max_time:.4f}s")
        
        if isinstance(output, tuple):
            print(f"   Output shapes: {[o.shape for o in output]}")
        else:
            print(f"   Output shape: {output.shape}")
        
        return avg_time, sum(p.numel() for p in model.parameters())
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_models():
    """Compare Enhanced PGAT vs Base PGAT performance"""
    
    print("=== Model Performance Comparison ===")
    
    # Test Enhanced PGAT
    try:
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
        enhanced_time, enhanced_params = benchmark_model(
            Enhanced_SOTA_PGAT, 
            "configs/enhanced_sota_pgat_synthetic.yaml",
            "Enhanced SOTA PGAT"
        )
    except Exception as e:
        print(f"Enhanced PGAT failed: {e}")
        enhanced_time, enhanced_params = None, None
    
    # Test Base PGAT
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        base_time, base_params = benchmark_model(
            SOTA_Temporal_PGAT,
            "configs/enhanced_sota_pgat_synthetic.yaml",  # Use same config for fair comparison
            "Base SOTA PGAT"
        )
    except Exception as e:
        print(f"Base PGAT failed: {e}")
        base_time, base_params = None, None
    
    # Compare results
    print(f"\n=== Comparison Results ===")
    if enhanced_time and base_time:
        time_ratio = enhanced_time / base_time
        param_ratio = enhanced_params / base_params
        
        print(f"Enhanced PGAT: {enhanced_time:.4f}s, {enhanced_params:,} params")
        print(f"Base PGAT:     {base_time:.4f}s, {base_params:,} params")
        print(f"Time ratio:    {time_ratio:.2f}x (Enhanced vs Base)")
        print(f"Param ratio:   {param_ratio:.2f}x (Enhanced vs Base)")
        
        if time_ratio < param_ratio:
            print("⚠️  WARNING: Enhanced model is faster than expected given parameter increase!")
        elif time_ratio > param_ratio * 2:
            print("⚠️  WARNING: Enhanced model is much slower than expected!")
        else:
            print("✅ Performance scaling looks reasonable")
    else:
        print("Could not complete comparison due to errors")

if __name__ == "__main__":
    compare_models()