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

def test_enhanced_execution():
    """Test if Enhanced PGAT components are actually executing"""
    
    print("=== Testing Enhanced PGAT Component Execution ===")
    
    try:
        # Load Enhanced PGAT
        enhanced_config = "configs/enhanced_sota_pgat_synthetic.yaml"
        enhanced_args = create_args_from_config(enhanced_config)
        
        # Import and create the model
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT as EnhancedModel
        
        print(f"1. Creating Enhanced PGAT model...")
        enhanced_model = EnhancedModel(enhanced_args)
        enhanced_model.eval()
        
        print(f"   Model parameters: {sum(p.numel() for p in enhanced_model.parameters()):,}")
        
        # Create test input - data has 12 total features (7 waves + 3 targets + 2 covariates)
        batch_size = 2
        seq_len = enhanced_args.seq_len
        total_features = 12  # From the actual dataset
        
        wave_window = torch.randn(batch_size, seq_len, total_features)
        target_window = torch.randn(batch_size, enhanced_args.label_len + enhanced_args.pred_len, total_features)
        
        print(f"2. Testing forward pass...")
        print(f"   Input shapes: wave={wave_window.shape}, target={target_window.shape}")
        
        # Time the forward pass
        start_time = time.time()
        
        with torch.no_grad():
            output = enhanced_model(wave_window, target_window)
        
        forward_time = time.time() - start_time
        
        print(f"   Forward pass time: {forward_time:.4f}s")
        
        if isinstance(output, tuple):
            print(f"   Output shapes: {[o.shape for o in output]}")
        else:
            print(f"   Output shape: {output.shape}")
        
        # Check if enhanced components are active
        print(f"3. Checking enhanced component activity...")
        
        # Check patching composers
        has_patching = hasattr(enhanced_model, 'wave_patching_composer') and enhanced_model.wave_patching_composer is not None
        print(f"   Multi-scale patching active: {has_patching}")
        
        # Check hierarchical mapper
        has_hierarchical = hasattr(enhanced_model, 'use_hierarchical_mapper') and enhanced_model.use_hierarchical_mapper
        print(f"   Hierarchical mapping active: {has_hierarchical}")
        
        # Check stochastic learner
        has_stochastic = hasattr(enhanced_model, 'stochastic_learner') and enhanced_model.stochastic_learner is not None
        print(f"   Stochastic graph learning active: {has_stochastic}")
        
        # Check gated combiner
        has_gated = hasattr(enhanced_model, 'graph_combiner') and enhanced_model.graph_combiner is not None
        print(f"   Gated graph combination active: {has_gated}")
        
        # Check mixture density decoder
        has_mixture = hasattr(enhanced_model, 'decoder') and 'Mixture' in str(type(enhanced_model.decoder))
        print(f"   Mixture density decoder active: {has_mixture}")
        
        # Test with profiling to see computational load
        print(f"4. Profiling computational load...")
        
        # Multiple forward passes to get average
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = enhanced_model(wave_window, target_window)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"   Average forward pass time ({num_runs} runs): {avg_time:.4f}s")
        
        # Compare with base model if possible
        print(f"5. Comparing with base PGAT...")
        
        try:
            from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT as BaseModel
            base_model = BaseModel(enhanced_args)  # Use same config
            base_model.eval()
            
            base_params = sum(p.numel() for p in base_model.parameters())
            print(f"   Base model parameters: {base_params:,}")
            
            # Time base model
            base_times = []
            for i in range(num_runs):
                start = time.time()
                with torch.no_grad():
                    _ = base_model(wave_window, target_window)
                base_times.append(time.time() - start)
            
            base_avg_time = sum(base_times) / len(base_times)
            print(f"   Base model avg time: {base_avg_time:.4f}s")
            print(f"   Enhanced vs Base time ratio: {avg_time / base_avg_time:.2f}x")
            print(f"   Enhanced vs Base param ratio: {(sum(p.numel() for p in enhanced_model.parameters()) / base_params):.2f}x")
            
        except Exception as e:
            print(f"   Base model comparison failed: {e}")
        
    except Exception as e:
        print(f"Enhanced model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_execution()