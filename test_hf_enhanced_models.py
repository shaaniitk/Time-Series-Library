"""
Test script for HF Enhanced Models

This script validates that our HF enhanced models work correctly with:
1. Different loss functions from existing infrastructure
2. Bayesian uncertainty quantification
3. Hierarchical multi-scale processing
4. Quantile regression
5. All combinations of features

Run this to verify the implementation works before using in experiments.
"""

import sys
import os
import torch
import numpy as np
from argparse import Namespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.HFAdvancedFactory import (
    create_hf_bayesian_model,
    create_hf_hierarchical_model,
    create_hf_quantile_model,
    create_hf_full_model,
    create_hf_model_from_config
)
from models.HFEnhancedAutoformer import HFEnhancedAutoformer
from utils.losses import get_loss_function
from utils.bayesian_losses import create_bayesian_loss
from utils.logger import logger


def create_test_config():
    """Create test configuration"""
    config = {
        # Data dimensions
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 24,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'c_out_evaluation': 7,
        
        # Model architecture
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'dropout': 0.1,
        'activation': 'gelu',
        
        # Loss configuration
        'loss_function': 'mse',
        
        # Bayesian configuration
        'uncertainty_method': 'bayesian',
        'n_samples': 5,  # Reduced for testing
        'kl_weight': 1e-5,
        'bayesian_layers': ['output_projection'],
        
        # Hierarchical configuration
        'hierarchy_levels': [1, 2, 4],
        'aggregation_method': 'adaptive',
        'level_weights_learnable': True,
        
        # Wavelet configuration
        'wavelet_type': 'db4',
        'decomposition_levels': 3,
        'reconstruction_method': 'learnable',
        
        # Quantile configuration
        'quantile_levels': [0.1, 0.5, 0.9],
        
        # Device configuration
        'device': 'cpu'  # Use CPU for testing
    }
    
    return Namespace(**config)


def create_test_data(config):
    """Create test data"""
    batch_size = 2
    
    # Encoder input: [batch_size, seq_len, enc_in]
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    
    # Encoder time features: [batch_size, seq_len, time_features]
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # Common time features
    
    # Decoder input: [batch_size, label_len + pred_len, dec_in]
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
    
    # Decoder time features: [batch_size, label_len + pred_len, time_features]
    x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
    
    # Target for loss computation: [batch_size, pred_len, c_out]
    targets = torch.randn(batch_size, config.pred_len, config.c_out)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec, targets


def test_standard_hf_model():
    """Test standard HF model (baseline)"""
    print("Testing Standard HF Model...")
    
    config = create_test_config()
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = create_test_data(config)
    
    try:
        model = HFEnhancedAutoformer(config)
        print(f" Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f" Forward pass: {output.shape}")
        
        # Loss computation
        loss_fn = get_loss_function('mse', config)
        loss = loss_fn(output, targets)
        print(f" Loss computation: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        print(" Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        return False


def test_bayesian_hf_model():
    """Test Bayesian HF model"""
    print("\nTesting Bayesian HF Model...")
    
    config = create_test_config()
    config.use_bayesian = True
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = create_test_data(config)
    
    try:
        model = create_hf_bayesian_model(config)
        info = model.get_model_info()
        print(f" Model created: {info['parameters']['total']:,} parameters")
        print(f"  - Backbone: {info['parameters']['backbone']:,}")
        print(f"  - Extensions: {info['parameters']['extensions']:,}")
        print(f"  - Capabilities: {list(info['capabilities'].keys())}")
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(output, dict):
            print(f" Uncertainty forward: prediction {output['prediction'].shape}")
            print(f"  - Uncertainty: {output['uncertainty'].shape}")
            print(f"  - Confidence intervals: {list(output['confidence_intervals'].keys())}")
            
            # Test loss computation
            total_loss, components = model.compute_loss(output, targets, x_enc)
            print(f" Loss computation: total={total_loss.item():.6f}")
            print(f"  - Components: {list(components.keys())}")
            
        else:
            print(f" Standard forward: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hierarchical_hf_model():
    """Test Hierarchical HF model"""
    print("\nTesting Hierarchical HF Model...")
    
    config = create_test_config()
    config.use_hierarchical = True
    config.use_wavelet = True
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = create_test_data(config)
    
    try:
        model = create_hf_hierarchical_model(config)
        info = model.get_model_info()
        print(f" Model created: {info['parameters']['total']:,} parameters")
        print(f"  - Extensions: {info['extensions']}")
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f" Forward pass: {output.shape}")
        
        # Loss computation
        total_loss, components = model.compute_loss(output, targets, x_enc)
        print(f" Loss computation: {total_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantile_hf_model():
    """Test Quantile HF model"""
    print("\nTesting Quantile HF Model...")
    
    config = create_test_config()
    config.use_quantile = True
    config.loss_function = 'pinball'
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = create_test_data(config)
    
    # Adjust target for quantile output
    n_quantiles = len(config.quantile_levels)
    quantile_targets = targets.repeat(1, 1, n_quantiles)  # Expand for quantiles
    
    try:
        model = create_hf_quantile_model(config)
        info = model.get_model_info()
        print(f" Model created: {info['parameters']['total']:,} parameters")
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        expected_shape = (targets.shape[0], targets.shape[1], targets.shape[2] * n_quantiles)
        print(f" Quantile forward: {output.shape} (expected: {expected_shape})")
        
        # Loss computation with quantile targets
        total_loss, components = model.compute_loss(output, quantile_targets, x_enc)
        print(f" Pinball loss computation: {total_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_hf_model():
    """Test Full HF model (all features)"""
    print("\nTesting Full HF Model (All Features)...")
    
    config = create_test_config()
    config.use_bayesian = True
    config.use_hierarchical = True
    config.use_wavelet = True
    config.use_quantile = True
    config.loss_function = 'pinball'
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = create_test_data(config)
    
    # Adjust target for quantile output
    n_quantiles = len(config.quantile_levels)
    quantile_targets = targets.repeat(1, 1, n_quantiles)
    
    try:
        model = create_hf_full_model(config)
        info = model.get_model_info()
        print(f" Model created: {info['parameters']['total']:,} parameters")
        print(f"  - Extensions: {info['extensions']}")
        print(f"  - All capabilities: {info['capabilities']}")
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(output, dict) and 'prediction' in output:
            expected_shape = (targets.shape[0], targets.shape[1], targets.shape[2] * n_quantiles)
            print(f" Full forward: prediction {output['prediction'].shape} (expected: {expected_shape})")
            print(f"  - With uncertainty: {output['uncertainty'].shape}")
            
            # Loss computation
            total_loss, components = model.compute_loss(output, quantile_targets, x_enc)
            print(f" Full loss computation: {total_loss.item():.6f}")
            print(f"  - Components: {list(components.keys())}")
            
        else:
            print(f" Output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test different loss functions with HF models"""
    print("\nTesting Different Loss Functions...")
    
    config = create_test_config()
    config.use_bayesian = True  # Use Bayesian for variety
    x_enc, x_mark_enc, x_dec, x_mark_dec, targets = create_test_data(config)
    
    loss_functions = ['mse', 'mae', 'huber', 'pinball']
    
    for loss_name in loss_functions:
        try:
            print(f"  Testing {loss_name} loss...")
            
            # Configure for loss
            test_config = create_test_config()
            test_config.use_bayesian = True
            test_config.loss_function = loss_name
            
            if loss_name == 'pinball':
                test_config.use_quantile = True
                test_targets = targets.repeat(1, 1, len(test_config.quantile_levels))
            else:
                test_targets = targets
                
            model = create_hf_bayesian_model(test_config)
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            total_loss, components = model.compute_loss(output, test_targets, x_enc)
            print(f"     {loss_name}: {total_loss.item():.6f}")
            
        except Exception as e:
            print(f"     {loss_name}: {e}")


def test_auto_model_creation():
    """Test automatic model creation from config"""
    print("\nTesting Auto Model Creation...")
    
    test_cases = [
        {'name': 'Auto Standard', 'config': {}},
        {'name': 'Auto Bayesian', 'config': {'use_bayesian': True}},
        {'name': 'Auto Hierarchical', 'config': {'use_hierarchical': True}},
        {'name': 'Auto Quantile', 'config': {'quantile_levels': [0.1, 0.5, 0.9]}},
        {'name': 'Auto Full', 'config': {'use_bayesian': True, 'use_hierarchical': True, 'quantile_levels': [0.1, 0.5, 0.9]}}
    ]
    
    for case in test_cases:
        try:
            print(f"  Testing {case['name']}...")
            
            config = create_test_config()
            for key, value in case['config'].items():
                setattr(config, key, value)
                
            model = create_hf_model_from_config(config, model_type='auto')
            print(f"     Created: {type(model).__name__}")
            
        except Exception as e:
            print(f"     {case['name']}: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("HF Enhanced Models Test Suite")
    print("=" * 60)
    
    tests = [
        test_standard_hf_model,
        test_bayesian_hf_model,
        test_hierarchical_hf_model,
        test_quantile_hf_model,
        test_full_hf_model,
        test_loss_functions,
        test_auto_model_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f" Test failed with exception: {e}")
            results.append(False)
            
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = " PASS" if result else " FAIL"
        print(f"{status} {test.__name__}")
        
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("PARTY All tests passed! HF Enhanced Models are ready to use.")
    else:
        print("WARN  Some tests failed. Check the output above for details.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
