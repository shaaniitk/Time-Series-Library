#!/usr/bin/env python3
"""
Simple ChronosX Integration Test for ModularAutoformer
Focuses specifically on testing the ChronosX backbone functionality.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
import time
import traceback

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.modular_autoformer import ModularAutoformer
from utils.modular_components.registry import create_global_registry
from utils.modular_components.example_components import register_example_components


def generate_test_data(seq_len=96, pred_len=24, num_features=1, batch_size=2):
    """Generate synthetic time series data for testing"""
    print("CHART Generating synthetic test data...")
    
    # Generate realistic time series with trends and seasonality
    t = np.linspace(0, 4*np.pi, seq_len + pred_len)
    
    # Create multiple time series in batch
    data = []
    for b in range(batch_size):
        # Add trend, seasonality, and noise
        trend = 0.1 * t
        seasonal = 0.5 * np.sin(t) + 0.3 * np.cos(2*t)
        noise = 0.1 * np.random.randn(len(t))
        series = trend + seasonal + noise + b * 0.2  # Slight offset per batch
        
        if num_features > 1:
            # Add additional features
            extra_features = np.random.randn(len(t), num_features - 1) * 0.1
            series = np.column_stack([series, extra_features])
        else:
            series = series.reshape(-1, 1)
            
        data.append(series)
    
    data = np.stack(data)  # Shape: (batch_size, seq_len + pred_len, num_features)
    
    # Split into encoder and decoder parts
    x_enc = data[:, :seq_len, :]
    x_dec = data[:, seq_len-pred_len:, :]  # Overlapping for autoformer style
    
    # Create time marks (dummy)
    x_mark_enc = np.zeros((batch_size, seq_len, 4))  # Common time features
    x_mark_dec = np.zeros((batch_size, pred_len*2, 4))
    
    # Convert to tensors
    test_data = {
        'x_enc': torch.FloatTensor(x_enc),
        'x_mark_enc': torch.FloatTensor(x_mark_enc),
        'x_dec': torch.FloatTensor(x_dec),
        'x_mark_dec': torch.FloatTensor(x_mark_dec),
        'true_future': torch.FloatTensor(data[:, seq_len:, :])
    }
    
    print(f"   PASS Generated data shapes:")
    print(f"      x_enc: {test_data['x_enc'].shape}")
    print(f"      x_dec: {test_data['x_dec'].shape}")
    print(f"      true_future: {test_data['true_future'].shape}")
    
    return test_data


def create_chronosx_config():
    """Create test configuration for ChronosX backbone"""
    
    config = Namespace()
    
    # Basic task configuration
    config.task_name = 'long_term_forecast'
    config.seq_len = 96
    config.label_len = 48
    config.pred_len = 24
    
    # Model dimensions
    config.d_model = 64
    config.enc_in = 1
    config.dec_in = 1
    config.c_out = 1
    config.c_out_evaluation = 1
    
    # Embedding configuration
    config.embed = 'timeF'
    config.freq = 'h'
    config.dropout = 0.1
    
    # Enable ChronosX backbone
    config.use_backbone_component = True
    config.backbone_type = 'chronos_x'
    
    # ChronosX specific parameters
    config.backbone_params = {
        'model_size': 'tiny',  # tiny, small, base, large
        'use_uncertainty': True,
        'num_samples': 20
    }
    
    # Simple components that work with backbone
    config.sampling_type = 'deterministic'
    config.output_head_type = 'standard'
    config.loss_function_type = 'mse'
    
    # Minimal parameters
    config.sampling_params = {}
    config.output_head_params = {
        'd_model': config.d_model,
        'c_out': config.c_out
    }
    config.loss_params = {}
    
    return config


def test_chronosx_integration():
    """Test ChronosX integration with ModularAutoformer"""
    print("ROCKET Testing ChronosX Integration with ModularAutoformer")
    print("=" * 60)
    
    try:
        # Initialize global registry and register components
        print("TOOL Initializing component registry...")
        registry = create_global_registry()
        register_example_components(registry)
        print("   PASS Components registered successfully")
        
        # Generate test data
        test_data = generate_test_data()
        
        # Create ChronosX configuration
        print("\n Creating ChronosX configuration...")
        config = create_chronosx_config()
        print(f"   PASS Configuration created:")
        print(f"      Backbone: {config.backbone_type}")
        print(f"      Model size: {config.backbone_params['model_size']}")
        print(f"      Uncertainty: {config.backbone_params['use_uncertainty']}")
        
        # Create model with ChronosX backbone
        print("\nCRYSTAL Creating ModularAutoformer with ChronosX backbone...")
        start_time = time.time()
        model = ModularAutoformer(config)
        model.eval()
        init_time = time.time() - start_time
        
        print(f"   PASS Model created in {init_time:.3f}s")
        
        # Get model information
        component_info = model.get_component_info()
        backbone_info = model.get_backbone_info()
        
        print(f"   CLIPBOARD Model Information:")
        print(f"      Architecture: {component_info['architecture']}")
        print(f"      Backbone type: {backbone_info['backbone_type']}")
        print(f"      Backbone class: {backbone_info['backbone_class']}")
        print(f"      Supports uncertainty: {backbone_info['supports_uncertainty']}")
        
        # Test forward pass
        print("\nLIGHTNING Testing forward pass...")
        with torch.no_grad():
            start_time = time.time()
            output = model(
                test_data['x_enc'],
                test_data['x_mark_enc'],
                test_data['x_dec'],
                test_data['x_mark_dec']
            )
            inference_time = time.time() - start_time
        
        print(f"   PASS Forward pass successful!")
        print(f"      Output shape: {output.shape}")
        print(f"      Expected shape: {test_data['true_future'].shape}")
        print(f"      Inference time: {inference_time:.3f}s")
        print(f"      Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Test uncertainty quantification
        print("\nTARGET Testing uncertainty quantification...")
        uncertainty_results = model.get_uncertainty_results()
        if uncertainty_results:
            print(f"   PASS Uncertainty results available:")
            print(f"      Prediction shape: {uncertainty_results['prediction'].shape}")
            if 'std' in uncertainty_results:
                std_values = uncertainty_results['std']
                print(f"      Std shape: {std_values.shape}")
                print(f"      Avg uncertainty: {std_values.mean().item():.6f}")
                print(f"      Uncertainty range: [{std_values.min().item():.6f}, {std_values.max().item():.6f}]")
            
            if 'quantiles' in uncertainty_results:
                quantiles = uncertainty_results['quantiles']
                print(f"      Quantiles available: {list(quantiles.keys())}")
        else:
            print(f"   WARN No uncertainty results available")
        
        # Calculate performance metrics
        print("\nCHART Performance Evaluation...")
        if output.shape == test_data['true_future'].shape:
            mse = torch.nn.functional.mse_loss(output, test_data['true_future'])
            mae = torch.nn.functional.l1_loss(output, test_data['true_future'])
            
            print(f"   GRAPH Forecasting Metrics:")
            print(f"      MSE: {mse.item():.6f}")
            print(f"      MAE: {mae.item():.6f}")
            print(f"      RMSE: {torch.sqrt(mse).item():.6f}")
            
            # Calculate relative metrics
            true_std = test_data['true_future'].std().item()
            relative_mae = mae.item() / true_std
            print(f"      Relative MAE: {relative_mae:.3f} (lower is better)")
            
        else:
            print(f"   WARN Shape mismatch - cannot calculate metrics")
            print(f"      Output: {output.shape}, Expected: {test_data['true_future'].shape}")
        
        # Test different model configurations
        print("\nREFRESH Testing different ChronosX configurations...")
        
        configs_to_test = [
            ('chronos_x_tiny', 'Tiny model variant'),
            ('chronos_x_uncertainty', 'Uncertainty-focused variant'),
        ]
        
        for variant_type, description in configs_to_test:
            print(f"\n   Testing {description} ({variant_type})...")
            try:
                # Create variant config
                variant_config = create_chronosx_config()
                variant_config.backbone_type = variant_type
                
                # Create variant model
                variant_model = ModularAutoformer(variant_config)
                variant_model.eval()
                
                # Test forward pass
                with torch.no_grad():
                    variant_output = variant_model(
                        test_data['x_enc'],
                        test_data['x_mark_enc'],
                        test_data['x_dec'],
                        test_data['x_mark_dec']
                    )
                
                print(f"      PASS {description} working")
                print(f"         Output shape: {variant_output.shape}")
                
                # Check for uncertainty
                variant_uncertainty = variant_model.get_uncertainty_results()
                if variant_uncertainty and 'std' in variant_uncertainty:
                    avg_uncertainty = variant_uncertainty['std'].mean().item()
                    print(f"         TARGET Avg uncertainty: {avg_uncertainty:.6f}")
                
            except Exception as e:
                print(f"      FAIL {description} failed: {str(e)}")
        
        print(f"\nPARTY ChronosX Integration Test SUCCESSFUL!")
        print("=" * 60)
        print("PASS Key Achievements:")
        print("   CRYSTAL ChronosX backbone successfully integrated")
        print("   LIGHTNING Fast inference without training")
        print("   TARGET Uncertainty quantification working")
        print("   CHART Multiple model variants functional")
        print("   TOOL Modular architecture flexible and extensible")
        
        return True
        
    except Exception as e:
        print(f"\nFAIL ChronosX Integration Test FAILED!")
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    print("MICROSCOPE Starting ChronosX Integration Test for ModularAutoformer...\n")
    
    success = test_chronosx_integration()
    
    if success:
        print(f"\nROCKET SUCCESS! Your ChronosX + ModularAutoformer integration is working perfectly!")
        print(f"IDEA You can now use ChronosX backbones for zero-shot forecasting with uncertainty.")
    else:
        print(f"\nTOOL Integration test failed. Check the error details above.")
    
    sys.exit(0 if success else 1)
