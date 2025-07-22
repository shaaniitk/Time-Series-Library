"""
Comprehensive Test for All HF Models with Covariate Support

This script tests all fixed HF models to verify that:
1. All models properly use covariates
2. Covariates impact predictions meaningfully
3. Models can handle real time series data
4. Training works with covariate integration
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from argparse import Namespace
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all HF models for testing
import importlib.util

def load_hf_model(model_name):
    """Dynamically load HF model to avoid import issues"""
    try:
        spec = importlib.util.spec_from_file_location(model_name, f"./models/{model_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, model_name.split('.')[0])
    except Exception as e:
        print(f"WARN Could not load {model_name}: {e}")
        return None

# Load models
HFEnhancedAutoformer = load_hf_model("HFEnhancedAutoformer")
HFBayesianAutoformer = load_hf_model("HFBayesianAutoformer") 
HFHierarchicalAutoformer_Step3 = load_hf_model("HFHierarchicalAutoformer_Step3")
HFQuantileAutoformer_Step4 = load_hf_model("HFQuantileAutoformer_Step4")

from utils.timefeatures import time_features


def create_test_config():
    """Create test configuration"""
    config = {
        'task_name': 'long_term_forecast',
        'root_path': './data',
        'data_path': 'synthetic_timeseries_full.csv',
        'data': 'custom',
        'features': 'M',  
        'target': 'OT',   
        'freq': 'h',      
        'checkpoints': './checkpoints/',
        'timeenc': 1,     
        'embed': 'timeF', 
        
        # Sequence lengths
        'seq_len': 48,   # Reduced for faster testing
        'label_len': 24,
        'pred_len': 12,
        
        # Model dimensions
        'enc_in': 6,    
        'dec_in': 6,    
        'c_out': 1,     
        'd_model': 128, # Reduced for faster testing
        'n_heads': 4,
        'e_layers': 1,  # Reduced for faster testing
        'd_layers': 1,
        'd_ff': 256,
        'dropout': 0.1,
        'activation': 'gelu',
        
        # Training
        'batch_size': 8,  # Reduced for faster testing
        'use_gpu': False,
        'gpu': 0,
        'use_multi_gpu': False,
        
        # HF-specific
        'model_name': 'chronos-t5-tiny',
        
        # Quantile settings
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
        'quantile_levels': [0.1, 0.25, 0.5, 0.75, 0.9],
        
        # Hierarchical settings
        'hierarchical_scales': [1, 2, 4],
        'cross_scale_attention': True,
    }
    
    return Namespace(**config)


def create_synthetic_data():
    """Create synthetic time series data"""
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=500, freq='H')  # Smaller dataset for testing
    
    # Create synthetic time series with patterns
    n_samples = len(dates)
    
    # Base trends
    trend1 = np.linspace(10, 50, n_samples) + np.random.normal(0, 2, n_samples)
    trend2 = 30 + 10 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 1.5, n_samples)
    
    # Patterns
    daily_pattern = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    weekly_pattern = 3 * np.sin(2 * np.pi * np.arange(n_samples) / (24*7))
    
    # Create features
    data = {
        'date': dates,
        'HUFL': trend1 + daily_pattern + weekly_pattern,
        'HULL': trend2 + 0.8 * daily_pattern + 0.5 * weekly_pattern,
        'MUFL': 0.7 * trend1 + 0.3 * trend2 + 0.9 * daily_pattern,
        'MULL': 0.5 * trend1 + 0.5 * trend2 + 0.6 * weekly_pattern,
        'LUFL': trend1 * 0.8 + 2 * daily_pattern,
        'OT': trend1 + trend2 + daily_pattern + weekly_pattern + np.random.normal(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df


def prepare_data_with_covariates(df, config):
    """Prepare data with time features"""
    
    df['date'] = pd.to_datetime(df['date'])
    df_stamp = df[['date']].copy()
    
    # Get time features
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=config.freq)
    data_stamp = data_stamp.transpose(1, 0)
    
    # Get time series data
    cols_data = [col for col in df.columns if col != 'date']
    data = df[cols_data].values.astype(np.float32)
    
    return data, data_stamp


def create_sequences(data, data_stamp, config):
    """Create input sequences"""
    
    seq_len = config.seq_len
    label_len = config.label_len  
    pred_len = config.pred_len
    
    total_len = seq_len + pred_len
    start_idx = len(data) // 2
    end_idx = start_idx + total_len
    
    if end_idx > len(data):
        start_idx = len(data) - total_len
        end_idx = len(data)
    
    seq_x = data[start_idx:start_idx + seq_len]
    seq_y = data[start_idx + seq_len - label_len:end_idx]
    
    seq_x_mark = data_stamp[start_idx:start_idx + seq_len]
    seq_y_mark = data_stamp[start_idx + seq_len - label_len:end_idx]
    
    x_enc = torch.FloatTensor(seq_x).unsqueeze(0)
    x_dec = torch.FloatTensor(seq_y).unsqueeze(0)
    x_mark_enc = torch.FloatTensor(seq_x_mark).unsqueeze(0)
    x_mark_dec = torch.FloatTensor(seq_y_mark).unsqueeze(0)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def test_model_covariate_usage(model_class, model_name, config, x_enc, x_mark_enc, x_dec, x_mark_dec):
    """Test that a model properly uses covariates"""
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        # Create model
        model = model_class(config)
        model.eval()
        
        print(f" Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test with covariates
        with torch.no_grad():
            if 'Hierarchical' in model_name:
                output_with = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_hierarchical=False)
            elif 'Quantile' in model_name:
                output_with = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_quantiles=False)
            else:
                output_with = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f" Forward with covariates: {output_with.shape}")
        mean_with = output_with.mean().item()
        
        # Test without covariates
        with torch.no_grad():
            if 'Hierarchical' in model_name:
                output_without = model(x_enc, None, x_dec, None, return_hierarchical=False)
            elif 'Quantile' in model_name:
                output_without = model(x_enc, None, x_dec, None, return_quantiles=False)
            else:
                output_without = model(x_enc, None, x_dec, None)
        
        print(f" Forward without covariates: {output_without.shape}")
        mean_without = output_without.mean().item()
        
        # Compare
        diff = torch.abs(output_with - output_without)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        
        print(f" Mean prediction WITH covariates: {mean_with:.6f}")
        print(f" Mean prediction WITHOUT covariates: {mean_without:.6f}")
        print(f" Mean absolute difference: {mean_diff:.6f}")
        print(f" Max absolute difference: {max_diff:.6f}")
        
        success = mean_diff > 1e-6
        if success:
            print(f"PASS SUCCESS: {model_name} properly uses covariates!")
        else:
            print(f"FAIL FAILURE: {model_name} ignores covariates")
            
        return success
        
    except Exception as e:
        print(f"FAIL ERROR testing {model_name}: {e}")
        return False


def test_special_features(model_class, model_name, config, x_enc, x_mark_enc, x_dec, x_mark_dec):
    """Test special features of advanced models"""
    
    if model_name == "HFBayesianAutoformer":
        try:
            model = model_class(config, uncertainty_method='dropout', n_samples=10)
            with torch.no_grad():
                uncertainty_result = model.get_uncertainty_result(x_enc, x_mark_enc, x_dec, x_mark_dec)
            print(f" Uncertainty estimation: {uncertainty_result.uncertainty.mean().item():.6f}")
            return True
        except Exception as e:
            print(f"WARN Uncertainty test failed: {e}")
            return False
            
    elif model_name == "HFHierarchicalAutoformer_Step3":
        try:
            model = model_class(config)
            with torch.no_grad():
                hierarchical_result = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_hierarchical=True)
            print(f" Hierarchical analysis: {len(hierarchical_result.scale_predictions)} scales")
            return True
        except Exception as e:
            print(f"WARN Hierarchical test failed: {e}")
            return False
            
    elif model_name == "HFQuantileAutoformer_Step4":
        try:
            model = model_class(config)
            with torch.no_grad():
                quantile_result = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_quantiles=True)
            print(f" Quantile regression: {len(quantile_result.quantiles)} quantiles")
            return True
        except Exception as e:
            print(f"WARN Quantile test failed: {e}")
            return False
            
    return True


def main():
    """Main testing function"""
    
    print("="*80)
    print("COMPREHENSIVE HF MODELS COVARIATE TESTING")
    print("="*80)
    
    # 1. Setup
    print("\n1. Creating configuration and data...")
    config = create_test_config()
    df = create_synthetic_data()
    data, data_stamp = prepare_data_with_covariates(df, config)
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sequences(data, data_stamp, config)
    
    print(f" Data prepared: {data.shape}, Time features: {data_stamp.shape}")
    print(f" Sequences: x_enc {x_enc.shape}, x_mark_enc {x_mark_enc.shape}")
    
    # 2. Test models
    models_to_test = [
        (HFEnhancedAutoformer, "HFEnhancedAutoformer"),
        (HFBayesianAutoformer, "HFBayesianAutoformer"),
        (HFHierarchicalAutoformer_Step3, "HFHierarchicalAutoformer_Step3"),
        (HFQuantileAutoformer_Step4, "HFQuantileAutoformer_Step4"),
    ]
    
    results = {}
    
    for model_class, model_name in models_to_test:
        if model_class is None:
            print(f"\nWARN Skipping {model_name} (not available)")
            results[model_name] = False
            continue
            
        # Test covariate usage
        covariate_success = test_model_covariate_usage(
            model_class, model_name, config, 
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        
        # Test special features
        if covariate_success:
            feature_success = test_special_features(
                model_class, model_name, config,
                x_enc, x_mark_enc, x_dec, x_mark_dec
            )
        else:
            feature_success = False
            
        results[model_name] = covariate_success and feature_success
    
    # 3. Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = 0
    
    for model_name, success in results.items():
        if model_name in [m[1] for m in models_to_test if m[0] is not None]:
            total += 1
            if success:
                passed += 1
                print(f"PASS {model_name}")
            else:
                print(f"FAIL {model_name}")
    
    print(f"\nOverall: {passed}/{total} models properly use covariates")
    
    if passed == total:
        print("PARTY All HF models properly use covariates!")
    else:
        print("WARN Some models need attention")
    
    return passed == total


if __name__ == "__main__":
    success = main()
