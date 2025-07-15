"""
Test HFEnhancedAutoformer with Real Data and Covariates

This script tests the HFEnhancedAutoformer with actual time series data
and covariates to verify that:
1. Covariates are properly processed and used
2. Model works with real data dimensions
3. Forward pass produces correct outputs
4. Temporal embeddings are functioning
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import HFEnhancedAutoformer directly
import importlib.util
spec = importlib.util.spec_from_file_location("HFEnhancedAutoformer", "./models/HFEnhancedAutoformer.py")
hf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_module)
HFEnhancedAutoformer = hf_module.HFEnhancedAutoformer

from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')


def create_test_config():
    """Create test configuration matching synthetic data"""
    config = {
        'task_name': 'long_term_forecast',
        'root_path': './data',
        'data_path': 'synthetic_timeseries_full.csv',
        'data': 'custom',
        'features': 'M',  # Multivariate
        'target': 'OT',   # Target column
        'freq': 'h',      # Hourly frequency
        'checkpoints': './checkpoints/',
        'timeenc': 1,     # Use time features
        'embed': 'timeF', # Time feature embedding
        
        # Sequence lengths
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 24,
        
        # Model dimensions
        'enc_in': 6,    # Number of input features (actual data features)
        'dec_in': 6,    # Number of decoder input features  
        'c_out': 1,     # Number of output features (target only)
        'd_model': 256, # Model dimension
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1,
        'activation': 'gelu',
        
        # Training
        'batch_size': 32,
        'use_gpu': False,  # Use CPU for testing
        'gpu': 0,
        'use_multi_gpu': False,
    }
    
    return Namespace(**config)


def load_real_data():
    """Load real synthetic time series data"""
    
    # Try to load the synthetic data
    data_path = './synthetic_timeseries_full.csv'
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found, creating synthetic data...")
        return create_synthetic_data()
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded real data: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}, creating synthetic data...")
        return create_synthetic_data()


def create_synthetic_data():
    """Create synthetic data with time features"""
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=2000, freq='H')
    
    # Create synthetic time series with trend and seasonality
    n_samples = len(dates)
    
    # Base trends
    trend1 = np.linspace(10, 50, n_samples) + np.random.normal(0, 2, n_samples)
    trend2 = 30 + 10 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 1.5, n_samples)
    
    # Seasonal patterns (daily and weekly)
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
        'OT': trend1 + trend2 + daily_pattern + weekly_pattern + np.random.normal(0, 1, n_samples)  # Target
    }
    
    df = pd.DataFrame(data)
    print(f"✓ Created synthetic data: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def prepare_data_with_covariates(df, config):
    """Prepare data with time features (covariates)"""
    
    # Ensure we have a date column
    if 'date' not in df.columns:
        # Create synthetic dates
        df['date'] = pd.date_range('2020-01-01', periods=len(df), freq='H')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract time features using the library's time_features function
    df_stamp = df[['date']].copy()
    
    # Get time features (this creates the covariates)
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=config.freq)
    data_stamp = data_stamp.transpose(1, 0)  # Shape: (n_samples, n_time_features)
    
    # Get the actual time series data (exclude date column)
    cols_data = [col for col in df.columns if col != 'date']
    data = df[cols_data].values.astype(np.float32)
    
    print(f"✓ Time series data shape: {data.shape}")
    print(f"✓ Time features (covariates) shape: {data_stamp.shape}")
    print(f"✓ Time features: {data_stamp.shape[1]} features (hour, day, month, etc.)")
    
    return data, data_stamp


def create_sequences(data, data_stamp, config):
    """Create input sequences for the model"""
    
    seq_len = config.seq_len
    label_len = config.label_len
    pred_len = config.pred_len
    
    # Calculate total length needed
    total_len = seq_len + pred_len
    
    # Get a sample sequence from the middle of the data
    start_idx = len(data) // 2
    end_idx = start_idx + total_len
    
    if end_idx > len(data):
        start_idx = len(data) - total_len
        end_idx = len(data)
    
    # Extract sequences
    seq_x = data[start_idx:start_idx + seq_len]           # Encoder input
    seq_y = data[start_idx + seq_len - label_len:end_idx] # Decoder input (with label_len overlap)
    
    # Extract corresponding time features
    seq_x_mark = data_stamp[start_idx:start_idx + seq_len]
    seq_y_mark = data_stamp[start_idx + seq_len - label_len:end_idx]
    
    # Convert to tensors and add batch dimension
    x_enc = torch.FloatTensor(seq_x).unsqueeze(0)         # (1, seq_len, features)
    x_dec = torch.FloatTensor(seq_y).unsqueeze(0)         # (1, label_len + pred_len, features)
    x_mark_enc = torch.FloatTensor(seq_x_mark).unsqueeze(0)  # (1, seq_len, time_features)
    x_mark_dec = torch.FloatTensor(seq_y_mark).unsqueeze(0)  # (1, label_len + pred_len, time_features)
    
    print(f"✓ Created sequences:")
    print(f"  x_enc: {x_enc.shape} (encoder input)")
    print(f"  x_mark_enc: {x_mark_enc.shape} (encoder covariates)")
    print(f"  x_dec: {x_dec.shape} (decoder input)")
    print(f"  x_mark_dec: {x_mark_dec.shape} (decoder covariates)")
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def test_covariate_usage(model, x_enc, x_mark_enc, x_dec, x_mark_dec):
    """Test that covariates are actually being used"""
    
    print("\n" + "="*60)
    print("Testing Covariate Usage")
    print("="*60)
    
    # Test 1: Forward pass with covariates
    print("Test 1: Forward pass WITH covariates")
    model.eval()
    with torch.no_grad():
        output_with_covariates = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"✓ Output shape: {output_with_covariates.shape}")
        print(f"✓ Mean prediction: {output_with_covariates.mean().item():.4f}")
    
    # Test 2: Forward pass without covariates (should be different)
    print("\nTest 2: Forward pass WITHOUT covariates")
    with torch.no_grad():
        output_without_covariates = model(x_enc, None, x_dec, None)
        print(f"✓ Output shape: {output_without_covariates.shape}")
        print(f"✓ Mean prediction: {output_without_covariates.mean().item():.4f}")
    
    # Test 3: Compare outputs (they should be different if covariates are used)
    print("\nTest 3: Comparing outputs")
    diff = torch.abs(output_with_covariates - output_without_covariates)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"✓ Mean absolute difference: {mean_diff:.6f}")
    print(f"✓ Max absolute difference: {max_diff:.6f}")
    
    if mean_diff > 1e-6:
        print("✅ SUCCESS: Covariates are being used! (outputs are different)")
    else:
        print("❌ FAILURE: Covariates appear to be ignored (outputs are identical)")
    
    return mean_diff > 1e-6


def test_temporal_embedding(model, x_mark_enc):
    """Test that temporal embedding is working"""
    
    print("\n" + "="*60)
    print("Testing Temporal Embedding")
    print("="*60)
    
    # Test temporal embedding directly
    print(f"Input covariates shape: {x_mark_enc.shape}")
    print(f"Input covariates sample: {x_mark_enc[0, :3, :].numpy()}")  # First 3 timesteps
    
    # Get temporal embedding
    with torch.no_grad():
        temporal_emb = model.temporal_embedding(x_mark_enc)
        print(f"✓ Temporal embedding shape: {temporal_emb.shape}")
        print(f"✓ Temporal embedding mean: {temporal_emb.mean().item():.6f}")
        print(f"✓ Temporal embedding std: {temporal_emb.std().item():.6f}")
    
    # Check that different time features produce different embeddings
    # Create modified covariates (change hour feature)
    x_mark_modified = x_mark_enc.clone()
    x_mark_modified[:, :, 0] = (x_mark_modified[:, :, 0] + 0.5) % 1.0  # Modify hour feature
    
    with torch.no_grad():
        temporal_emb_modified = model.temporal_embedding(x_mark_modified)
    
    emb_diff = torch.abs(temporal_emb - temporal_emb_modified).mean().item()
    print(f"✓ Embedding difference with modified time: {emb_diff:.6f}")
    
    if emb_diff > 1e-6:
        print("✅ SUCCESS: Temporal embedding responds to time features!")
    else:
        print("❌ FAILURE: Temporal embedding may not be working properly")
    
    return emb_diff > 1e-6


def test_training_step(model, x_enc, x_mark_enc, x_dec, x_mark_dec):
    """Test a training step to ensure gradients flow properly"""
    
    print("\n" + "="*60)
    print("Testing Training Step")
    print("="*60)
    
    # Set to training mode
    model.train()
    
    # Create a simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Create dummy target (last pred_len timesteps of decoder input)
    target = x_dec[:, -model.configs.pred_len:, :model.configs.c_out]
    print(f"Target shape: {target.shape}")
    
    # Forward pass
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"✓ Forward pass successful: {output.shape}")
    
    # Compute loss
    loss = criterion(output, target)
    print(f"✓ Loss computed: {loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            if 'temporal_embedding' in name:
                print(f"✓ Temporal embedding gradient norm: {grad_norm:.6f}")
    
    print(f"✓ Total parameters with gradients: {param_count}")
    print(f"✓ Average gradient norm: {total_grad_norm/param_count:.6f}")
    
    # Optimizer step
    optimizer.step()
    print("✅ Training step completed successfully!")
    
    return True


def main():
    """Main test function"""
    
    print("="*80)
    print("Testing HFEnhancedAutoformer with Real Data and Covariates")
    print("="*80)
    
    # 1. Create configuration
    print("\n1. Creating configuration...")
    config = create_test_config()
    print(f"✓ Configuration created")
    
    # 2. Load real data
    print("\n2. Loading real data...")
    df = load_real_data()
    
    # 3. Prepare data with covariates
    print("\n3. Preparing data with time features...")
    data, data_stamp = prepare_data_with_covariates(df, config)
    
    # 4. Create sequences
    print("\n4. Creating input sequences...")
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sequences(data, data_stamp, config)
    
    # 5. Create model
    print("\n5. Creating HFEnhancedAutoformer...")
    try:
        model = HFEnhancedAutoformer(config)
        print(f"✓ Model created successfully")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # 6. Test covariate usage
    try:
        covariate_test_passed = test_covariate_usage(model, x_enc, x_mark_enc, x_dec, x_mark_dec)
    except Exception as e:
        print(f"❌ Covariate test failed: {e}")
        return False
    
    # 7. Test temporal embedding
    try:
        temporal_test_passed = test_temporal_embedding(model, x_mark_enc)
    except Exception as e:
        print(f"❌ Temporal embedding test failed: {e}")
        return False
    
    # 8. Test training step
    try:
        training_test_passed = test_training_step(model, x_enc, x_mark_enc, x_dec, x_mark_dec)
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    # Final results
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    tests = [
        ("Covariate Usage", covariate_test_passed),
        ("Temporal Embedding", temporal_test_passed), 
        ("Training Step", training_test_passed)
    ]
    
    passed_tests = sum(1 for _, passed in tests if passed)
    total_tests = len(tests)
    
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! HFEnhancedAutoformer properly uses covariates.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # 9. Visual test (if possible)
    try:
        print("\n9. Creating visualization...")
        create_prediction_plot(model, x_enc, x_mark_enc, x_dec, x_mark_dec, data, config)
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    
    return passed_tests == total_tests


def create_prediction_plot(model, x_enc, x_mark_enc, x_dec, x_mark_dec, data, config):
    """Create a plot showing the prediction"""
    
    model.eval()
    with torch.no_grad():
        prediction = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    # Get the actual data for comparison
    start_idx = len(data) // 2
    actual_input = data[start_idx:start_idx + config.seq_len, -1]  # Last feature (target)
    actual_future = data[start_idx + config.seq_len:start_idx + config.seq_len + config.pred_len, -1]
    
    # Convert prediction to numpy
    pred_np = prediction[0, :, 0].numpy()  # First batch, all timesteps, first output feature
    
    # Create time axis
    time_input = np.arange(len(actual_input))
    time_future = np.arange(len(actual_input), len(actual_input) + len(actual_future))
    time_pred = np.arange(len(actual_input), len(actual_input) + len(pred_np))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_input, actual_input, 'b-', label='Historical Data', linewidth=2)
    plt.plot(time_future, actual_future, 'g-', label='Actual Future', linewidth=2)
    plt.plot(time_pred, pred_np, 'r--', label='HF Prediction', linewidth=2)
    plt.axvline(x=len(actual_input), color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('HFEnhancedAutoformer Prediction with Covariates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('hf_prediction_test.png', dpi=150, bbox_inches='tight')
    print(f"✓ Prediction plot saved as 'hf_prediction_test.png'")
    plt.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
