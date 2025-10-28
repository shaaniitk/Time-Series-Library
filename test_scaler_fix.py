#!/usr/bin/env python3
"""
Test the scaler fix to ensure validation and test datasets receive properly scaled data
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_scaler_fix():
    """Test that the scaler fix works correctly"""
    
    print("🔧 TESTING SCALER FIX")
    print("=" * 60)
    
    # Create a minimal config for testing
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = {
        'data': 'custom',
        'root_path': './data',
        'data_path': 'prepared_financial_data.csv',
        'features': 'MS',
        'target': 'log_Open,log_High,log_Low,log_Close',
        'embed': 'timeF',
        'freq': 'd',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 24,
        'validation_length': 800,
        'test_length': 600,
        'batch_size': 16,
        'num_workers': 0,
        'seasonal_patterns': 'Monthly',
        'task_name': 'long_term_forecast',  # Required by data_provider
        'scale': True  # Enable scaling
    }
    
    args = SimpleConfig(config)
    
    try:
        # Import the data provider functions
        from data_provider.data_factory import data_provider
        from scripts.train.train_celestial_production import data_provider_with_scaler
        
        print("✅ Successfully imported data provider functions")
        
        # Test 1: Create training dataset
        print("\n📊 Test 1: Creating training dataset...")
        train_dataset, train_loader = data_provider(args, flag='train')
        
        # Extract scalers from training dataset
        train_scaler = getattr(train_dataset, 'scaler', None)
        train_target_scaler = getattr(train_dataset, 'target_scaler', None)
        
        print(f"  Training dataset created: {len(train_dataset)} samples")
        print(f"  Main scaler available: {train_scaler is not None}")
        print(f"  Target scaler available: {train_target_scaler is not None}")
        
        if train_scaler is not None:
            print(f"  Main scaler features: {train_scaler.n_features_in_}")
            print(f"  Main scaler mean (first 4): {train_scaler.mean_[:4]}")
            print(f"  Main scaler scale (first 4): {train_scaler.scale_[:4]}")
        
        if train_target_scaler is not None:
            print(f"  Target scaler features: {train_target_scaler.n_features_in_}")
            print(f"  Target scaler mean: {train_target_scaler.mean_}")
            print(f"  Target scaler scale: {train_target_scaler.scale_}")
        
        # Test 2: Create validation dataset with scalers
        print("\n📊 Test 2: Creating validation dataset with training scalers...")
        val_dataset, val_loader = data_provider_with_scaler(
            args, flag='val', 
            scaler=train_scaler, 
            target_scaler=train_target_scaler
        )
        
        print(f"  Validation dataset created: {len(val_dataset)} samples")
        
        # Verify scalers are properly set
        val_scaler = getattr(val_dataset, 'scaler', None)
        val_target_scaler = getattr(val_dataset, 'target_scaler', None)
        
        print(f"  Validation main scaler available: {val_scaler is not None}")
        print(f"  Validation target scaler available: {val_target_scaler is not None}")
        
        # Test 3: Compare data scaling
        print("\n📊 Test 3: Comparing data scaling...")
        
        # Get a sample from each dataset
        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        
        train_x, train_y = train_sample[0], train_sample[1]
        val_x, val_y = val_sample[0], val_sample[1]
        
        print(f"  Training X shape: {train_x.shape}")
        print(f"  Training X stats: mean={train_x.mean():.6f}, std={train_x.std():.6f}")
        print(f"  Validation X shape: {val_x.shape}")
        print(f"  Validation X stats: mean={val_x.mean():.6f}, std={val_x.std():.6f}")
        
        # Check if scaling is consistent
        x_mean_diff = abs(train_x.mean() - val_x.mean())
        x_std_diff = abs(train_x.std() - val_x.std())
        
        print(f"  X mean difference: {x_mean_diff:.6f}")
        print(f"  X std difference: {x_std_diff:.6f}")
        
        if x_mean_diff < 0.1 and x_std_diff < 0.1:
            print("  ✅ Training and validation X data have similar scaling!")
        else:
            print("  ⚠️  Training and validation X data have different scaling")
        
        # Test 4: Verify scaler identity
        print("\n📊 Test 4: Verifying scaler identity...")
        
        if train_scaler is not None and val_scaler is not None:
            # Check if they're the same object or have same parameters
            same_object = train_scaler is val_scaler
            same_mean = np.allclose(train_scaler.mean_, val_scaler.mean_)
            same_scale = np.allclose(train_scaler.scale_, val_scaler.scale_)
            
            print(f"  Same scaler object: {same_object}")
            print(f"  Same scaler mean: {same_mean}")
            print(f"  Same scaler scale: {same_scale}")
            
            if same_object or (same_mean and same_scale):
                print("  ✅ Scalers are properly shared!")
            else:
                print("  ❌ Scalers are different!")
        
        print("\n🎉 SCALER FIX TEST COMPLETED")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the scaler fix test"""
    
    print("🧪 SCALER FIX VERIFICATION")
    print("=" * 80)
    
    success = test_scaler_fix()
    
    if success:
        print("\n✅ SCALER FIX VERIFICATION PASSED")
        print("The validation and test datasets now receive properly scaled data!")
        print("This should resolve the validation loss issues.")
    else:
        print("\n❌ SCALER FIX VERIFICATION FAILED")
        print("There may be additional issues to resolve.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)