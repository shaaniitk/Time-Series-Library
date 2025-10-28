#!/usr/bin/env python3
"""
Quick test to verify data loading works with updated configuration
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_data_loading():
    """Test data loading with updated configuration"""
    
    # Updated configuration with better data splits
    config = {
        'model': 'Celestial_Enhanced_PGAT',
        'data': 'custom',
        'root_path': './data',
        'data_path': 'prepared_financial_data.csv',
        'features': 'MS',
        'target': 'log_Open,log_High,log_Low,log_Close',
        'embed': 'timeF',
        'freq': 'd',
        
        # Sequence settings
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 24,
        
        # FIXED: Better data splits for 7109 total samples
        'validation_length': 500,  # ~7% for validation
        'test_length': 500,        # ~7% for testing
        # This leaves ~6109 samples for training
        
        # Model configuration
        'd_model': 128,  # Smaller for faster testing
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        
        # Input/Output dimensions
        'enc_in': 113,
        'dec_in': 113,
        'c_out': 4,
        
        # Training configuration
        'train_epochs': 1,  # Just test data loading
        'batch_size': 8,
        'learning_rate': 0.001,
        'patience': 5,
        
        # Required fields
        'task_name': 'long_term_forecast',
        'model_name': 'Celestial_Enhanced_PGAT',
        'data_name': 'custom',
        'checkpoints': './checkpoints/',
        'inverse': False,
        'cols': None,
        'num_workers': 0,
        'itr': 1,
        'train_only': False,
        'do_predict': False,
        'model_id': 'test_data_loading',
        
        # Celestial system
        'use_celestial_graph': True,
        'aggregate_waves_to_celestial': True,
        'num_celestial_bodies': 13,
        'target_wave_indices': [0, 1, 2, 3],
    }
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = SimpleConfig(config)
    
    try:
        from data_provider.data_factory import data_provider
        
        print("🔍 Testing data loading with updated configuration...")
        print(f"   Dataset: {args.data_path}")
        print(f"   Sequence length: {args.seq_len}")
        print(f"   Validation length: {args.validation_length}")
        print(f"   Test length: {args.test_length}")
        print(f"   Batch size: {args.batch_size}")
        
        # Test each data split
        for flag in ['train', 'val', 'test']:
            print(f"\n📊 Loading {flag} data...")
            
            dataset, loader = data_provider(args, flag=flag)
            
            dataset_size = len(dataset) if hasattr(dataset, '__len__') else 'unknown'
            loader_batches = len(loader)
            
            print(f"   ✅ {flag.upper()} - Dataset size: {dataset_size}, Batches: {loader_batches}")
            
            if loader_batches == 0:
                print(f"   ❌ ERROR: {flag} loader has 0 batches!")
                return False
            
            # Test loading one batch
            try:
                batch = next(iter(loader))
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                    print(f"   ✅ Batch shape: {batch_x.shape}")
                else:
                    print(f"   ⚠️  Unexpected batch format: {type(batch)}")
            except Exception as e:
                print(f"   ❌ Failed to load batch: {e}")
                return False
        
        print("\n🎉 Data loading test PASSED!")
        print("✅ All data splits have batches")
        print("✅ Batch loading works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test FAILED: {e}")
        return False

def main():
    """Run data loading test"""
    print("🚀 TESTING DATA LOADING WITH UPDATED CONFIGURATION")
    print("=" * 60)
    
    success = test_data_loading()
    
    if success:
        print("\n✅ Data loading configuration is FIXED!")
        print("✅ Ready to run systematic component testing")
    else:
        print("\n❌ Data loading still has issues")
        print("⚠️  Need to investigate further")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)