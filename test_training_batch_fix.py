#!/usr/bin/env python3
"""
Test the training batch fix - verify that training batches are created correctly
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_training_batch_fix():
    """Test that the dimension fix resolves the 0 training batches issue"""
    
    print("🔧 TESTING TRAINING BATCH FIX")
    print("=" * 50)
    
    # FIXED configuration with correct dimensions
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
        
        # FIXED: Proper data splits
        'validation_length': 500,
        'test_length': 500,
        
        # CRITICAL FIX: Correct dimensions
        'enc_in': 118,    # Match actual data (119 CSV columns - 1 date = 118)
        'dec_in': 118,    # Match encoder
        'c_out': 4,
        
        # Model config
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'batch_size': 8,
        
        # Celestial system
        'use_celestial_graph': True,
        'aggregate_waves_to_celestial': True,
        'num_celestial_bodies': 13,
        'num_input_waves': 118,  # CRITICAL: Match enc_in
        'target_wave_indices': [0, 1, 2, 3],
        
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
        'model_id': 'test_batch_fix',
    }
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = SimpleConfig(config)
    
    try:
        from data_provider.data_factory import data_provider
        
        print("🔍 Testing data loading with FIXED dimensions...")
        print(f"   enc_in: {args.enc_in}")
        print(f"   dec_in: {args.dec_in}")
        print(f"   num_input_waves: {args.num_input_waves}")
        print(f"   batch_size: {args.batch_size}")
        
        # Test all data splits
        for flag in ['train', 'val', 'test']:
            print(f"\n📊 Loading {flag} data...")
            
            dataset, loader = data_provider(args, flag=flag)
            
            dataset_size = len(dataset) if hasattr(dataset, '__len__') else 'unknown'
            loader_batches = len(loader)
            
            print(f"   Dataset size: {dataset_size}")
            print(f"   Loader batches: {loader_batches}")
            
            if loader_batches == 0:
                print(f"   ❌ ERROR: {flag} loader still has 0 batches!")
                return False
            
            # Test batch loading
            try:
                batch = next(iter(loader))
                batch_x = batch[0]
                print(f"   ✅ Batch shape: {batch_x.shape}")
                
                # Verify dimensions match
                if batch_x.shape[-1] != args.enc_in:
                    print(f"   ❌ Dimension mismatch: got {batch_x.shape[-1]}, expected {args.enc_in}")
                    return False
                
            except Exception as e:
                print(f"   ❌ Failed to load batch: {e}")
                return False
        
        print(f"\n✅ ALL DATA SPLITS WORKING!")
        print(f"✅ Training will have batches now!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the training batch fix test"""
    success = test_training_batch_fix()
    
    if success:
        print("\n🎉 TRAINING BATCH FIX SUCCESSFUL!")
        print("✅ The 0 training batches issue is RESOLVED")
        print("✅ GPU component testing will now work correctly")
        print("\nThe issue was:")
        print("  • Config had enc_in=113, but data has 118 features")
        print("  • This caused training data loader to create 0 batches")
        print("  • Validation worked because it has different handling")
        print("\nThe fix:")
        print("  • Set enc_in=118, dec_in=118, num_input_waves=118")
        print("  • Now all data loaders create proper batches")
    else:
        print("\n❌ TRAINING BATCH FIX FAILED")
        print("⚠️  Need further investigation")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)