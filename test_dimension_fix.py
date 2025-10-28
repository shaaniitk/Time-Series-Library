#!/usr/bin/env python3
"""
Test the dimension fix for data loading
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_dimension_fix():
    """Test that dimensions are properly aligned"""
    
    # Test configuration with FIXED dimensions
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
        
        # Data splits
        'validation_length': 500,
        'test_length': 500,
        
        # FIXED: Match actual data dimensions
        'enc_in': 118,    # Actual data has 118 features
        'dec_in': 118,    # Match encoder
        'c_out': 4,
        
        # Model config
        'd_model': 128,
        'n_heads': 8,
        'dropout': 0.1,
        'batch_size': 8,  # Required for data_provider
        
        # Celestial system
        'use_celestial_graph': True,
        'aggregate_waves_to_celestial': True,
        'num_celestial_bodies': 13,
        'num_input_waves': 118,  # Match enc_in
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
        'model_id': 'test_dimension_fix',
    }
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = SimpleConfig(config)
    
    try:
        print("🔍 Testing dimension alignment...")
        print(f"   enc_in: {args.enc_in}")
        print(f"   dec_in: {args.dec_in}")
        print(f"   num_input_waves: {args.num_input_waves}")
        
        # Test data loading
        from data_provider.data_factory import data_provider
        
        dataset, loader = data_provider(args, flag='train')
        
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Loader batches: {len(loader)}")
        
        if len(loader) == 0:
            print("❌ Still getting 0 batches!")
            return False
        
        # Test batch loading
        batch = next(iter(loader))
        batch_x = batch[0]
        
        print(f"   Batch shape: {batch_x.shape}")
        print(f"   Expected: [batch_size, {args.seq_len}, {args.enc_in}]")
        
        if batch_x.shape[-1] == args.enc_in:
            print("✅ Dimensions match!")
            
            # Test model creation
            from models.Celestial_Enhanced_PGAT_Modular import Model
            
            model = Model(args)
            print("✅ Model created successfully!")
            
            # Test forward pass
            import torch
            with torch.no_grad():
                x_enc = batch_x[:2].float()  # Take 2 samples and convert to float32
                x_mark_enc = torch.randn(2, args.seq_len, 4).float()
                x_dec = torch.randn(2, args.label_len + args.pred_len, args.dec_in).float()
                x_mark_dec = torch.randn(2, args.label_len + args.pred_len, 4).float()
                
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                if isinstance(output, tuple):
                    predictions = output[0]
                else:
                    predictions = output
                
                print(f"✅ Forward pass successful: {predictions.shape}")
                return True
        else:
            print(f"❌ Dimension mismatch: got {batch_x.shape[-1]}, expected {args.enc_in}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run dimension fix test"""
    print("🔧 TESTING DIMENSION FIX")
    print("=" * 40)
    
    success = test_dimension_fix()
    
    if success:
        print("\n✅ DIMENSION FIX SUCCESSFUL!")
        print("✅ Ready to run component tests")
    else:
        print("\n❌ DIMENSION FIX FAILED")
        print("⚠️  Need further investigation")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)