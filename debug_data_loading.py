#!/usr/bin/env python3
"""
Debug data loading issue
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def debug_data_loading():
    """Debug the data loading issue"""
    print("🔍 Debugging data loading...")
    
    try:
        # Create a simple config
        class SimpleConfig:
            def __init__(self):
                self.seq_len = 96
                self.pred_len = 24
                self.label_len = 48
                self.enc_in = 118
                self.c_out = 4
                self.d_model = 128
                self.n_heads = 8
                
                # Data settings
                self.data = 'custom'
                self.root_path = './data'
                self.data_path = 'prepared_financial_data.csv'
                self.features = 'MS'
                self.target = 'log_Open,log_High,log_Low,log_Close'
                self.embed = 'timeF'
                self.freq = 'd'
                
                # Data split
                self.validation_length = 8
                self.test_length = 8
                
                # Training
                self.batch_size = 2
                self.num_workers = 0
                
                # Required by data_factory
                self.task_name = 'long_term_forecast'
                self.model_name = 'Enhanced_SOTA_PGAT'
                self.data_name = 'custom'
                self.checkpoints = './checkpoints/'
                self.inverse = False
                self.cols = None
        
        config = SimpleConfig()
        
        # Test data loading
        from data_provider.data_factory import data_provider
        
        print("✅ Testing train data loading...")
        train_dataset, train_loader = data_provider(config, flag='train')
        print(f"✅ Train dataset size: {len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'}")
        print(f"✅ Train loader batches: {len(train_loader)}")
        
        print("✅ Testing val data loading...")
        val_dataset, val_loader = data_provider(config, flag='val')
        print(f"✅ Val dataset size: {len(val_dataset) if hasattr(val_dataset, '__len__') else 'unknown'}")
        print(f"✅ Val loader batches: {len(val_loader)}")
        
        print("✅ Testing test data loading...")
        test_dataset, test_loader = data_provider(config, flag='test')
        print(f"✅ Test dataset size: {len(test_dataset) if hasattr(test_dataset, '__len__') else 'unknown'}")
        print(f"✅ Test loader batches: {len(test_loader)}")
        
        # Test a batch
        if len(train_loader) > 0:
            print("✅ Testing batch loading...")
            for batch in train_loader:
                print(f"✅ Batch loaded successfully! Batch elements: {len(batch)}")
                for i, elem in enumerate(batch):
                    if hasattr(elem, 'shape'):
                        print(f"   Element {i}: {elem.shape}")
                break
        else:
            print("❌ Train loader is empty!")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data_loading()