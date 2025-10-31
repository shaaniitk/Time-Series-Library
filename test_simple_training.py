#!/usr/bin/env python3
"""
Test simple training with minimal configuration
"""

import torch
import yaml
import sys
from pathlib import Path

def test_simple_training():
    print("🧪 TESTING SIMPLE TRAINING")
    print("=" * 40)
    
    # Create a minimal config with all required attributes
    minimal_config = {
        'seq_len': 500,
        'pred_len': 20,
        'label_len': 250,
        'batch_size': 2,  # Smaller batch
        'learning_rate': 0.001,
        'd_model': 64,    # Much smaller model
        'n_heads': 4,     # Fewer heads
        'e_layers': 2,    # Fewer layers
        'd_layers': 1,    # Fewer layers
        'enc_in': 118,
        'dec_in': 118,
        'c_out': 4,
        'data': 'custom',
        'root_path': './data',
        'data_path': 'prepared_financial_data.csv',
        'target_wave_indices': [0, 1, 2, 3],
        'num_celestial_bodies': 13,
        'dropout': 0.1,
        'use_celestial_graph': False,  # Disable complex features
        'aggregate_waves_to_celestial': False,
        'use_mixture_decoder': False,
        'enable_mdn_decoder': False,
        'use_stochastic_learner': False,
        'debug_mode': False,
        # Required data provider attributes
        'embed': 'timeF',
        'freq': 'd',
        'features': 'MS',
        'target': 'log_Open,log_High,log_Low,log_Close',
        'seasonal_patterns': 'Monthly',
        'scale': True,
        # Other required attributes
        'activation': 'gelu',
        'factor': 1,
        'd_ff': 256,
        'num_workers': 0
    }
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = SimpleConfig(minimal_config)
    
    try:
        # Load data
        from data_provider.data_factory import data_provider
        train_data, train_loader = data_provider(args, flag='train')
        print(f"✅ Data loaded: {len(train_loader)} batches")
        
        # Get first batch
        batch_iter = iter(train_loader)
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(batch_iter)
        
        # Convert to float32
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        
        print(f"✅ Batch loaded: {batch_x.shape}")
        
        # Load model
        from models.Celestial_Enhanced_PGAT_Modular import Model
        model = Model(args)
        print(f"✅ Model loaded")
        
        # Create decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len-args.label_len:, :]).float()
        dec_inp[:, :args.label_len, :] = batch_x[:, -args.label_len:, :]
        
        # Test training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        print("Testing training step...")
        
        # Forward pass
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Extract targets
        targets = batch_y[:, -args.pred_len:, :args.c_out]
        
        # Loss
        loss = torch.nn.MSELoss()(predictions, targets)
        print(f"Loss: {loss.item():.6f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Step
        optimizer.step()
        
        print("✅ Training step successful!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_training()
    sys.exit(0 if success else 1)