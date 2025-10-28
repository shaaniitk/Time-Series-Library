#!/usr/bin/env python3
"""
Quick validation test to verify scaler fix is working
Just runs a few training steps to check validation loss behavior
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def quick_validation_test():
    """Quick test to verify scaler fix with minimal training"""
    
    print("🚀 QUICK VALIDATION TEST")
    print("=" * 60)
    
    # Create minimal config
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
        'batch_size': 8,  # Small batch for quick test
        'num_workers': 0,
        'seasonal_patterns': 'Monthly',
        'task_name': 'long_term_forecast',
        'scale': True,
        # Model config - minimal
        'd_model': 64,  # Very small model
        'n_heads': 4,
        'e_layers': 1,
        'd_layers': 1,
        'd_ff': 128,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False,
        'distil': True,
        'mix': True,
        'enc_in': 118,  # Will be set properly
        'dec_in': 118,
        'c_out': 4,
        'factor': 1,
        'learning_rate': 0.001,
        'lradj': 'type1',
        'use_amp': False,
        'patience': 3,
        'train_epochs': 1,  # Just 1 epoch
        'itr': 1,
        'des': 'test',
        'loss': 'MSE',
        'checkpoints': './checkpoints/',
        'use_gpu': torch.cuda.is_available(),
        'gpu': 0,
        'use_multi_gpu': False,
        'devices': '0',
        'seed': 2021,
        'detail_freq': 'h',
        'inverse': False,
        'model_id': 'quick_test',
        'model': 'Celestial_Enhanced_PGAT'
    }
    
    args = SimpleConfig(config)
    
    try:
        # Import necessary modules
        from scripts.train.train_celestial_production import data_provider_with_scaler
        from data_provider.data_factory import data_provider
        from models.Celestial_Enhanced_PGAT import Model
        
        print("✅ Imports successful")
        
        # Set device
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {device}")
        
        # Create datasets with scaler fix
        print("\n📊 Creating datasets...")
        train_dataset, train_loader = data_provider(args, flag='train')
        
        # Get scalers from training
        train_scaler = getattr(train_dataset, 'scaler', None)
        train_target_scaler = getattr(train_dataset, 'target_scaler', None)
        
        # Create validation dataset with proper scalers
        val_dataset, val_loader = data_provider_with_scaler(
            args, flag='val', 
            scaler=train_scaler, 
            target_scaler=train_target_scaler
        )
        
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Scalers shared: {train_scaler is getattr(val_dataset, 'scaler', None)}")
        
        # Create model
        print("\n🧠 Creating model...")
        # Get feature dimensions from dataset
        sample_batch = next(iter(train_loader))
        feature_dim = sample_batch[0].shape[-1]  # Last dimension is features
        args.enc_in = feature_dim
        args.dec_in = feature_dim
        # Set d_model to match feature dimension to avoid compression issues
        args.d_model = feature_dim
        
        model = Model(args).to(device)
        model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Quick training test - just 3 batches
        print("\n🏃 Quick training test (3 batches)...")
        model.train()
        
        train_losses = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            if i >= 3:  # Only 3 batches
                break
                
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            model_optim.zero_grad()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            loss = criterion(outputs, batch_y)
            train_losses.append(loss.item())
            
            loss.backward()
            model_optim.step()
            
            print(f"    Batch {i+1}: loss = {loss.item():.6f}")
        
        avg_train_loss = np.mean(train_losses)
        print(f"  Average training loss: {avg_train_loss:.6f}")
        
        # Quick validation test - just 3 batches
        print("\n🔍 Quick validation test (3 batches)...")
        model.eval()
        
        val_losses = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                if i >= 3:  # Only 3 batches
                    break
                    
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
                
                print(f"    Batch {i+1}: loss = {loss.item():.6f}")
        
        avg_val_loss = np.mean(val_losses)
        print(f"  Average validation loss: {avg_val_loss:.6f}")
        
        # Analysis
        print("\n📈 ANALYSIS:")
        print(f"  Training loss: {avg_train_loss:.6f}")
        print(f"  Validation loss: {avg_val_loss:.6f}")
        print(f"  Loss ratio (val/train): {avg_val_loss/avg_train_loss:.2f}")
        
        # Check if validation loss is reasonable
        if avg_val_loss < 1000:  # Should be much lower than before fix
            print("  ✅ Validation loss looks reasonable!")
        else:
            print("  ⚠️  Validation loss still very high")
            
        if abs(avg_val_loss - avg_train_loss) / avg_train_loss < 10:  # Within 10x
            print("  ✅ Training and validation losses are in similar range!")
        else:
            print("  ⚠️  Large gap between training and validation losses")
        
        # Check data statistics
        print("\n📊 Data statistics check:")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        train_x_stats = f"mean={train_batch[0].mean():.4f}, std={train_batch[0].std():.4f}"
        val_x_stats = f"mean={val_batch[0].mean():.4f}, std={val_batch[0].std():.4f}"
        
        print(f"  Training X: {train_x_stats}")
        print(f"  Validation X: {val_x_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick validation test"""
    
    print("🧪 QUICK VALIDATION TEST - SCALER FIX VERIFICATION")
    print("=" * 80)
    
    success = quick_validation_test()
    
    if success:
        print("\n✅ QUICK TEST COMPLETED")
        print("Check the results above to see if scaler fix is working!")
    else:
        print("\n❌ QUICK TEST FAILED")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)