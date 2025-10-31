#!/usr/bin/env python3
"""
Diagnose why training stopped without showing loss
"""

import torch
import yaml
import sys
from pathlib import Path
import traceback

def diagnose_training_crash():
    print("🔍 DIAGNOSING TRAINING CRASH")
    print("=" * 50)
    
    # Load config
    config_path = "configs/celestial_production_deep_ultimate.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = SimpleConfig(config)
    
    print("1. Testing data loading...")
    try:
        from data_provider.data_factory import data_provider
        train_data, train_loader = data_provider(args, flag='train')
        print(f"✅ Data loaded: {len(train_loader)} batches")
        
        # Get first batch
        batch_iter = iter(train_loader)
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(batch_iter)
        print(f"✅ First batch loaded: {batch_x.shape}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n2. Testing model initialization...")
    try:
        from models.Celestial_Enhanced_PGAT_Modular import Model
        model = Model(args)
        print(f"✅ Model initialized")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n3. Testing forward pass...")
    try:
        model.eval()
        
        # Convert to float32
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        
        # Create decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len-args.label_len:, :]).float()
        dec_inp[:, :args.label_len, :] = batch_x[:, -args.label_len:, :]
        
        print(f"  Input shapes: x={batch_x.shape}, dec_inp={dec_inp.shape}")
        
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        print(f"✅ Forward pass successful")
        if isinstance(outputs, tuple):
            print(f"  Output: tuple with {len(outputs)} elements")
            print(f"  Predictions shape: {outputs[0].shape}")
        else:
            print(f"  Output shape: {outputs.shape}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n4. Testing loss computation...")
    try:
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Extract targets (same as training script)
        batch_y_targets = batch_y[:, -args.pred_len:, :]
        if hasattr(args, 'target_wave_indices') and args.target_wave_indices:
            batch_y_targets = batch_y_targets[:, :, args.target_wave_indices]
        elif hasattr(args, 'c_out') and args.c_out < batch_y_targets.shape[-1]:
            batch_y_targets = batch_y_targets[:, :, :args.c_out]
        
        # Simple MSE loss test
        loss = torch.nn.MSELoss()(predictions, batch_y_targets)
        print(f"✅ Loss computation successful: {loss.item():.6f}")
        
        if loss.item() == 0.0:
            print(f"⚠️  Loss is zero - this might be the issue!")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n5. Testing training step...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Forward pass
        print("  Forward pass...")
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Loss computation
        print("  Loss computation...")
        loss = torch.nn.MSELoss()(predictions, batch_y_targets)
        print(f"  Loss: {loss.item():.6f}")
        
        # Backward pass
        print("  Backward pass...")
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        print("  Gradient clipping...")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"✅ Backward pass successful, grad_norm: {grad_norm:.6f}")
        
        # Optimizer step
        print("  Optimizer step...")
        optimizer.step()
        print(f"✅ Optimizer step successful")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 50)
    
    print("✅ All components working correctly")
    print("🤔 Training crash might be due to:")
    print("  1. Memory issues (OOM)")
    print("  2. Infinite loop in training loop")
    print("  3. Exception not being caught properly")
    print("  4. Logging configuration issues")
    
    return True

if __name__ == "__main__":
    success = diagnose_training_crash()
    sys.exit(0 if success else 1)