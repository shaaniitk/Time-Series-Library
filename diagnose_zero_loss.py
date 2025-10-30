#!/usr/bin/env python3
"""
Systematic diagnosis of zero loss issue
Let's trace exactly what's happening step by step
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def diagnose_zero_loss():
    """Systematically diagnose why loss is zero"""
    
    print("=" * 60)
    print("ZERO LOSS DIAGNOSTIC")
    print("=" * 60)
    
    # Load config
    config_path = "configs/celestial_production_fixed.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = SimpleConfig(config_dict)
    
    print(f"✓ Config loaded: {config_path}")
    print(f"  - seq_len: {args.seq_len}")
    print(f"  - pred_len: {args.pred_len}")
    print(f"  - c_out: {args.c_out}")
    print(f"  - batch_size: {args.batch_size}")
    
    # Load data
    from data_provider.data_factory import data_provider
    
    print("\n1. LOADING DATA...")
    train_data, train_loader = data_provider(args, flag='train')
    print(f"✓ Train data loaded: {len(train_loader)} batches")
    
    # Get first batch
    batch_iter = iter(train_loader)
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(batch_iter)
    
    print(f"\n2. BATCH ANALYSIS...")
    print(f"  batch_x shape: {batch_x.shape}")
    print(f"  batch_y shape: {batch_y.shape}")
    print(f"  batch_x_mark shape: {batch_x_mark.shape}")
    print(f"  batch_y_mark shape: {batch_y_mark.shape}")
    
    # Check data ranges
    print(f"\n3. DATA RANGES...")
    print(f"  batch_x: min={batch_x.min():.6f}, max={batch_x.max():.6f}, mean={batch_x.mean():.6f}")
    print(f"  batch_y: min={batch_y.min():.6f}, max={batch_y.max():.6f}, mean={batch_y.mean():.6f}")
    
    # Check for NaN/Inf
    print(f"\n4. DATA VALIDITY...")
    print(f"  batch_x has NaN: {torch.isnan(batch_x).any()}")
    print(f"  batch_x has Inf: {torch.isinf(batch_x).any()}")
    print(f"  batch_y has NaN: {torch.isnan(batch_y).any()}")
    print(f"  batch_y has Inf: {torch.isinf(batch_y).any()}")
    
    # Load model
    print(f"\n5. LOADING MODEL...")
    try:
        from models.Celestial_Enhanced_PGAT_Modular import Model
        model = Model(args)
        print(f"✓ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test model forward pass
    print(f"\n6. MODEL FORWARD PASS...")
    model.eval()
    
    # Prepare decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len-args.label_len:, :]).float()
    dec_inp[:, :args.label_len, :] = batch_x[:, -args.label_len:, :]
    
    print(f"  dec_inp shape: {dec_inp.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        print(f"✓ Forward pass successful")
        
        # Analyze outputs
        if isinstance(outputs, tuple):
            predictions = outputs[0]
            print(f"  Output is tuple with {len(outputs)} elements")
            print(f"  Predictions shape: {predictions.shape}")
        else:
            predictions = outputs
            print(f"  Output shape: {predictions.shape}")
        
        print(f"  Predictions: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
        print(f"  Predictions has NaN: {torch.isnan(predictions).any()}")
        print(f"  Predictions has Inf: {torch.isinf(predictions).any()}")
        
        # Check if predictions are constant
        pred_std = predictions.std()
        print(f"  Predictions std: {pred_std:.6f}")
        if pred_std < 1e-6:
            print(f"  ⚠️  PREDICTIONS ARE NEARLY CONSTANT!")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test loss computation
    print(f"\n7. LOSS COMPUTATION...")
    
    # Extract targets - FIXED: Only extract target features, not all features
    targets_all = batch_y[:, -args.pred_len:, :]
    
    # Extract only the target features (first c_out columns)
    if hasattr(args, 'target_wave_indices') and args.target_wave_indices:
        targets = targets_all[:, :, args.target_wave_indices]
    else:
        targets = targets_all[:, :, :args.c_out]
    
    print(f"  All targets shape: {targets_all.shape}")
    print(f"  Target features shape: {targets.shape}")
    print(f"  Targets: min={targets.min():.6f}, max={targets.max():.6f}, mean={targets.mean():.6f}")
    
    # Test different loss functions
    print(f"\n8. TESTING LOSS FUNCTIONS...")
    
    # MSE Loss
    mse_loss = nn.MSELoss()
    loss_mse = mse_loss(predictions, targets)
    print(f"  MSE Loss: {loss_mse.item():.6f}")
    
    # MAE Loss
    mae_loss = nn.L1Loss()
    loss_mae = mae_loss(predictions, targets)
    print(f"  MAE Loss: {loss_mae.item():.6f}")
    
    # Manual MSE calculation
    manual_mse = ((predictions - targets) ** 2).mean()
    print(f"  Manual MSE: {manual_mse.item():.6f}")
    
    # Check if predictions and targets are identical (fix dtype)
    if torch.allclose(predictions.double(), targets.double(), atol=1e-6):
        print(f"  ⚠️  PREDICTIONS AND TARGETS ARE NEARLY IDENTICAL!")
    
    # Check difference statistics
    diff = predictions - targets
    print(f"  Difference: min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")
    
    # Test with random predictions
    print(f"\n9. TESTING WITH RANDOM PREDICTIONS...")
    random_preds = torch.randn_like(predictions)
    random_loss = mse_loss(random_preds, targets)
    print(f"  Random predictions MSE: {random_loss.item():.6f}")
    
    # Test with zero predictions
    zero_preds = torch.zeros_like(predictions)
    zero_loss = mse_loss(zero_preds, targets)
    print(f"  Zero predictions MSE: {zero_loss.item():.6f}")
    
    # Test with constant predictions
    constant_preds = torch.ones_like(predictions) * targets.mean()
    constant_loss = mse_loss(constant_preds, targets)
    print(f"  Constant predictions MSE: {constant_loss.item():.6f}")
    
    print(f"\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    # Summary
    if loss_mse.item() < 1e-6:
        print("🔍 LIKELY CAUSES OF ZERO LOSS:")
        print("  1. Model outputs are identical to targets (perfect prediction)")
        print("  2. Model outputs are constant/zero")
        print("  3. Targets are constant/zero")
        print("  4. Loss function is not working correctly")
        print("  5. Gradient computation is broken")
    else:
        print("✓ Loss computation appears to be working correctly")

if __name__ == "__main__":
    diagnose_zero_loss()