#!/usr/bin/env python3
"""
Simple test to verify scaling consistency between training and validation loss
Uses a basic linear model to isolate the scaling issue
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

from data_provider.data_factory import data_provider

class SimpleLinearModel(nn.Module):
    """Very simple linear model for testing scaling"""
    
    def __init__(self, seq_len, pred_len, enc_in, c_out):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        
        # Simple linear transformation
        self.linear = nn.Linear(seq_len * enc_in, pred_len * c_out)
        
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # Flatten input: [batch, seq_len, features] -> [batch, seq_len * features]
        batch_size = batch_x.size(0)
        x_flat = batch_x.view(batch_size, -1)
        
        # Linear transformation
        out_flat = self.linear(x_flat)
        
        # Reshape to [batch, pred_len, c_out]
        outputs = out_flat.view(batch_size, self.pred_len, self.c_out)
        
        return outputs

class SimpleConfig:
    """Simple configuration class."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def scale_targets_for_loss(targets_unscaled, target_scaler, target_indices, device):
    """Scale target values for loss computation"""
    try:
        # Extract only the target features
        targets_only = targets_unscaled[:, :, target_indices]
        targets_np = targets_only.cpu().numpy()
        
        # Reshape for scaler
        batch_size, seq_len, n_targets = targets_np.shape
        targets_reshaped = targets_np.reshape(-1, n_targets)
        
        # Scale using target scaler
        targets_scaled_reshaped = target_scaler.transform(targets_reshaped)
        
        # Reshape back
        targets_scaled_np = targets_scaled_reshaped.reshape(batch_size, seq_len, n_targets)
        
        return torch.from_numpy(targets_scaled_np).float().to(device)
        
    except Exception as e:
        print(f"⚠️  Target scaling failed: {e}")
        return targets_unscaled[:, :, target_indices].to(device)

def test_simple_scaling():
    """Test scaling consistency with simple model"""
    print("🧪 Testing Scaling Consistency with Simple Linear Model")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/celestial_enhanced_pgat.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    args = SimpleConfig(config_dict)
    
    # Add required attributes
    args.task_name = 'long_term_forecast'
    args.model_name = 'SimpleLinear'
    args.data_name = 'custom'
    args.checkpoints = './checkpoints/'
    args.inverse = False
    args.cols = None
    args.num_workers = 0
    args.itr = 1
    args.train_only = False
    args.do_predict = False
    
    device = torch.device("cpu")
    
    # Get data loaders
    print("📂 Loading data...")
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(vali_loader)}")
    
    # Get scalers
    train_scaler = getattr(train_data, 'scaler', None)
    target_scaler = getattr(train_data, 'target_scaler', None)
    
    if train_scaler is None:
        print("❌ No scaler found!")
        return False
    
    print(f"   - ✅ Found main scaler: {train_scaler.n_features_in_} features")
    if target_scaler:
        print(f"   - ✅ Found target scaler: {target_scaler.n_features_in_} features")
    else:
        print("   - ⚠️  Using main scaler for targets")
        target_scaler = train_scaler
    
    # Target indices for OHLC
    target_indices = [0, 1, 2, 3]
    
    # Initialize simple model
    print("🏗️  Initializing Simple Linear Model...")
    model = SimpleLinearModel(
        seq_len=args.seq_len,
        pred_len=args.pred_len, 
        enc_in=args.enc_in,
        c_out=args.c_out
    ).to(device)
    
    criterion = nn.MSELoss()
    
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test one training batch
    print("\n🔥 Testing Training Batch...")
    model.train()
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i >= 1:  # Only test first batch
            break
            
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        # Forward pass
        outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
        
        # Scale targets for loss computation
        y_true_for_loss = scale_targets_for_loss(
            batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
        )
        
        # Compute loss
        y_pred_for_loss = outputs[:, -args.pred_len:, :len(target_indices)]
        train_loss = criterion(y_pred_for_loss, y_true_for_loss)
        
        print(f"📊 TRAIN - Input shape: {batch_x.shape}")
        print(f"📊 TRAIN - Output shape: {outputs.shape}")
        print(f"📊 TRAIN - Pred for loss shape: {y_pred_for_loss.shape}")
        print(f"📊 TRAIN - Target for loss shape: {y_true_for_loss.shape}")
        print(f"📊 TRAIN - Raw batch_y stats: mean={batch_y.mean():.6f}, std={batch_y.std():.6f}")
        print(f"📊 TRAIN - OHLC unscaled: mean={batch_y[:, -args.pred_len:, :4].mean():.6f}, std={batch_y[:, -args.pred_len:, :4].std():.6f}")
        print(f"📊 TRAIN - Scaled targets: mean={y_true_for_loss.mean():.6f}, std={y_true_for_loss.std():.6f}")
        print(f"📊 TRAIN - Model outputs: mean={y_pred_for_loss.mean():.6f}, std={y_pred_for_loss.std():.6f}")
        print(f"🎯 TRAIN LOSS: {train_loss.item():.6f}")
        
        break
    
    # Test one validation batch
    print("\n🔍 Testing Validation Batch...")
    model.eval()
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            if i >= 1:  # Only test first batch
                break
                
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
            
            # Scale targets for loss computation (same as training)
            y_true_for_loss = scale_targets_for_loss(
                batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
            )
            
            # Compute loss
            y_pred_for_loss = outputs[:, -args.pred_len:, :len(target_indices)]
            val_loss = criterion(y_pred_for_loss, y_true_for_loss)
            
            print(f"📊 VAL - Input shape: {batch_x.shape}")
            print(f"📊 VAL - Output shape: {outputs.shape}")
            print(f"📊 VAL - Pred for loss shape: {y_pred_for_loss.shape}")
            print(f"📊 VAL - Target for loss shape: {y_true_for_loss.shape}")
            print(f"📊 VAL - Raw batch_y stats: mean={batch_y.mean():.6f}, std={batch_y.std():.6f}")
            print(f"📊 VAL - OHLC unscaled: mean={batch_y[:, -args.pred_len:, :4].mean():.6f}, std={batch_y[:, -args.pred_len:, :4].std():.6f}")
            print(f"📊 VAL - Scaled targets: mean={y_true_for_loss.mean():.6f}, std={y_true_for_loss.std():.6f}")
            print(f"📊 VAL - Model outputs: mean={y_pred_for_loss.mean():.6f}, std={y_pred_for_loss.std():.6f}")
            print(f"🎯 VAL LOSS: {val_loss.item():.6f}")
            
            break
    
    print("\n" + "=" * 60)
    print("✅ Simple scaling test completed!")
    print(f"📈 Loss comparison: Train={train_loss.item():.6f}, Val={val_loss.item():.6f}")
    
    # Check if losses are in similar scale
    loss_ratio = max(train_loss.item(), val_loss.item()) / min(train_loss.item(), val_loss.item())
    if loss_ratio < 10:  # Within 10x is reasonable for different data
        print("✅ Losses are in similar scale - scaling appears consistent!")
        return True
    else:
        print(f"⚠️  Large loss ratio ({loss_ratio:.2f}x) - may indicate scaling issues")
        return False

if __name__ == "__main__":
    test_simple_scaling()