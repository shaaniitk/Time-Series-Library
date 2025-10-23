#!/usr/bin/env python3
"""
Test Mixed Precision Training

Verify that mixed precision training works without dtype errors.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import yaml
from models.Celestial_Enhanced_PGAT import Model

def test_mixed_precision():
    """Test mixed precision training."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping mixed precision test")
        return False
    
    print("🧪 TESTING MIXED PRECISION TRAINING")
    print("="*50)
    
    # Load config
    with open('configs/celestial_enhanced_pgat_production_no_amp.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    
    # Initialize model on GPU
    device = torch.device('cuda')
    model = Model(configs).to(device)
    model.train()
    
    # Initialize mixed precision components
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"✅ Model initialized on {device}")
    
    # Create test data on GPU
    batch_size = 4  # Smaller batch for testing
    seq_len = configs.seq_len
    
    x_enc = torch.randn(batch_size, seq_len, configs.enc_in, device=device)
    x_mark_enc = torch.randn(batch_size, seq_len, 4, device=device)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in, device=device)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4, device=device)
    y_true = torch.randn(batch_size, configs.pred_len, configs.c_out, device=device)
    
    print(f"✅ Test data created on GPU")
    
    try:
        # Test mixed precision forward and backward pass
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if isinstance(outputs, tuple):
                predictions, _ = outputs
            else:
                predictions = outputs
            
            loss = criterion(predictions, y_true)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✅ Mixed precision training step successful!")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Scaler scale: {scaler.get_scale()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mixed_precision()
    
    if success:
        print(f"\n🎉 MIXED PRECISION TEST PASSED!")
        print(f"✅ Mixed precision training is working correctly")
    else:
        print(f"\n❌ MIXED PRECISION TEST FAILED!")
        print(f"   Use the no_amp config for training without mixed precision")
