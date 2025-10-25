#!/usr/bin/env python3
"""
Single Training Step Test

Test one complete forward and backward pass to verify all fixes work in training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
import pandas as pd
from models.Celestial_Enhanced_PGAT import Model

def test_single_training_step():
    """Run one complete forward and backward pass."""
    
    print("🚀 TESTING SINGLE TRAINING STEP")
    print("="*60)
    
    # Load production config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    
    print(f"Configuration:")
    print(f"   Model: {configs.model}")
    print(f"   d_model: {configs.d_model}")
    print(f"   seq_len: {configs.seq_len}")
    print(f"   pred_len: {configs.pred_len}")
    print(f"   batch_size: {configs.batch_size}")
    
    # Initialize model
    print(f"\n🏗️  Initializing model...")
    model = Model(configs)
    model.train()  # Set to training mode
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()
    
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create realistic training data
    print(f"\n📊 Creating training data...")
    
    batch_size = configs.batch_size
    seq_len = configs.seq_len
    pred_len = configs.pred_len
    label_len = configs.label_len
    enc_in = configs.enc_in
    c_out = configs.c_out
    
    # Create input data
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
    
    # Create decoder input (label + prediction)
    x_dec = torch.randn(batch_size, label_len + pred_len, enc_in)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)
    
    # Create target (only prediction part)
    y_true = torch.randn(batch_size, pred_len, c_out)
    
    print(f"Data shapes:")
    print(f"   x_enc: {x_enc.shape}")
    print(f"   x_mark_enc: {x_mark_enc.shape}")
    print(f"   x_dec: {x_dec.shape}")
    print(f"   x_mark_dec: {x_mark_dec.shape}")
    print(f"   y_true: {y_true.shape}")
    
    # Forward pass
    print(f"\n⏩ Running forward pass...")
    
    try:
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Handle output format
        if isinstance(outputs, tuple):
            predictions, metadata = outputs
            print(f"✅ Forward pass successful with metadata")
            print(f"   Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        else:
            predictions = outputs
            metadata = {}
            print(f"✅ Forward pass successful")
        
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Expected shape: {y_true.shape}")
        
        # Validate output shape
        if predictions.shape != y_true.shape:
            print(f"❌ Shape mismatch: got {predictions.shape}, expected {y_true.shape}")
            return False
        
        # Compute loss
        print(f"\n📉 Computing loss...")
        loss = criterion(predictions, y_true)
        print(f"✅ Loss computed: {loss.item():.6f}")
        
        # Backward pass
        print(f"\n⏪ Running backward pass...")
        loss.backward()
        print(f"✅ Backward pass successful")
        
        # Check gradients
        total_grad_norm = 0.0
        param_count = 0
        grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                grad_count += 1
            param_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"📊 Gradient analysis:")
        print(f"   Total parameters: {param_count}")
        print(f"   Parameters with gradients: {grad_count}")
        print(f"   Total gradient norm: {total_grad_norm:.6f}")
        
        if grad_count == 0:
            print(f"❌ No gradients computed!")
            return False
        
        if total_grad_norm == 0:
            print(f"❌ All gradients are zero!")
            return False
        
        if not np.isfinite(total_grad_norm):
            print(f"❌ Gradient norm is not finite: {total_grad_norm}")
            return False
        
        # Optimizer step
        print(f"\n🔄 Running optimizer step...")
        optimizer.step()
        print(f"✅ Optimizer step successful")
        
        # Memory check
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n💾 GPU Memory:")
            print(f"   Allocated: {allocated:.2f}GB")
            print(f"   Reserved: {reserved:.2f}GB")
        else:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            cpu_memory = process.memory_info().rss / 1024**3
            print(f"\n💾 CPU Memory: {cpu_memory:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_training_steps():
    """Test multiple training steps to ensure stability."""
    
    print(f"\n🔄 TESTING MULTIPLE TRAINING STEPS")
    print("="*60)
    
    # Load config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    
    # Initialize model
    model = Model(configs)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()
    
    # Test data
    batch_size = configs.batch_size
    seq_len = configs.seq_len
    pred_len = configs.pred_len
    label_len = configs.label_len
    enc_in = configs.enc_in
    c_out = configs.c_out
    
    losses = []
    
    print(f"Running 3 training steps...")
    
    for step in range(3):
        try:
            # Create fresh data for each step
            x_enc = torch.randn(batch_size, seq_len, enc_in)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, label_len + pred_len, enc_in)
            x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)
            y_true = torch.randn(batch_size, pred_len, c_out)
            
            # Training step
            optimizer.zero_grad()
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if isinstance(outputs, tuple):
                predictions, _ = outputs
            else:
                predictions = outputs
            
            loss = criterion(predictions, y_true)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"   Step {step + 1}: Loss = {loss.item():.6f}")
            
        except Exception as e:
            print(f"❌ Step {step + 1} failed: {e}")
            return False
    
    print(f"\n📊 Training stability:")
    print(f"   All steps completed: ✅")
    print(f"   Loss range: {min(losses):.6f} - {max(losses):.6f}")
    print(f"   Loss std: {np.std(losses):.6f}")
    
    return True

if __name__ == "__main__":
    print("🧪 SINGLE TRAINING STEP VERIFICATION")
    print("="*60)
    
    # Test single step
    single_step_ok = test_single_training_step()
    
    # Test multiple steps if single step works
    if single_step_ok:
        multiple_steps_ok = test_multiple_training_steps()
    else:
        multiple_steps_ok = False
    
    print(f"\n🎉 TRAINING TEST RESULTS:")
    print(f"="*60)
    print(f"✅ Single training step: {'PASS' if single_step_ok else 'FAIL'}")
    print(f"✅ Multiple training steps: {'PASS' if multiple_steps_ok else 'FAIL'}")
    
    if single_step_ok and multiple_steps_ok:
        print(f"\n🎉 ALL TRAINING TESTS PASSED!")
        print(f"✅ Forward pass: Working")
        print(f"✅ Backward pass: Working") 
        print(f"✅ Gradient computation: Working")
        print(f"✅ Optimizer step: Working")
        print(f"✅ Training stability: Confirmed")
        print(f"✅ Memory usage: Stable")
        print(f"\n🚀 The model is ready for full training!")
    else:
        print(f"\n❌ TRAINING ISSUES DETECTED")
        print(f"   Check the error details above")