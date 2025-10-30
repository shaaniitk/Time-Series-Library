#!/usr/bin/env python3
"""
Debug script to isolate the production config issue
"""

import sys
import os
import traceback
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test loading the production config"""
    print("1. Testing config loading...")
    try:
        import yaml
        with open('configs/celestial_production_deep_ultimate.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_args_creation(config):
    """Test creating args from config"""
    print("\n2. Testing args creation...")
    try:
        from scripts.train.train_celestial_production import SimpleConfig
        args = SimpleConfig(config)
        print("✅ Args created successfully")
        print(f"  Model: {args.model}")
        print(f"  d_model: {args.d_model}")
        print(f"  seq_len: {args.seq_len}")
        return args
    except Exception as e:
        print(f"❌ Args creation failed: {e}")
        traceback.print_exc()
        return None

def test_model_initialization(args):
    """Test model initialization"""
    print("\n3. Testing model initialization...")
    try:
        # Import the model
        from models.Celestial_Enhanced_PGAT_Modular import Model
        
        print(f"Creating model with:")
        print(f"  d_model: {args.d_model}")
        print(f"  seq_len: {args.seq_len}")
        print(f"  pred_len: {args.pred_len}")
        print(f"  enc_in: {args.enc_in}")
        print(f"  c_out: {args.c_out}")
        
        # Create model
        model = Model(args)
        print("✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        traceback.print_exc()
        return None

def test_forward_pass(model, args):
    """Test a forward pass"""
    print("\n4. Testing forward pass...")
    try:
        # Create dummy data
        batch_size = 2
        seq_len = args.seq_len
        pred_len = args.pred_len
        enc_in = args.enc_in
        
        print(f"Creating dummy data:")
        print(f"  batch_size: {batch_size}")
        print(f"  seq_len: {seq_len}")
        print(f"  enc_in: {enc_in}")
        
        # Create input tensors
        batch_x = torch.randn(batch_size, seq_len, enc_in)
        batch_y = torch.randn(batch_size, seq_len, enc_in)
        batch_x_mark = torch.randn(batch_size, seq_len, 4)  # Time features
        batch_y_mark = torch.randn(batch_size, seq_len, 4)  # Time features
        
        print("✅ Dummy data created")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        
        print(f"✅ Forward pass successful")
        if isinstance(outputs, tuple):
            print(f"  Output is tuple with {len(outputs)} elements")
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape'):
                    print(f"    Element {i} shape: {output.shape}")
                else:
                    print(f"    Element {i}: {type(output)}")
        else:
            print(f"  Output shape: {outputs.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False

def test_loss_computation(args):
    """Test loss computation"""
    print("\n5. Testing loss computation...")
    try:
        # Test the loss configuration
        loss_config = args.loss
        print(f"Loss config: {loss_config}")
        
        # Try to create the loss function
        from layers.modular.losses.directional_trend_loss import HybridMDNDirectionalLoss
        
        loss_fn = HybridMDNDirectionalLoss(
            nll_weight=loss_config.get('nll_weight', 0.25),
            direction_weight=loss_config.get('direction_weight', 4.0),
            trend_weight=loss_config.get('trend_weight', 2.0),
            magnitude_weight=loss_config.get('magnitude_weight', 0.15)
        )
        
        print("✅ Loss function created successfully")
        return True
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PRODUCTION CONFIG DEBUG")
    print("=" * 60)
    
    # Test each component
    config = test_config_loading()
    if not config:
        exit(1)
    
    args = test_args_creation(config)
    if not args:
        exit(1)
    
    model = test_model_initialization(args)
    if not model:
        exit(1)
    
    forward_success = test_forward_pass(model, args)
    if not forward_success:
        exit(1)
    
    loss_success = test_loss_computation(args)
    if not loss_success:
        exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("The production config should work correctly.")
    print("The issue might be in the training loop itself.")
    print("=" * 60)