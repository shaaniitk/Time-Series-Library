#!/usr/bin/env python3
"""
Test script to isolate the first batch processing issue
"""

import sys
import os
import traceback
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_first_batch():
    """Test processing the first batch"""
    print("Testing first batch processing...")
    
    try:
        # Import necessary modules
        from scripts.train.train_celestial_production import create_args_from_config
        from models.Celestial_Enhanced_PGAT_Modular import Model
        from data_provider.data_factory import data_provider
        
        # Load config
        config_path = "configs/celestial_production_deep_ultimate.yaml"
        args = create_args_from_config(config_path)
        
        print("✅ Config loaded and args created")
        
        # Create model
        model = Model(args)
        model.eval()  # Set to eval mode for testing
        
        print("✅ Model created")
        
        # Create data loader
        train_data, train_loader = data_provider(args, 'train')
        
        print(f"✅ Data loader created with {len(train_loader)} batches")
        
        # Get first batch
        first_batch = next(iter(train_loader))
        batch_x, batch_y, batch_x_mark, batch_y_mark = first_batch
        
        print(f"✅ First batch loaded:")
        print(f"  batch_x shape: {batch_x.shape}")
        print(f"  batch_y shape: {batch_y.shape}")
        print(f"  batch_x_mark shape: {batch_x_mark.shape}")
        print(f"  batch_y_mark shape: {batch_y_mark.shape}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        
        print(f"✅ Forward pass successful")
        if isinstance(outputs, tuple):
            print(f"  Output tuple with {len(outputs)} elements")
            if hasattr(outputs[0], 'shape'):
                print(f"  Main output shape: {outputs[0].shape}")
        else:
            print(f"  Output shape: {outputs.shape}")
        
        # Test loss computation
        from layers.modular.losses.directional_trend_loss import HybridMDNDirectionalLoss
        
        loss_config = args.loss
        loss_fn = HybridMDNDirectionalLoss(
            nll_weight=loss_config.get('nll_weight', 0.25),
            direction_weight=loss_config.get('direction_weight', 4.0),
            trend_weight=loss_config.get('trend_weight', 2.0),
            magnitude_weight=loss_config.get('magnitude_weight', 0.15)
        )
        
        # Prepare targets
        y_true = batch_y[:, -args.pred_len:, :args.c_out]
        
        print(f"✅ Target shape: {y_true.shape}")
        
        # Compute loss
        if isinstance(outputs, tuple):
            # Use the main output for loss
            main_output = outputs[0]
            loss = loss_fn(main_output, y_true)
        else:
            loss = loss_fn(outputs, y_true)
        
        print(f"✅ Loss computation successful: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ First batch processing failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("FIRST BATCH PROCESSING TEST")
    print("=" * 60)
    
    success = test_first_batch()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 FIRST BATCH PROCESSING SUCCESSFUL!")
        print("The issue might be elsewhere in the training loop.")
    else:
        print("❌ FIRST BATCH PROCESSING FAILED!")
    print("=" * 60)