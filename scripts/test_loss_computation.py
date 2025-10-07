#!/usr/bin/env python3
"""
Test script to verify loss is computed only on target features
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_handle_model_output():
    """Test the corrected handle_model_output function"""
    
    # Import the function
    sys.path.append('scripts/train')
    from train_financial_enhanced_pgat import handle_model_output
    
    print("🧪 Testing Loss Computation Fix")
    print("=" * 50)
    
    # Create test data
    batch_size = 2
    seq_len = 20
    total_features = 118  # All features
    target_features = 4   # OHLC targets
    
    # Simulate model output (should be 4 target features)
    model_output = torch.randn(batch_size, seq_len, target_features)
    
    # Simulate batch_y (contains all 118 features)
    batch_y_full = torch.randn(batch_size, seq_len, total_features)
    
    print(f"Input shapes:")
    print(f"  Model output: {model_output.shape} (predictions for {target_features} targets)")
    print(f"  Batch Y full: {batch_y_full.shape} (all {total_features} features)")
    
    # Test the function
    outputs_processed, targets_only = handle_model_output(
        model_output, batch_y_full, num_targets=target_features
    )
    
    print(f"\nAfter handle_model_output:")
    print(f"  Processed outputs: {outputs_processed.shape}")
    print(f"  Targets only: {targets_only.shape}")
    
    # Verify correctness
    assert outputs_processed.shape == (batch_size, seq_len, target_features), \
        f"Output shape mismatch: {outputs_processed.shape}"
    
    assert targets_only.shape == (batch_size, seq_len, target_features), \
        f"Target shape mismatch: {targets_only.shape}"
    
    # Verify targets_only contains the FIRST 4 features from batch_y_full
    expected_targets = batch_y_full[:, :, :target_features]
    assert torch.allclose(targets_only, expected_targets), \
        "Targets don't match first 4 features of batch_y"
    
    print(f"\n✅ SUCCESS: Loss will be computed on {target_features} target features only!")
    print(f"✅ Target features are correctly extracted from FIRST {target_features} columns")
    print(f"✅ Remaining {total_features - target_features} features are ignored in loss computation")
    
    # Test loss computation
    criterion = torch.nn.MSELoss()
    loss = criterion(outputs_processed, targets_only)
    print(f"\n📊 Test loss computation:")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss computed on shape: {targets_only.shape}")
    
    return True

def test_feature_extraction():
    """Test that we extract the correct features"""
    
    print(f"\n🔍 Testing Feature Extraction")
    print("-" * 30)
    
    # Create test data with known values
    batch_size = 1
    seq_len = 5
    
    # Create batch_y with identifiable values
    # First 4 features (targets): [1, 2, 3, 4]
    # Remaining features: [5, 6, 7, ..., 118]
    batch_y = torch.zeros(batch_size, seq_len, 118)
    for i in range(118):
        batch_y[:, :, i] = i + 1  # Feature i has value i+1
    
    print(f"Created test batch_y with:")
    print(f"  Target features (first 4): {batch_y[0, 0, :4].tolist()}")
    print(f"  Other features (sample): {batch_y[0, 0, 4:8].tolist()}")
    print(f"  Last features: {batch_y[0, 0, -4:].tolist()}")
    
    # Test extraction
    sys.path.append('scripts/train')
    from train_financial_enhanced_pgat import handle_model_output
    
    model_output = torch.randn(batch_size, seq_len, 4)
    _, targets_extracted = handle_model_output(model_output, batch_y, num_targets=4)
    
    print(f"\nExtracted targets: {targets_extracted[0, 0, :].tolist()}")
    
    # Verify we got the first 4 features
    expected = [1.0, 2.0, 3.0, 4.0]
    actual = targets_extracted[0, 0, :].tolist()
    
    assert actual == expected, f"Expected {expected}, got {actual}"
    
    print(f"✅ Correctly extracted FIRST 4 features as targets")
    print(f"✅ Features 5-118 are NOT used in loss computation")
    
    return True

if __name__ == "__main__":
    print("Testing Enhanced SOTA PGAT Loss Computation Fix")
    print("=" * 60)
    
    try:
        test_handle_model_output()
        test_feature_extraction()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Loss computation is now CORRECT")
        print(f"✅ Only 4 target features (OHLC) are used in loss")
        print(f"✅ 114 covariate features are ignored in loss")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()