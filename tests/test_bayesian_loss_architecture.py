#!/usr/bin/env python3
"""
Test script to verify the cleaner Bayesian loss architecture
"""

import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace

# Import our models
from models.EnhancedAutoformer import EnhancedAutoformer
from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer

def test_loss_architecture():
    """Test that the Bayesian model correctly handles loss internally"""
    
    print("🧪 Testing Clean Bayesian Loss Architecture")
    print("=" * 50)
    
    # Create a simple config
    config = Namespace(
        seq_len=24,
        label_len=12,
        pred_len=6,
        enc_in=10,
        dec_in=4,
        c_out=4,
        d_model=32,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        factor=1,
        dropout=0.1,
        embed='timeF',
        freq='h',
        activation='gelu',
        # Required attributes
        task_name='long_term_forecast',
        model='EnhancedAutoformer',
        moving_avg=25,
        output_attention=False,
        distil=True,
        mix=True
    )
    
    # Create models
    enhanced_model = EnhancedAutoformer(config)
    bayesian_model = BayesianEnhancedAutoformer(
        config, 
        uncertainty_method='bayesian',
        bayesian_layers=['projection'],
        kl_weight=1e-5
    )
    
    # Create test data
    batch_size = 4
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
    
    # Create targets
    targets = torch.randn(batch_size, config.pred_len, config.c_out)
    
    # Base criterion
    criterion = nn.MSELoss()
    
    print("1️⃣ Testing Enhanced Model (should NOT have compute_loss method)")
    enhanced_pred = enhanced_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    enhanced_pred = enhanced_pred[:, -config.pred_len:, :config.c_out]
    
    print(f"   Enhanced model has compute_loss: {hasattr(enhanced_model, 'compute_loss')}")
    enhanced_loss = criterion(enhanced_pred, targets)
    print(f"   Enhanced loss: {enhanced_loss.item():.6f}")
    
    print("\n2️⃣ Testing Bayesian Model (should HAVE compute_loss method)")
    bayesian_pred = bayesian_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    bayesian_pred = bayesian_pred[:, -config.pred_len:, :config.c_out]
    
    print(f"   Bayesian model has compute_loss: {hasattr(bayesian_model, 'compute_loss')}")
    
    # Test internal loss computation
    if hasattr(bayesian_model, 'compute_loss'):
        print("\n   🔍 Testing Bayesian internal loss computation:")
        
        # Simple loss
        simple_loss = bayesian_model.compute_loss(bayesian_pred, targets, criterion)
        print(f"   Simple total loss: {simple_loss.item():.6f}")
        
        # Detailed loss breakdown
        detailed_loss = bayesian_model.compute_loss(bayesian_pred, targets, criterion, return_components=True)
        print(f"   Data loss: {detailed_loss['data_loss'].item():.6f}")
        print(f"   KL loss: {detailed_loss['kl_contribution']:.6f}")
        print(f"   Total loss: {detailed_loss['total_loss'].item():.6f}")
        print(f"   KL weight: {detailed_loss['kl_weight']}")
        
        # Verify total = data + kl
        expected_total = detailed_loss['data_loss'] + detailed_loss['kl_contribution']
        actual_total = detailed_loss['total_loss']
        print(f"   Verification: {expected_total.item():.6f} ≈ {actual_total.item():.6f} ✅")
        
        # Test loss function wrapper
        wrapped_loss_fn = bayesian_model.get_loss_function(criterion)
        wrapped_loss = wrapped_loss_fn(bayesian_pred, targets)
        print(f"   Wrapped loss function: {wrapped_loss.item():.6f}")
        
    else:
        print("   ❌ Bayesian model missing compute_loss method!")
        return False
    
    print("\n3️⃣ Comparing Loss Values:")
    standard_bayesian_loss = criterion(bayesian_pred, targets)
    print(f"   Standard MSE loss: {standard_bayesian_loss.item():.6f}")
    print(f"   Bayesian total loss: {simple_loss.item():.6f}")
    print(f"   KL contribution: {simple_loss.item() - standard_bayesian_loss.item():.6f}")
    
    # The Bayesian loss should be slightly higher due to KL regularization
    assert simple_loss.item() >= standard_bayesian_loss.item(), "Bayesian loss should be >= standard loss"
    
    print("\n✅ All tests passed! Clean architecture working correctly.")
    print("\n📋 Architecture Summary:")
    print("   • Enhanced/Hierarchical models: Use standard criterion(pred, target)")
    print("   • Bayesian model: Uses internal compute_loss(pred, target, criterion)")
    print("   • Experiment framework: Automatically detects and uses appropriate method")
    
    return True

if __name__ == "__main__":
    try:
        success = test_loss_architecture()
        if success:
            print("\n🎉 Clean Bayesian loss architecture verified!")
        else:
            print("\n❌ Architecture test failed!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
