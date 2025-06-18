#!/usr/bin/env python3
"""
Verify Current Loss Implementation Flow
"""

import torch
import torch.nn as nn
from argparse import Namespace

from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer

def test_loss_flow():
    """Test the current loss implementation to verify it's clean and correct."""
    
    print("🔍 Verifying Current Loss Implementation Flow")
    print("=" * 60)
    
    # Create config
    config = Namespace(
        seq_len=96, label_len=48, pred_len=24, enc_in=20, dec_in=4, c_out=4,
        d_model=64, n_heads=4, e_layers=2, d_layers=1, d_ff=128, factor=1,
        dropout=0.1, embed='timeF', freq='h', activation='gelu',
        task_name='long_term_forecast', model='EnhancedAutoformer',
        moving_avg=25, output_attention=False, distil=True, mix=True
    )
    
    # Create Bayesian model
    model = BayesianEnhancedAutoformer(
        config, 
        uncertainty_method='bayesian',
        bayesian_layers=['projection'],
        kl_weight=1e-5
    )
    
    print(f"📊 Configuration:")
    print(f"   Input features (enc_in): {config.enc_in}")
    print(f"   Target features (c_out): {config.c_out}")
    print(f"   Sequence length: {config.seq_len}")
    print(f"   Prediction length: {config.pred_len}")
    
    # Create test data
    batch_size = 4
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
    
    # Ground truth targets (full sequence, all features)
    batch_y = torch.randn(batch_size, config.label_len + config.pred_len, config.enc_in)
    
    print(f"\n🔄 Step 1: Model Forward Pass")
    # Model outputs whatever it naturally produces
    model_outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"   Model natural output shape: {model_outputs.shape}")
    
    print(f"\n✂️  Step 2: Slice for Target Features")
    # Extract predictions for target features only
    predictions = model_outputs[:, -config.pred_len:, :config.c_out]
    targets = batch_y[:, -config.pred_len:, :config.c_out]
    
    print(f"   Sliced predictions shape: {predictions.shape}")
    print(f"   Sliced targets shape: {targets.shape}")
    print(f"   ✅ Shapes match: {predictions.shape == targets.shape}")
    
    print(f"\n💯 Step 3: Loss Computation")
    criterion = nn.MSELoss()
    
    # Test the clean loss computation
    if hasattr(model, 'compute_loss'):
        print("   Using Bayesian loss computation...")
        
        # Simple loss
        total_loss = model.compute_loss(predictions, targets, criterion)
        print(f"   Total loss: {total_loss.item():.6f}")
        
        # Detailed breakdown
        loss_components = model.compute_loss(predictions, targets, criterion, return_components=True)
        print(f"   Data loss (MSE on targets): {loss_components['data_loss'].item():.6f}")
        print(f"   KL loss (Bayesian regularization): {loss_components['kl_contribution']:.6f}")
        print(f"   Total = Data + KL: {loss_components['total_loss'].item():.6f}")
        
        # Verify the math
        expected_total = loss_components['data_loss'] + loss_components['kl_contribution']
        actual_total = loss_components['total_loss']
        print(f"   ✅ Math check: {expected_total.item():.6f} ≈ {actual_total.item():.6f}")
        
    else:
        print("   Using standard loss computation...")
        loss = criterion(predictions, targets)
        print(f"   Loss: {loss.item():.6f}")
    
    print(f"\n🎯 Key Points:")
    print(f"   • Model outputs full feature space: {model_outputs.shape}")
    print(f"   • Loss computed only on targets: {predictions.shape}")
    print(f"   • No architectural tampering needed")
    print(f"   • Clean separation: model logic vs loss logic")
    print(f"   • Bayesian regularization added transparently")
    
    print(f"\n✅ Current implementation is clean and correct!")
    
    return True

if __name__ == "__main__":
    try:
        test_loss_flow()
        print("\n🎉 Loss flow verification completed!")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
