#!/usr/bin/env python3
"""
Enhanced test showing KL loss behavior during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from argparse import Namespace

from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer

def test_kl_loss_during_training():
    """Test KL loss behavior during training"""
    
    print("🧪 Testing KL Loss During Training")
    print("=" * 50)
    
    # Create medium config
    config = Namespace(
        seq_len=96, label_len=48, pred_len=24, enc_in=16, dec_in=8, c_out=8,
        d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=256, factor=1,
        dropout=0.1, embed='timeF', freq='h', activation='gelu',
        task_name='long_term_forecast', model='EnhancedAutoformer',
        moving_avg=25, output_attention=False, distil=True, mix=True
    )
    
    # Create Bayesian model
    model = BayesianEnhancedAutoformer(
        config, 
        uncertainty_method='bayesian',
        bayesian_layers=['projection'],
        kl_weight=1e-3  # Higher weight to see effect
    )
    
    # Create synthetic data with zero future covariates (medium config)
    batch_size = 4  # Smaller batch for medium config
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    # Use zeros for future covariates (more realistic when no future info available)
    x_dec = torch.zeros(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.zeros(batch_size, config.label_len + config.pred_len, 4)
    targets = torch.randn(batch_size, config.pred_len, config.c_out)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("📊 Training Progress (KL Loss Monitoring):")
    print("Epoch | Data Loss | KL Loss   | Total Loss | KL Contribution")
    print("-" * 60)
    
    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = pred[:, -config.pred_len:, :config.c_out]
        
        # Compute loss with breakdown
        loss_components = model.compute_loss(pred, targets, criterion, return_components=True)
        
        # Backward pass
        total_loss = loss_components['total_loss']
        total_loss.backward()
        optimizer.step()
        
        # Log progress
        data_loss = loss_components['data_loss'].item()
        kl_contribution = loss_components['kl_contribution']
        total_loss_val = loss_components['total_loss'].item()
        kl_percentage = (kl_contribution / total_loss_val) * 100 if total_loss_val > 0 else 0
        
        print(f"{epoch+1:5d} | {data_loss:9.6f} | {kl_contribution:9.6f} | {total_loss_val:10.6f} | {kl_percentage:5.2f}%")
    
    print("\n✅ Training completed!")
    print(f"Final KL contribution: {kl_contribution:.6f}")
    print(f"KL weight used: {model.kl_weight}")
    
    # Test the cleaner architecture integration
    print(f"\n🔍 Architecture Verification:")
    print(f"Model has compute_loss method: {hasattr(model, 'compute_loss')}")
    print(f"Model has get_loss_function method: {hasattr(model, 'get_loss_function')}")
    
    # Show that experiment framework will use it correctly
    if hasattr(model, 'compute_loss'):
        framework_loss = model.compute_loss(pred, targets, criterion)
        print(f"Framework would use: {framework_loss.item():.6f}")
    else:
        framework_loss = criterion(pred, targets)
        print(f"Framework would use: {framework_loss.item():.6f}")
    
    return True

if __name__ == "__main__":
    try:
        test_kl_loss_during_training()
        print("\n🎉 KL Loss training test completed successfully!")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
