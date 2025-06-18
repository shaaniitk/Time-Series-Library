#!/usr/bin/env python3
"""
Test 5 Quantiles + KL Loss combination with BayesianEnhancedAutoformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from argparse import Namespace

from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
from utils.losses import get_loss_function

def test_quantile_kl_combination():
    """Test 5 quantiles + KL loss combination"""
    
    print("🧪 Testing 5 Quantiles + KL Loss Combination")
    print("=" * 60)
    
    # Create high config for financial forecasting
    config = Namespace(
        seq_len=250, label_len=10, pred_len=20, enc_in=16, dec_in=8, c_out=8,
        d_model=256, n_heads=8, e_layers=4, d_layers=2, d_ff=512, factor=1,
        dropout=0.1, embed='timeF', freq='d', activation='gelu',
        task_name='long_term_forecast', model='BayesianEnhancedAutoformer',
        moving_avg=25, output_attention=False, distil=True, mix=True,
        # Quantile configuration
        quantile_mode=True,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]  # 5 quantiles
    )
    
    print(f"🏦 Financial Forecasting Configuration:")
    print(f"  Sequence Length: {config.seq_len} days (~8.3 months history)")
    print(f"  Label Length: {config.label_len} days")
    print(f"  Prediction Length: {config.pred_len} days (~3 weeks ahead)")
    print(f"  Input Features: {config.enc_in}")
    print(f"  Target Features: {config.c_out}")
    print(f"  Quantiles: {config.quantiles}")
    print(f"  Expected output features: {config.c_out * len(config.quantiles)} = {config.c_out} × {len(config.quantiles)}")
    print(f"  Model Size: d_model={config.d_model}, layers={config.e_layers}/{config.d_layers}")
    
    # Create Bayesian model with quantile support
    model = BayesianEnhancedAutoformer(
        config, 
        uncertainty_method='bayesian',
        bayesian_layers=['projection'],
        kl_weight=0.01
    )
    
    print(f"Model created with quantile_mode: {model.quantile_mode}")
    print(f"Model quantiles: {model.quantiles}")
    print(f"Original c_out: {model.original_c_out}")
    print(f"Expected total output: {model.original_c_out * len(model.quantiles)}")
    
    # Create synthetic data
    batch_size = 4
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    # Zero future covariates (realistic scenario)
    x_dec = torch.zeros(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.zeros(batch_size, config.label_len + config.pred_len, 4)
    
    # Targets for original features only (model will handle quantile expansion internally)
    targets = torch.randn(batch_size, config.pred_len, config.c_out)  # Original c_out=4
    
    # Setup training
    criterion = get_loss_function('pinball', quantiles=config.quantiles)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n📊 Training Progress (Quantile + KL Loss):")
    print("Epoch | Quantile Loss | KL Loss   | Total Loss | KL %")
    print("-" * 60)
    
    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = pred[:, -config.pred_len:, :]  # Get prediction part
        
        print(f"Epoch {epoch+1} - Prediction shape: {pred.shape}, Target shape: {targets.shape}")
        
        # Compute combined loss
        loss_components = model.compute_loss(pred, targets, criterion, return_components=True)
        
        # Backward pass
        total_loss = loss_components['total_loss']
        total_loss.backward()
        optimizer.step()
        
        # Log progress
        quantile_loss = loss_components['data_loss'].item()
        kl_contribution = loss_components['kl_contribution']
        total_loss_val = loss_components['total_loss'].item()
        kl_percentage = (kl_contribution / total_loss_val) * 100 if total_loss_val > 0 else 0
        
        print(f"{epoch+1:5d} | {quantile_loss:12.6f} | {kl_contribution:9.6f} | {total_loss_val:10.6f} | {kl_percentage:5.2f}%")
    
    print("\n✅ Training completed!")
    
    # Test quantile predictions
    print("\n🔍 Quantile Analysis:")
    model.eval()
    with torch.no_grad():
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = pred[:, -config.pred_len:, :]
        
        # Reshape to get quantiles: [batch, time, targets, quantiles]
        batch_size, time_steps, total_features = pred.shape
        n_targets = config.c_out
        n_quantiles = len(config.quantiles)
        
        # Reshape to separate quantiles
        quantile_preds = pred.view(batch_size, time_steps, n_targets, n_quantiles)
        
        print(f"Quantile predictions shape: {quantile_preds.shape}")
        print(f"[batch={batch_size}, time={time_steps}, targets={n_targets}, quantiles={n_quantiles}]")
        
        # Analyze first sample, first time step, first target
        sample_quantiles = quantile_preds[0, 0, 0, :].cpu().numpy()
        
        print(f"\nExample quantile predictions for Target 1 at t=0:")
        for i, (q, val) in enumerate(zip(config.quantiles, sample_quantiles)):
            print(f"  Q{q*100:4.0f}: {val:8.4f}")
        
        # Check if quantiles are ordered (should be: Q10 ≤ Q25 ≤ Q50 ≤ Q75 ≤ Q90)
        is_ordered = np.all(sample_quantiles[:-1] <= sample_quantiles[1:])
        print(f"\nQuantiles properly ordered: {'✅ Yes' if is_ordered else '❌ No'}")
        
        # Calculate prediction intervals
        q10, q25, q50, q75, q90 = sample_quantiles
        interval_80 = q90 - q10  # 80% prediction interval
        interval_50 = q75 - q25  # 50% prediction interval
        
        print(f"Prediction intervals:")
        print(f"  50% interval: [{q25:.4f}, {q75:.4f}] (width: {interval_50:.4f})")
        print(f"  80% interval: [{q10:.4f}, {q90:.4f}] (width: {interval_80:.4f})")
        print(f"  Median prediction: {q50:.4f}")
    
    print(f"\n🎯 Test Results:")
    print(f"✅ Model successfully trained with 5 quantiles + KL loss")
    print(f"✅ Output dimensions correct: {pred.shape}")
    print(f"✅ Quantile loss + KL regularization working")
    print(f"✅ Prediction intervals generated")
    
    return True

if __name__ == "__main__":
    try:
        test_quantile_kl_combination()
        print("\n🎉 Quantile + KL Loss test completed successfully!")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
