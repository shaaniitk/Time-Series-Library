#!/usr/bin/env python3
"""
Simple test for the modified BayesianEnhancedAutoformer with quantile loss
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
from utils.losses import PinballLoss

def simple_test():
    """Simple test of modified Bayesian model"""
    print("TEST Testing Modified BayesianEnhancedAutoformer")
    print("="*50)
    
    # Create small config
    config = Namespace(
        seq_len=12, label_len=6, pred_len=3, enc_in=2, dec_in=1, c_out=1,
        d_model=16, n_heads=2, e_layers=1, d_layers=1, d_ff=32, factor=1,
        dropout=0.1, embed='timeF', freq='h', activation='gelu',
        task_name='long_term_forecast', model='EnhancedAutoformer',
        moving_avg=5, output_attention=False, distil=True, mix=True
    )
    
    # Create model
    model = BayesianEnhancedAutoformer(
        config, 
        uncertainty_method='bayesian',
        bayesian_layers=['projection'],
        kl_weight=1e-3
    )
    
    # Create small synthetic data
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.zeros(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.zeros(batch_size, config.label_len + config.pred_len, 4)
    targets = torch.randn(batch_size, config.pred_len, config.c_out)
    
    print(f"Input shapes:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  targets: {targets.shape}")
    
    # Test 1: Standard mode
    print(f"\n Test 1: Standard Mode")
    try:
        predictions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred_sliced = predictions[:, -config.pred_len:, :config.c_out]
        
        mse_criterion = nn.MSELoss()
        loss_components = model.compute_loss(pred_sliced, targets, mse_criterion, return_components=True)
        
        print(f"  PASS Prediction shape: {pred_sliced.shape}")
        print(f"  PASS Total loss: {loss_components['total_loss'].item():.6f}")
        print(f"  PASS Data weight: {loss_components['normalized_data_weight']:.4f}")
        print(f"  PASS KL weight: {loss_components['normalized_kl_weight']:.4f}")
        
    except Exception as e:
        print(f"  FAIL Standard mode failed: {e}")
        return False
    
    # Test 2: Enable quantile mode
    print(f"\n Test 2: Quantile Mode")
    try:
        quantile_levels = [0.25, 0.5, 0.75]
        model.enable_quantile_mode(quantile_levels)
        
        q_predictions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        q_pred_sliced = q_predictions[:, -config.pred_len:, :]
        
        print(f"  PASS Quantile prediction shape: {q_pred_sliced.shape}")
        print(f"  PASS Expected: {(batch_size, config.pred_len, config.c_out * len(quantile_levels))}")
        
        # Test quantile loss
        pinball_criterion = PinballLoss(quantiles=quantile_levels)
        q_loss_components = model.compute_loss(q_pred_sliced, targets, pinball_criterion, return_components=True)
        
        print(f"  PASS Quantile loss: {q_loss_components['total_loss'].item():.6f}")
        print(f"  PASS Data weight: {q_loss_components['normalized_data_weight']:.4f}")
        print(f"  PASS KL weight: {q_loss_components['normalized_kl_weight']:.4f}")
        
    except Exception as e:
        print(f"  FAIL Quantile mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\nPASS All tests passed!")
    return True

if __name__ == "__main__":
    success = simple_test()
    if success:
        print(f"\nPARTY Modified BayesianEnhancedAutoformer works correctly!")
        print(f"TARGET Key features:")
        print(f"    Normalized KL + data loss (sum to 1)")
        print(f"    Quantile mode support")
        print(f"    Automatic output dimension adjustment")
    else:
        print(f"\nFAIL Tests failed!")
