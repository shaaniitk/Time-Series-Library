#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Quick test of model initialization
try:
    from models.Celestial_Enhanced_PGAT import Model
    import argparse
    
    print("🧪 Testing model initialization...")
    
    # Create config manually
    args = argparse.Namespace()
    args.seq_len = 96
    args.pred_len = 24
    args.label_len = 48
    args.enc_in = 118
    args.dec_in = 118
    args.c_out = 1
    args.d_model = 64
    args.n_heads = 4
    args.e_layers = 2
    args.d_layers = 1
    args.dropout = 0.1
    args.use_celestial_graph = True
    args.aggregate_waves_to_celestial = True
    args.celestial_fusion_layers = 3
    args.num_input_waves = 118
    args.target_wave_indices = [0]
    args.use_mixture_decoder = True
    args.use_stochastic_learner = True
    args.use_hierarchical_mapping = True
    args.embed = 'timeF'
    args.freq = 'd'
    args.features = 'MS'
    
    # Initialize model
    model = Model(args)
    print(f"✅ Model initialized successfully!")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    import torch
    batch_size = 2
    seq_len = 96
    pred_len = 24
    
    # Create dummy input
    batch_x = torch.randn(batch_size, seq_len, 118)
    batch_x_mark = torch.randn(batch_size, seq_len, 4)
    dec_inp = torch.randn(batch_size, pred_len + 48, 118)
    batch_y_mark = torch.randn(batch_size, pred_len + 48, 4)
    
    print("🔥 Testing forward pass...")
    model.eval()
    with torch.no_grad():
        outputs, metadata = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
    if isinstance(outputs, dict):
        print(f"✅ Forward pass successful! Output keys: {list(outputs.keys())}")
        if 'point_prediction' in outputs:
            print(f"   - Point prediction shape: {outputs['point_prediction'].shape}")
    else:
        print(f"✅ Forward pass successful! Output shape: {outputs.shape}")
        
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()