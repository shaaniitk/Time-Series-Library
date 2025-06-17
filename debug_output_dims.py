#!/usr/bin/env python3
"""
Debug the output dimension issue in HierarchicalEnhancedAutoformer.
"""

import torch
import torch.nn.functional as F
from models.HierarchicalEnhancedAutoformer import HierarchicalEnhancedAutoformer
from argparse import Namespace

def debug_output_dimensions():
    """Debug the dimension flow through HierarchicalEnhancedAutoformer."""
    
    # Create test config
    configs = Namespace(
        task_name='long_term_forecast',
        seq_len=96,
        label_len=48, 
        pred_len=12,  # Expected output length
        enc_in=7,
        dec_in=4, 
        c_out=4,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        factor=1,
        dropout=0.1,
        embed='timeF',
        freq='h',
        activation='gelu'
    )
    
    print(f"Expected output shape: (batch_size, {configs.pred_len}, {configs.c_out})")
    
    # Create model
    model = HierarchicalEnhancedAutoformer(configs, n_levels=3)
    
    # Create test inputs
    batch_size = 2
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
    
    print(f"\nInput shapes:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    print(f"  Expected decoder input length: {configs.label_len + configs.pred_len} = {configs.label_len} + {configs.pred_len}")
    
    # Hook into intermediate outputs to debug
    intermediate_shapes = {}
    
    def hook_decomposer(module, input, output):
        intermediate_shapes['decomposer'] = [f.shape for f in output]
    
    def hook_encoder(module, input, output):
        intermediate_shapes['encoder'] = [f.shape for f in output]
    
    def hook_decoder(module, input, output):
        intermediate_shapes['decoder'] = (output[0].shape, output[1].shape)
        
    def hook_fusion(module, input, output):
        intermediate_shapes['fusion_seasonal'] = output.shape if hasattr(output, 'shape') else 'no shape'
    
    # Register hooks
    model.decomposer.register_forward_hook(hook_decomposer)
    model.encoder.register_forward_hook(hook_encoder)
    model.decoder.register_forward_hook(hook_decoder)
    
    # Forward pass
    try:
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"\nActual output shape: {output.shape}")
        print(f"Expected: (2, {configs.pred_len}, {configs.c_out})")
        
        print(f"\nIntermediate shapes:")
        for name, shape in intermediate_shapes.items():
            print(f"  {name}: {shape}")
            
        # Check if the issue is in the final slicing
        if output.shape[1] != configs.pred_len:
            print(f"\n❌ Length mismatch: got {output.shape[1]}, expected {configs.pred_len}")
            print("This suggests an issue in:")
            print("  1. Fusion target_length parameter")
            print("  2. Final slicing logic") 
            print("  3. Multi-resolution length handling")
        else:
            print(f"\n✅ Output length correct!")
            
    except Exception as e:
        print(f"\n❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_output_dimensions()
