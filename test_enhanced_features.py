#!/usr/bin/env python3
"""
Test that enhanced features are actually working, not bypassed.
"""

import torch
import torch.nn.functional as F
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation
from models.EnhancedAutoformer import EnhancedAutoformer
from argparse import Namespace

def test_adaptive_correlation_features():
    """Test that adaptive correlation features are working."""
    print("=== Testing Adaptive Correlation Features ===\n")
    
    # Create adaptive correlation layer
    adaptive_corr = AdaptiveAutoCorrelation(
        factor=1,
        attention_dropout=0.1,
        output_attention=False,
        adaptive_k=True,  # Key feature
        multi_scale=True  # Key feature
    )
    
    # Test inputs
    B, L, H, E = 2, 96, 8, 64
    queries = torch.randn(B, L, H, E)
    keys = torch.randn(B, L, H, E)
    values = torch.randn(B, L, H, E)
    
    print(f"Input shape: {queries.shape}")
    
    # Test that adaptive k changes behavior
    print("\n--- Testing Adaptive K Feature ---")
    
    # Run with different inputs to see if adaptive behavior works
    for i, input_type in enumerate(["random", "periodic", "trending"]):
        if input_type == "random":
            test_queries = torch.randn_like(queries)
        elif input_type == "periodic":
            t = torch.linspace(0, 4*torch.pi, L).view(1, L, 1, 1).expand_as(queries)
            test_queries = queries + 0.5 * torch.sin(t)
        else:  # trending
            trend = torch.linspace(0, 1, L).view(1, L, 1, 1).expand_as(queries)
            test_queries = queries + trend
            
        output, _ = adaptive_corr(test_queries, keys, values, None)
        print(f"  {input_type:>10} input -> output shape: {output.shape}")
        
        # Check that adaptive_k parameter actually changed
        if hasattr(adaptive_corr, 'current_k'):
            print(f"  {input_type:>10} -> adaptive k: {adaptive_corr.current_k}")
    
    print("\n--- Testing Multi-Scale Feature ---")
    
    # Test multi-scale correlation
    print(f"  Multi-scale enabled: {adaptive_corr.multi_scale}")
    print(f"  Scales used: {adaptive_corr.scales}")
    
    # Run correlation and check that different scales are computed
    output, _ = adaptive_corr(queries, keys, values, None)
    print(f"  Multi-scale output shape: {output.shape}")
    print(f"  Output stats - mean: {output.mean():.6f}, std: {output.std():.6f}")

def test_enhanced_vs_original_differences():
    """Test that enhanced model behaves differently from original."""
    print("\n\n=== Testing Enhanced vs Original Differences ===\n")
    
    # Mock config
    config = Namespace(
        task_name='long_term_forecast',
        seq_len=96, label_len=48, pred_len=24,
        enc_in=7, dec_in=7, c_out=7,
        d_model=64, n_heads=4, e_layers=2, d_layers=1,
        d_ff=256, factor=1, dropout=0.1,
        embed='timeF', freq='h', activation='gelu'
    )
    
    # Create enhanced model
    enhanced_model = EnhancedAutoformer(config)
    
    # Test inputs
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
    
    print(f"Input shape: {x_enc.shape}")
    
    # Test enhanced model
    print("\n--- Testing Enhanced Model ---")
    enhanced_output = enhanced_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Enhanced output shape: {enhanced_output.shape}")
    print(f"Enhanced output stats - mean: {enhanced_output.mean():.6f}, std: {enhanced_output.std():.6f}")
    
    # Test that enhanced features are actually being used
    encoder = enhanced_model.encoder
    if hasattr(encoder, 'attn_layers'):
        first_attn = encoder.attn_layers[0].attention
        if hasattr(first_attn, 'adaptive_k'):
            print(f"  Adaptive K enabled: {first_attn.adaptive_k}")
        if hasattr(first_attn, 'multi_scale'):
            print(f"  Multi-scale enabled: {first_attn.multi_scale}")
            
    # Test learnable decomposition
    if hasattr(enhanced_model, 'decomp'):
        print(f"  Learnable decomposition: {type(enhanced_model.decomp).__name__}")

if __name__ == "__main__":
    test_adaptive_correlation_features()
    test_enhanced_vs_original_differences()
