#!/usr/bin/env python3
"""
Debug the multi-resolution length issue in detail.
"""

import torch
import torch.nn.functional as F
from models.HierarchicalEnhancedAutoformer import WaveletHierarchicalDecomposer

def debug_decomposer_lengths():
    """Debug the decomposer to see what lengths it's producing."""
    
    # Test with the same config as the failing test
    seq_len = 24
    d_model = 64
    n_levels = 3
    
    decomposer = WaveletHierarchicalDecomposer(
        seq_len=seq_len,
        d_model=d_model,
        wavelet_type='db4',
        levels=n_levels
    )
    
    # Create test input
    batch_size = 2
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input: {x.shape}")
    
    # Decompose
    multi_res_features = decomposer(x)
    
    print(f"\nDecomposer output lengths:")
    for i, features in enumerate(multi_res_features):
        print(f"  Level {i}: {features.shape}")
    
    # The issue: we expect pred_len=12, but we're getting much shorter sequences
    pred_len = 12
    label_len = 12
    
    print(f"\nTarget lengths:")
    print(f"  pred_len: {pred_len}")
    print(f"  label_len: {label_len}")
    print(f"  total decoder length: {label_len + pred_len}")
    
    # Check what happens when we resize these to pred_len
    print(f"\nIf we resize to pred_len ({pred_len}):")
    for i, features in enumerate(multi_res_features):
        if features.size(1) != pred_len:
            resized = F.interpolate(
                features.transpose(1, 2),
                size=pred_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            print(f"  Level {i}: {features.shape} → {resized.shape}")
        else:
            print(f"  Level {i}: {features.shape} (no resize needed)")

if __name__ == "__main__":
    debug_decomposer_lengths()
