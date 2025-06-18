#!/usr/bin/env python3
"""
Analyze the MultiWaveletCross integration issue and propose proper fix.
"""

import torch
import torch.nn as nn
from layers.MultiWaveletCorrelation import MultiWaveletCross

def analyze_multiwavelet_interface():
    """Analyze what MultiWaveletCross actually expects."""
    print("=== MultiWaveletCross Interface Analysis ===\n")
    
    # Current architecture expectations
    print("Current HierarchicalEnhancedAutoformer:")
    print("  - Input format: (B, N, D) - 3D tensors")
    print("  - D = d_model (e.g., 512)")
    print("  - Standard transformer-like architecture")
    
    print("\nMultiWaveletCross expectations:")
    print("  - Input format: (B, N, H, E) - 4D tensors") 
    print("  - H = number of heads, E = dimension per head")
    print("  - H * E = total dimension")
    print("  - Specific wavelet processing architecture")
    
    # Test with actual MultiWaveletCross
    try:
        # Parameters that MultiWaveletCross expects
        test_cross = MultiWaveletCross(
            in_channels=16,  # This is actually H*E, not total channels
            out_channels=16,
            seq_len_q=96,
            seq_len_kv=96,
            modes=32,
            c=64,
            k=8,
            ich=512,  # This should be our d_model
            base='legendre',
            activation='tanh'  # Use supported activation
        )
        print(f"\nMultiWaveletCross created successfully")
        print(f"  - in_channels: 16 (H*E)")
        print(f"  - ich: 512 (our d_model)")
        
        # Test the expected input format
        B, N, H, E = 2, 96, 8, 64  # H*E = 512 = d_model
        q = torch.randn(B, N, H, E)
        k = torch.randn(B, N, H, E) 
        v = torch.randn(B, N, H, E)
        
        print(f"\nTesting with proper 4D format:")
        print(f"  Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # This should work
        output, _ = test_cross(q, k, v)
        print(f"  Output shape: {output.shape}")
        print("  ✅ MultiWaveletCross works with proper 4D format!")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

def test_proper_integration():
    """Test how to properly integrate MultiWaveletCross."""
    print("\n\n=== Proper MultiWaveletCross Integration ===\n")
    
    # Simulate our current 3D tensor
    B, N, D = 2, 96, 512
    features_3d = torch.randn(B, N, D)
    print(f"Our 3D features: {features_3d.shape}")
    
    # Method 1: Reshape to 4D with fixed heads
    n_heads = 8
    head_dim = D // n_heads
    features_4d = features_3d.view(B, N, n_heads, head_dim)
    print(f"Reshaped to 4D: {features_4d.shape}")
    
    # Method 2: Use learnable projections to get proper dimensions
    ich = D  # 512
    in_channels = head_dim  # 64 (this is what MultiWaveletCross expects)
    
    print(f"\nProper parameters for MultiWaveletCross:")
    print(f"  - ich (total dim): {ich}")
    print(f"  - in_channels (per head): {in_channels}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - head_dim: {head_dim}")
    
    try:
        # Create MultiWaveletCross with proper parameters
        multiwavelet_cross = MultiWaveletCross(
            in_channels=in_channels,  # 64 per head
            out_channels=in_channels, # 64 per head  
            seq_len_q=N,
            seq_len_kv=N,
            modes=min(32, in_channels//2),
            c=64,
            k=8,
            ich=ich,  # 512 total
            base='legendre',
            activation='tanh'  # Use supported activation
        )
        
        # Test with proper format
        output, _ = multiwavelet_cross(features_4d, features_4d, features_4d)
        output_3d = output.view(B, N, D)  # Reshape back to 3D
        
        print(f"\n✅ Success!")
        print(f"  Input (4D): {features_4d.shape}")
        print(f"  Output (3D): {output_3d.shape}")
        print(f"  Shape preserved: {output_3d.shape == features_3d.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    analyze_multiwavelet_interface()
    success = test_proper_integration()
    
    if success:
        print("\n🎯 CONCLUSION: MultiWaveletCross CAN be properly integrated!")
        print("   The issue was incorrect parameter configuration, not fundamental incompatibility.")
    else:
        print("\n⚠️  CONCLUSION: MultiWaveletCross integration needs more work.")
