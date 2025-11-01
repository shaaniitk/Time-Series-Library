#!/usr/bin/env python3
"""
Test script to verify MDN mixture mean computation fix
"""

import torch

def test_mdn_mixture_mean():
    """Test the corrected MDN mixture mean computation"""
    
    print("🧪 Testing MDN Mixture Mean Computation Fix")
    print("=" * 50)
    
    # Create test tensors with the shapes from the error
    batch_size = 2
    seq_len = 12
    num_targets = 4
    num_components = 3
    
    # MDN outputs: pi (weights), mu (means), sigma (std devs)
    pi = torch.randn(batch_size, seq_len, num_targets, num_components)
    mu = torch.randn(batch_size, seq_len, num_targets, num_components)
    sigma = torch.abs(torch.randn(batch_size, seq_len, num_targets, num_components)) + 0.1
    
    print(f"Input shapes:")
    print(f"  pi (weights): {pi.shape}")
    print(f"  mu (means): {mu.shape}")
    print(f"  sigma (stds): {sigma.shape}")
    
    # Test the CORRECTED computation
    print(f"\n✅ Testing CORRECTED computation:")
    try:
        w = torch.softmax(pi, dim=-1)
        print(f"  w (softmax weights): {w.shape}")
        
        if mu.dim() == 4:
            # CORRECTED: Direct multiplication and sum over last dimension
            mix_mean = torch.sum(w * mu, dim=-1)  # (batch, seq, targets)
            print(f"  mixture_mean: {mix_mean.shape}")
            print(f"  mixture_mean stats: mean={mix_mean.mean():.4f}, std={mix_mean.std():.4f}")
            print("  ✅ SUCCESS: Corrected computation works!")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Test the BROKEN computation (for comparison)
    print(f"\n❌ Testing BROKEN computation (for comparison):")
    try:
        w = torch.softmax(pi, dim=-1)
        if mu.dim() == 4:
            # BROKEN: Incorrect unsqueeze
            w_exp = w.unsqueeze(2)  # This creates wrong dimensions!
            print(f"  w_expanded (BROKEN): {w_exp.shape}")
            mix_mean_broken = torch.sum(w_exp * mu, dim=-1)
            print(f"  This should fail with dimension mismatch...")
            
    except Exception as e:
        print(f"  ❌ EXPECTED FAILURE: {e}")
        print("  This confirms the original bug!")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"  The fix removes the incorrect .unsqueeze(2) operation")
    print(f"  Both w and mu have shape (batch, seq, targets, components)")
    print(f"  Direct multiplication works: w * mu")
    print(f"  Sum over last dimension gives mixture mean: (batch, seq, targets)")

if __name__ == "__main__":
    test_mdn_mixture_mean()