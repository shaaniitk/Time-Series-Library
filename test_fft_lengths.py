#!/usr/bin/env python3
"""
Demonstrate the FFT length mismatch issue and how the fix works.
"""

import torch
import torch.nn.functional as F
import numpy as np

def demonstrate_fft_length_issue():
    """Show that FFT operations can produce inconsistent lengths."""
    print("=== Demonstrating FFT Length Mismatch Issue ===\n")
    
    # Original sequence
    original_length = 96
    x = torch.randn(1, 1, 1, original_length)
    
    print(f"Original length: {original_length}")
    
    # Test different downsampling scales
    scales = [1, 2, 3, 4]
    
    for scale in scales:
        print(f"\n--- Scale {scale} ---")
        
        if scale == 1:
            # No downsampling
            x_scaled = x
        else:
            # Downsample
            x_reshaped = x.reshape(-1, original_length).unsqueeze(1)
            x_down = F.avg_pool1d(x_reshaped, kernel_size=scale, stride=scale)
            downsampled_length = x_down.size(-1)
            x_scaled = x_down.squeeze(1).reshape(1, 1, 1, downsampled_length)
        
        # Apply FFT operations
        x_fft = torch.fft.rfft(x_scaled, dim=-1)
        x_reconstructed = torch.fft.irfft(x_fft, dim=-1)
        
        reconstructed_length = x_reconstructed.size(-1)
        expected_length = x_scaled.size(-1)
        
        print(f"  Input length: {x_scaled.size(-1)}")
        print(f"  FFT freq bins: {x_fft.size(-1)}")
        print(f"  Reconstructed length: {reconstructed_length}")
        print(f"  Expected length: {expected_length}")
        print(f"  Length matches: {reconstructed_length == expected_length}")
        
        # Show what happens when we try to upsample back to original
        if scale > 1:
            # Try to upsample back to original length
            x_reshaped_for_interp = x_reconstructed.reshape(-1, reconstructed_length).unsqueeze(1)
            x_upsampled = F.interpolate(x_reshaped_for_interp, size=original_length, mode='linear', align_corners=False)
            final_length = x_upsampled.size(-1)
            print(f"  After upsampling to original: {final_length}")
            print(f"  Matches original: {final_length == original_length}")

def demonstrate_fix_effectiveness():
    """Show how the fix ensures consistent lengths."""
    print("\n\n=== Demonstrating Fix Effectiveness ===\n")
    
    def apply_correlation_with_fix(x, target_length):
        """Apply correlation with length fix."""
        # Simulate the correlation computation
        x_fft = torch.fft.rfft(x, dim=-1)
        corr = torch.fft.irfft(x_fft, dim=-1)
        
        # Apply the fix
        curr_L = corr.size(-1)
        if curr_L != target_length:
            print(f"  Length mismatch detected: {curr_L} != {target_length}")
            # Reshape for interpolation
            B, H, E, _ = corr.shape
            corr_reshaped = corr.reshape(B * H * E, curr_L).unsqueeze(1)
            corr_upsampled = F.interpolate(corr_reshaped, size=target_length, mode='linear', align_corners=False)
            corr = corr_upsampled.squeeze(1).reshape(B, H, E, target_length)
            print(f"  Fixed to: {corr.size(-1)}")
        else:
            print(f"  No fix needed: {curr_L} == {target_length}")
            
        return corr
    
    target_length = 96
    scales = [1, 2, 3, 4]
    
    for scale in scales:
        print(f"\n--- Testing scale {scale} with fix ---")
        
        # Create input with different length
        if scale == 1:
            input_length = target_length
        else:
            input_length = target_length // scale
            
        x = torch.randn(1, 1, 1, input_length)
        print(f"  Input length: {input_length}")
        
        # Apply correlation with fix
        result = apply_correlation_with_fix(x, target_length)
        print(f"  Final result length: {result.size(-1)}")
        print(f"  Success: {result.size(-1) == target_length}")

if __name__ == "__main__":
    demonstrate_fft_length_issue()
    demonstrate_fix_effectiveness()
