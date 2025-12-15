#!/usr/bin/env python3
"""
AMD Rembrandt GPU Fix for PyTorch ROCm
This script tries different compatibility settings for AMD Rembrandt GPUs
"""

import os
import torch

def try_amd_gpu_fix():
    """Try different AMD GPU compatibility fixes"""
    
    print("=== AMD Rembrandt GPU Fix ===")
    
    # Common fixes for AMD Rembrandt (RDNA2) GPUs
    fixes = [
        {
            'name': 'Standard Rembrandt Fix',
            'env': {
                'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
                'AMD_SERIALIZE_KERNEL': '3',
                'HIP_VISIBLE_DEVICES': '0',
                'ROCR_VISIBLE_DEVICES': '0'
            }
        },
        {
            'name': 'Alternative GFX Version',
            'env': {
                'HSA_OVERRIDE_GFX_VERSION': '11.0.0',
                'AMD_SERIALIZE_KERNEL': '3',
                'HIP_VISIBLE_DEVICES': '0'
            }
        },
        {
            'name': 'RDNA2 Compatibility',
            'env': {
                'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
                'AMD_SERIALIZE_KERNEL': '3',
                'HIP_FORCE_DEV_KERNARG': '1',
                'AMD_LOG_LEVEL': '1'
            }
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"\n--- Trying Fix {i}: {fix['name']} ---")
        
        # Set environment variables
        for key, value in fix['env'].items():
            os.environ[key] = value
            print(f"Set {key}={value}")
        
        # Test the fix
        if test_gpu_operations():
            print(f"🎉 SUCCESS! Fix {i} worked!")
            print("\nTo make this permanent, add these to your shell profile:")
            for key, value in fix['env'].items():
                print(f"export {key}={value}")
            return True
        else:
            print(f"❌ Fix {i} didn't work, trying next...")
    
    print("\n😞 None of the standard fixes worked.")
    print("You may need to:")
    print("1. Update ROCm drivers")
    print("2. Install a different PyTorch version")
    print("3. Use CPU for now")
    return False

def test_gpu_operations():
    """Test if GPU operations work without errors"""
    try:
        if not torch.cuda.is_available():
            return False
        
        device = torch.device("cuda:0")
        
        # Test 1: Basic tensor creation
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        
        # Test 2: Tensor operations
        y = x * 2
        
        # Test 3: Matrix operations
        a = torch.randn(2, 2, device=device)
        b = torch.randn(2, 2, device=device)
        c = torch.matmul(a, b)
        
        # Test 4: Move to CPU (this often triggers the error)
        result = c.cpu()
        
        # Test 5: Print tensor (another common failure point)
        _ = str(result)
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)[:100]}...")
        return False

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    try_amd_gpu_fix()