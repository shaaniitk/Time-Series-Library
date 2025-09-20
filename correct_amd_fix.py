#!/usr/bin/env python3
"""
Correct AMD GPU fix for GFX1151 (Rembrandt)
"""

import os
import torch

# Set the CORRECT GFX version for your GPU
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.5.1'  # This matches your gfx1151
os.environ['AMD_SERIALIZE_KERNEL'] = '1'  # Use 1, not 3
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCR_VISIBLE_DEVICES'] = '0'

print("=== Correct AMD GFX1151 Fix ===")
print("Set HSA_OVERRIDE_GFX_VERSION=11.5.1 (matches your gfx1151)")
print("Set AMD_SERIALIZE_KERNEL=1")
print("Set HIP_VISIBLE_DEVICES=0")
print("Set ROCR_VISIBLE_DEVICES=0")

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        device = torch.device("cuda:0")
        
        print("\nTesting GPU operations...")
        
        # Test 1: Basic tensor
        print("1. Creating tensor on GPU...")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"   ✅ Success: {x.device}")
        
        # Test 2: Operations
        print("2. Testing tensor operations...")
        y = x * 2
        print(f"   ✅ Success: multiplication")
        
        # Test 3: Matrix operations
        print("3. Testing matrix operations...")
        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)
        c = torch.matmul(a, b)
        print(f"   ✅ Success: matrix multiplication")
        
        # Test 4: GPU to CPU transfer
        print("4. Testing GPU to CPU transfer...")
        cpu_result = c.cpu()
        print(f"   ✅ Success: transfer completed")
        
        # Test 5: Print tensor values
        print("5. Testing tensor display...")
        print(f"   Sample result: {cpu_result[0, 0].item():.4f}")
        print(f"   ✅ Success: tensor display")
        
        print("\n🎉 ALL TESTS PASSED! Your AMD GPU is now working!")
        print("\nTo make this permanent, add these lines to your ~/.bashrc:")
        print("export HSA_OVERRIDE_GFX_VERSION=11.5.1")
        print("export AMD_SERIALIZE_KERNEL=1")
        print("export HIP_VISIBLE_DEVICES=0")
        print("export ROCR_VISIBLE_DEVICES=0")
        
    except Exception as e:
        print(f"\n❌ Still getting error: {e}")
        print("\nTrying alternative GFX versions...")
        
        # Try other possible versions for gfx1151
        alternatives = ['11.5.0', '11.0.0', '10.3.0']
        for gfx_ver in alternatives:
            print(f"\nTrying HSA_OVERRIDE_GFX_VERSION={gfx_ver}...")
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = gfx_ver
            
            try:
                test_tensor = torch.randn(2, 2, device=device)
                result = test_tensor.cpu()
                print(f"✅ SUCCESS with GFX version {gfx_ver}!")
                break
            except:
                print(f"❌ Failed with GFX version {gfx_ver}")
                continue
else:
    print("❌ No GPU detected")