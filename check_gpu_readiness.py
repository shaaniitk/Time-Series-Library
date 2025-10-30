#!/usr/bin/env python3
"""
Check GPU readiness for training
"""

import torch
import sys

def check_gpu_readiness():
    print("🔍 GPU READINESS CHECK")
    print("=" * 40)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU tensor operations
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✅ GPU tensor operations working")
            print(f"✅ Ready for GPU training!")
            return True
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            return False
    else:
        print("❌ CUDA not available - will fall back to CPU")
        return False

if __name__ == "__main__":
    success = check_gpu_readiness()
    sys.exit(0 if success else 1)