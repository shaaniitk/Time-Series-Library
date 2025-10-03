#!/usr/bin/env python3
"""Check PyTorch installation and recommend appropriate PGAT configuration."""

import torch
import sys
from pathlib import Path

def check_pytorch_setup():
    """Check PyTorch installation and provide recommendations."""
    print("🔍 Checking PyTorch Installation...")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test GPU allocation
        try:
            test_tensor = torch.randn(10, 10).cuda()
            print("✅ GPU allocation test: PASSED")
            config_file = "configs/sota_pgat_synthetic_memory_optimized.yaml"
            batch_size = 16
            use_amp = True
        except Exception as e:
            print(f"❌ GPU allocation test: FAILED ({e})")
            config_file = "configs/sota_pgat_synthetic_cpu.yaml"
            batch_size = 8
            use_amp = False
    else:
        print("ℹ️  CUDA not available - will use CPU")
        config_file = "configs/sota_pgat_synthetic_cpu.yaml"
        batch_size = 8
        use_amp = False
    
    # Check MPS (Apple Silicon) availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 MPS (Apple Silicon) available")
        if not cuda_available:
            config_file = "configs/sota_pgat_synthetic_memory_optimized.yaml"
            batch_size = 12
    
    print("\n📋 Recommended Configuration:")
    print(f"Config file: {config_file}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision: {use_amp}")
    
    print(f"\n🚀 Recommended Command:")
    amp_flag = "" if not use_amp else ""
    disable_amp = "--disable-amp" if not use_amp else ""
    
    print(f"""python scripts/train/train_pgat_synthetic_fixed.py \\
    --config {config_file} \\
    --batch-size {batch_size} \\
    --regenerate-data {disable_amp}""".strip())
    
    return config_file, batch_size, use_amp

if __name__ == "__main__":
    check_pytorch_setup()