#!/usr/bin/env python3

import time
import torch
import psutil
from datetime import datetime

def diagnose_training_slowdown():
    """Diagnose training slowdown causes"""
    
    print("🔍 TRAINING SLOWDOWN DIAGNOSTICS")
    print("=" * 50)
    
    # Check GPU status
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # Time different initialization phases
    phases = {}
    
    # Phase 1: CUDA initialization
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
    phases['cuda_init'] = time.time() - start
    
    # Phase 2: Model creation
    start = time.time()
    # Simulate model creation
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    if torch.cuda.is_available():
        model = model.cuda()
    phases['model_creation'] = time.time() - start
    
    # Phase 3: First forward pass
    start = time.time()
    x = torch.randn(32, 100)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        _ = model(x)
    phases['first_forward'] = time.time() - start
    
    # Phase 4: Second forward pass
    start = time.time()
    with torch.no_grad():
        _ = model(x)
    phases['second_forward'] = time.time() - start
    
    print("\n⏱️  TIMING RESULTS:")
    for phase, duration in phases.items():
        print(f"  {phase}: {duration:.3f}s")
    
    print(f"\n📊 SPEEDUP RATIO: {phases['first_forward'] / phases['second_forward']:.1f}x")

if __name__ == "__main__":
    diagnose_training_slowdown()