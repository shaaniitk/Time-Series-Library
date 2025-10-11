#!/usr/bin/env python3
"""
Test script to validate Enhanced SOTA PGAT fixes
Tests memory usage, parameter creation, and training stability
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import time
import tracemalloc
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
from data_provider.data_factory import data_provider

class TestConfig:
    """Test configuration"""
    def __init__(self):
        # Use smaller, realistic values for testing
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.c_out = 3
        self.d_model = 64
        self.n_heads = 4
        self.dropout = 0.1
        self.batch_size = 8
        
        # Enhanced PGAT specific
        self.use_multi_scale_patching = True
        self.use_hierarchical_mapper = True
        self.use_stochastic_learner = True
        self.use_gated_graph_combiner = True
        self.use_mixture_decoder = True
        
        # Data settings
        self.data = 'ETTh1'
        self.root_path = './data/ETT/'
        self.data_path = 'ETTh1.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'

def test_memory_usage():
    """Test memory usage and parameter creation"""
    print("🧪 Testing Memory Usage and Parameter Creation")
    print("-" * 50)
    
    config = TestConfig()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Test model initialization
    print("1. Testing model initialization...")
    start_time = time.time()
    
    try:
        model = Enhanced_SOTA_PGAT(config)
        init_time = time.time() - start_time
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✅ Model initialized in {init_time:.2f}s")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {trainable_params:,}")
        
        # Check pre-allocated layers
        projection_layers = len(model.context_projection_layers)
        has_fusion_layer = hasattr(model, 'context_fusion_layer') and model.context_fusion_layer is not None
        
        print(f"   🔧 Pre-allocated projection layers: {projection_layers}")
        print(f"   🔧 Context fusion layer: {'✅' if has_fusion_layer else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        traceback.print_exc()
        return False
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    
    try:
        batch_size = config.batch_size
        seq_len = config.seq_len
        pred_len = config.pred_len
        enc_in = config.enc_in
        
        # Create dummy data
        wave_window = torch.randn(batch_size, seq_len, enc_in)
        target_window = torch.randn(batch_size, pred_len, enc_in)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model(wave_window, target_window)
            forward_time = time.time() - start_time
            
        print(f"   ✅ Forward pass completed in {forward_time:.3f}s")
        
        # Check output format
        if isinstance(outputs, tuple):
            print(f"   📊 Output format: tuple with {len(outputs)} elements")
            if len(outputs) >= 3:
                means, log_stds, log_weights = outputs[:3]
                print(f"   📊 Means shape: {means.shape}")
                print(f"   📊 Log stds shape: {log_stds.shape}")
                print(f"   📊 Log weights shape: {log_weights.shape}")
        else:
            print(f"   📊 Output shape: {outputs.shape}")
            
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False
    
    # Test multiple forward passes (check for memory leaks)
    print("\n3. Testing memory stability...")
    
    try:
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        for i in range(10):
            with torch.no_grad():
                outputs = model(wave_window, target_window)
        
        final_memory = tracemalloc.get_traced_memory()[0]
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"   📊 Memory increase after 10 forward passes: {memory_increase:.2f} MB")
        
        if memory_increase < 10:  # Less than 10MB increase is acceptable
            print("   ✅ Memory usage stable")
        else:
            print("   ⚠️  Potential memory leak detected")
            
    except Exception as e:
        print(f"   ❌ Memory stability test failed: {e}")
        return False
    
    tracemalloc.stop()
    return True

def test_configuration_consistency():
    """Test configuration consistency fixes"""
    print("\n🧪 Testing Configuration Consistency")
    print("-" * 50)
    
    # Test with training-like configuration
    config = TestConfig()
    config.seq_len = 256  # Training sequence length
    config.pred_len = 24  # Training prediction length
    config.d_model = 128  # Training model dimension
    
    try:
        model = Enhanced_SOTA_PGAT(config)
        
        # Check if model uses training config values
        print(f"   📊 Model d_model: {model.d_model}")
        print(f"   📊 Config d_model: {config.d_model}")
        
        if model.d_model == config.d_model:
            print("   ✅ Configuration consistency maintained")
        else:
            print("   ❌ Configuration mismatch detected")
            return False
            
        # Test patch configurations
        if hasattr(model, 'wave_patching_composer') and model.wave_patching_composer:
            print("   ✅ Wave patching composer initialized")
        if hasattr(model, 'target_patching_composer') and model.target_patching_composer:
            print("   ✅ Target patching composer initialized")
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    return True

def test_graph_validation():
    """Test graph component validation"""
    print("\n🧪 Testing Graph Component Validation")
    print("-" * 50)
    
    config = TestConfig()
    
    try:
        model = Enhanced_SOTA_PGAT(config)
        
        # Test graph validation method
        if hasattr(model, '_validate_graph_output'):
            print("   ✅ Graph validation method available")
            
            # Test with valid tuple
            test_tensor = torch.randn(2, 3, 3)
            test_weights = torch.randn(2, 3, 3)
            result = model._validate_graph_output((test_tensor, test_weights), "test_component")
            
            if len(result) == 2:
                print("   ✅ Tuple validation works")
            else:
                print("   ❌ Tuple validation failed")
                return False
                
            # Test with single tensor
            result = model._validate_graph_output(test_tensor, "test_component")
            if len(result) == 2 and result[1] is None:
                print("   ✅ Single tensor validation works")
            else:
                print("   ❌ Single tensor validation failed")
                return False
                
        else:
            print("   ❌ Graph validation method not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Graph validation test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Enhanced SOTA PGAT Fixes Validation")
    print("=" * 60)
    
    tests = [
        ("Memory Usage & Parameter Creation", test_memory_usage),
        ("Configuration Consistency", test_configuration_consistency),
        ("Graph Component Validation", test_graph_validation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<40} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes validated successfully!")
        return 0
    else:
        print("⚠️  Some fixes need attention")
        return 1

if __name__ == "__main__":
    exit(main())