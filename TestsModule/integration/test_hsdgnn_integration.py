#!/usr/bin/env python3
"""
Smoke tests for HSDGNN integration with Wave-Stock architecture
Tests component functionality and integration points
"""

import torch
import torch.nn as nn
import sys
import traceback
from typing import Dict, Any

# Import HSDGNN components
try:
    from layers.HSDGNNComponents import (
        IntraDependencyLearning, 
        HierarchicalSpatiotemporalBlock,
        HSDGNNResidualPredictor
    )
    HSDGNN_AVAILABLE = True
except ImportError as e:
    print(f"❌ HSDGNN components not available: {e}")
    HSDGNN_AVAILABLE = False

class TestConfig:
    """Test configuration"""
    seq_len = 20
    pred_len = 5
    n_waves = 4
    wave_features = 4
    d_model = 32
    rnn_units = 16
    batch_size = 2
    dropout = 0.1
    window_size = 10
    threshold = 0.5

def test_intra_dependency_learning():
    """Test IntraDependencyLearning component"""
    print("\n🧪 Testing IntraDependencyLearning...")
    
    config = TestConfig()
    
    try:
        # Initialize component
        intra_dep = IntraDependencyLearning(
            n_attributes=config.wave_features,
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # Test input: single wave [B, L, 4]
        wave_data = torch.randn(config.batch_size, config.seq_len, config.wave_features)
        
        # Forward pass
        output = intra_dep(wave_data)
        
        # Check output shape
        expected_shape = (config.batch_size, config.seq_len, config.d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        print(f"✅ IntraDependencyLearning: {wave_data.shape} → {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntraDependencyLearning failed: {e}")
        traceback.print_exc()
        return False

def test_hierarchical_spatiotemporal_block():
    """Test HierarchicalSpatiotemporalBlock component"""
    print("\n🧪 Testing HierarchicalSpatiotemporalBlock...")
    
    config = TestConfig()
    
    try:
        # Initialize component
        hst_block = HierarchicalSpatiotemporalBlock(
            n_waves=config.n_waves,
            wave_features=config.wave_features,
            d_model=config.d_model,
            rnn_units=config.rnn_units,
            seq_len=config.seq_len,
            window_size=config.window_size,
            threshold=config.threshold
        )
        
        # Test input: wave data [B, L, N_waves, wave_features]
        wave_data = torch.randn(config.batch_size, config.seq_len, config.n_waves, config.wave_features)
        
        # Forward pass
        output = hst_block(wave_data)
        
        # Check output shape
        expected_shape = (config.batch_size, config.seq_len, config.n_waves, config.rnn_units)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        print(f"✅ HierarchicalSpatiotemporalBlock: {wave_data.shape} → {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ HierarchicalSpatiotemporalBlock failed: {e}")
        traceback.print_exc()
        return False

def test_hsdgnn_residual_predictor():
    """Test HSDGNNResidualPredictor component"""
    print("\n🧪 Testing HSDGNNResidualPredictor...")
    
    config = TestConfig()
    
    try:
        # Initialize component
        predictor = HSDGNNResidualPredictor(
            n_waves=config.n_waves,
            wave_features=config.wave_features,
            d_model=config.d_model,
            rnn_units=config.rnn_units,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            n_blocks=2,  # Reduced for testing
            window_size=config.window_size,
            threshold=config.threshold
        )
        
        # Test input: wave data [B, L, N_waves, wave_features]
        wave_data = torch.randn(config.batch_size, config.seq_len, config.n_waves, config.wave_features)
        
        # Forward pass
        output = predictor(wave_data)
        
        # Check output shape
        expected_shape = (config.batch_size, config.pred_len, config.n_waves, 1)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        print(f"✅ HSDGNNResidualPredictor: {wave_data.shape} → {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ HSDGNNResidualPredictor failed: {e}")
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test gradient flow through HSDGNN components"""
    print("\n🧪 Testing gradient flow...")
    
    config = TestConfig()
    
    try:
        # Create a simple model with HSDGNN block
        model = HierarchicalSpatiotemporalBlock(
            n_waves=config.n_waves,
            wave_features=config.wave_features,
            d_model=config.d_model,
            rnn_units=config.rnn_units,
            seq_len=config.seq_len,
            window_size=config.window_size,
            threshold=config.threshold
        )
        
        # Test input
        wave_data = torch.randn(config.batch_size, config.seq_len, config.n_waves, config.wave_features, requires_grad=True)
        
        # Forward pass
        output = model(wave_data)
        
        # Compute dummy loss
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert wave_data.grad is not None, "Input gradients are None"
        assert not torch.isnan(wave_data.grad).any(), "Input gradients contain NaN"
        
        # Check model parameter gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
        
        assert grad_count > 0, "No model parameters have gradients"
        
        print(f"✅ Gradient flow: {grad_count} parameters have valid gradients")
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage and efficiency"""
    print("\n🧪 Testing memory usage...")
    
    config = TestConfig()
    
    try:
        # Create model
        model = HierarchicalSpatiotemporalBlock(
            n_waves=config.n_waves,
            wave_features=config.wave_features,
            d_model=config.d_model,
            rnn_units=config.rnn_units,
            seq_len=config.seq_len,
            window_size=config.window_size,
            threshold=config.threshold
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Memory usage: {total_params:,} total params, {trainable_params:,} trainable")
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            wave_data = torch.randn(batch_size, config.seq_len, config.n_waves, config.wave_features)
            output = model(wave_data)
            expected_shape = (batch_size, config.seq_len, config.n_waves, config.rnn_units)
            assert output.shape == expected_shape, f"Batch size {batch_size}: expected {expected_shape}, got {output.shape}"
        
        print("✅ Memory usage: Scales correctly with batch size")
        return True
        
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        traceback.print_exc()
        return False

def run_smoke_tests() -> Dict[str, bool]:
    """Run all smoke tests"""
    print("🚀 Starting HSDGNN Integration Smoke Tests")
    print("=" * 50)
    
    if not HSDGNN_AVAILABLE:
        print("❌ HSDGNN components not available. Skipping tests.")
        return {}
    
    tests = {
        "IntraDependencyLearning": test_intra_dependency_learning,
        "HierarchicalSpatiotemporalBlock": test_hierarchical_spatiotemporal_block,
        "HSDGNNResidualPredictor": test_hsdgnn_residual_predictor,
        "GradientFlow": test_gradient_flow,
        "MemoryUsage": test_memory_usage,
    }
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
            if results[test_name]:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    if passed == total:
        print("\n🎉 All tests passed! HSDGNN integration is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")
    
    return results

if __name__ == "__main__":
    results = run_smoke_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)
    else:
        sys.exit(0)
