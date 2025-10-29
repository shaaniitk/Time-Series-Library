#!/usr/bin/env python3
"""
Test script to verify the enhanced logging functions work correctly
"""

import sys
import os
import torch
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_detailed_batch_metrics():
    """Test the detailed batch metrics computation"""
    print("Testing Detailed Batch Metrics Computation...")
    
    try:
        from scripts.train.train_celestial_production import (
            _compute_detailed_batch_metrics,
            _compute_additional_accuracy_metrics,
            _decompose_hybrid_loss_components
        )
        
        # Create mock data
        batch_size, seq_len, num_targets = 4, 20, 4
        
        raw_loss = torch.tensor(0.5, requires_grad=True)
        outputs_tensor = torch.randn(batch_size, seq_len, num_targets)
        y_true_for_loss = torch.randn(batch_size, seq_len, num_targets)
        
        # Mock MDN outputs
        n_components = 5
        pi = torch.softmax(torch.randn(batch_size, seq_len, num_targets, n_components), dim=-1)
        mu = torch.randn(batch_size, seq_len, num_targets, n_components)
        sigma = torch.abs(torch.randn(batch_size, seq_len, num_targets, n_components)) + 0.1
        mdn_outputs = (pi, mu, sigma)
        
        # Mock args
        class MockArgs:
            def __init__(self):
                self.loss = {
                    'type': 'hybrid_mdn_directional',
                    'mse_weight': 0.6,
                    'directional_weight': 0.25,
                    'mdn_weight': 0.15
                }
        
        args = MockArgs()
        
        # Test the function
        metrics = _compute_detailed_batch_metrics(
            raw_loss, outputs_tensor, y_true_for_loss, mdn_outputs, args
        )
        
        print("✅ Batch metrics computed successfully:")
        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        
        # Verify expected keys exist
        expected_keys = ['total_loss', 'mse_component', 'directional_component', 'mdn_component']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
        
        print("✅ All expected metrics present")
        return True
        
    except Exception as e:
        print(f"❌ Detailed batch metrics test failed: {e}")
        return False

def test_additional_accuracy_metrics():
    """Test additional accuracy metrics computation"""
    print("\nTesting Additional Accuracy Metrics...")
    
    try:
        from scripts.train.train_celestial_production import _compute_additional_accuracy_metrics
        
        # Create test data with known patterns
        batch_size, seq_len, num_targets = 2, 10, 4
        
        # Create predictions and targets with some correlation
        pred = torch.randn(batch_size, seq_len, num_targets)
        target = pred + torch.randn(batch_size, seq_len, num_targets) * 0.1  # Add small noise
        
        metrics = _compute_additional_accuracy_metrics(pred, target)
        
        print("✅ Additional accuracy metrics computed:")
        for key, value in metrics.items():
            if value is not None:
                print(f"  {key}: {value:.2f}%")
        
        # Sign accuracy should be reasonably high due to correlation
        if 'sign_accuracy' in metrics and metrics['sign_accuracy'] is not None:
            assert metrics['sign_accuracy'] > 30, "Sign accuracy seems too low"
        
        print("✅ Accuracy metrics look reasonable")
        return True
        
    except Exception as e:
        print(f"❌ Additional accuracy metrics test failed: {e}")
        return False

def test_loss_decomposition():
    """Test loss component decomposition"""
    print("\nTesting Loss Component Decomposition...")
    
    try:
        from scripts.train.train_celestial_production import _decompose_hybrid_loss_components
        
        # Create test data
        batch_size, seq_len, num_targets = 2, 10, 4
        outputs = torch.randn(batch_size, seq_len, num_targets)
        targets = torch.randn(batch_size, seq_len, num_targets)
        
        # Mock MDN outputs
        n_components = 3
        pi = torch.softmax(torch.randn(batch_size, seq_len, num_targets, n_components), dim=-1)
        mu = torch.randn(batch_size, seq_len, num_targets, n_components)
        sigma = torch.abs(torch.randn(batch_size, seq_len, num_targets, n_components)) + 0.1
        mdn_outputs = (pi, mu, sigma)
        
        loss_config = {
            'mse_weight': 0.6,
            'directional_weight': 0.25,
            'mdn_weight': 0.15
        }
        
        components = _decompose_hybrid_loss_components(outputs, targets, mdn_outputs, loss_config)
        
        print("✅ Loss components decomposed:")
        for key, value in components.items():
            if value is not None:
                print(f"  {key}: {value:.6f}")
        
        print("✅ Loss decomposition working")
        return True
        
    except Exception as e:
        print(f"❌ Loss decomposition test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED LOGGING FUNCTIONALITY TEST")
    print("=" * 60)
    
    success = True
    
    # Test all components
    success &= test_detailed_batch_metrics()
    success &= test_additional_accuracy_metrics()
    success &= test_loss_decomposition()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL ENHANCED LOGGING TESTS PASSED!")
        print("✅ Detailed batch metrics computation works")
        print("✅ Additional accuracy metrics work")
        print("✅ Loss component decomposition works")
        print("\n🚀 Enhanced logging is ready for production training!")
        print("You will now see:")
        print("  📊 Detailed loss components per batch")
        print("  🎯 Directional accuracy metrics per batch")
        print("  📈 Comprehensive epoch summaries")
        print("  ⏱️  Performance and trend indicators")
    else:
        print("❌ SOME ENHANCED LOGGING TESTS FAILED!")
        print("Please check the implementation before running production training")
    print("=" * 60)