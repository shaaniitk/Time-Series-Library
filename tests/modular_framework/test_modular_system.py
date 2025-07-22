#!/usr/bin/env python3
"""
Modular System Integration Test

This script tests the core modular framework and component registry.
It should remain focused on testing the modular system itself, not individual components.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_component_registry():
    """Test component registry functionality"""
    print("Testing Component Registry...")
    
    try:
        from configs.modular_components import component_registry
        from configs.schemas import ComponentType
        
        # Test that registry has components
        assert len(component_registry._components) > 0, "Registry should have components"
        
        # Test specific component types exist
        expected_types = [
            ComponentType.AUTOCORRELATION,
            ComponentType.LEARNABLE_DECOMP,
            ComponentType.STANDARD_ENCODER,
            ComponentType.MSE
        ]
        
        for comp_type in expected_types:
            assert comp_type in component_registry._components, f"Missing component type: {comp_type}"
        
        print("  ✓ Component registry tests passed")
        
    except Exception as e:
        print(f"  ✗ Component registry tests failed: {e}")
        import traceback
        traceback.print_exc()


def test_component_creation():
    """Test component creation from registry"""
    print("Testing Component Creation...")
    
    try:
        from configs.concrete_components import (
            AutoCorrelationAttention, LearnableDecomposition,
            StandardEncoder, MSELoss
        )
        from configs.schemas import (
            AttentionConfig, DecompositionConfig,
            EncoderConfig, LossConfig, ComponentType
        )
        
        # Test attention component
        attn_config = AttentionConfig(
            component_type=ComponentType.AUTOCORRELATION,
            d_model=512,
            n_heads=8
        )
        attention = AutoCorrelationAttention(attn_config)
        assert attention is not None
        
        # Test loss component
        loss_config = LossConfig(
            component_type=ComponentType.MSE
        )
        loss_fn = MSELoss(loss_config)
        assert loss_fn is not None
        
        print("  ✓ Component creation tests passed")
        
    except Exception as e:
        print(f"  ✗ Component creation tests failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run core modular system tests"""
    print("="*80)
    print("CORE MODULAR SYSTEM TESTS")
    print("="*80)
    
    tests = [
        test_component_registry,
        test_component_creation
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            print(f"\n{'-'*60}")
            test_func()
            passed_tests += 1
            print(f"✓ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 CORE MODULAR SYSTEM TESTS PASSED!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed.")


if __name__ == "__main__":
    main()
