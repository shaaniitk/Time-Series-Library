#!/usr/bin/env python3
"""
Attention Components Test Script

Tests all attention mechanisms in the modular framework:
- AUTOCORRELATION_LAYER
- ADAPTIVE_AUTOCORRELATION_LAYER  
- CROSS_RESOLUTION
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType

def test_attention_components():
    """Test all attention components."""
    print("=== ATTENTION COMPONENTS TEST ===")
    
    # Ensure components are registered
    register_all_components()
    
    # Test parameters
    batch_size = 2
    seq_len = 96
    d_model = 64
    n_heads = 8
    
    test_input = torch.randn(batch_size, seq_len, d_model)
    test_cross = torch.randn(batch_size, seq_len, d_model)
    
    attention_types = [
        ComponentType.AUTOCORRELATION_LAYER,
        ComponentType.ADAPTIVE_AUTOCORRELATION_LAYER,
        ComponentType.CROSS_RESOLUTION
    ]
    
    results = {}
    
    for attention_type in attention_types:
        print(f"\nTesting {attention_type.value}...")
        
        try:
            # Get component
            component_info = component_registry.get_component(attention_type)
            assert component_info is not None
            
            # Create instance
            params = {
                'd_model': d_model,
                'n_heads': n_heads,
                'dropout': 0.1,
                'factor': 1
            }
            
            component = component_info.factory(**params)
            
            # Test forward pass
            with torch.no_grad():
                if attention_type == ComponentType.CROSS_RESOLUTION:
                    output, weights = component(test_input, test_cross, test_cross)
                else:
                    output, weights = component(test_input, test_input, test_input)
            
            # Validate
            assert output.shape == test_input.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            print(f"✅ {attention_type.value} - PASSED")
            results[attention_type.value] = "PASSED"
            
        except Exception as e:
            print(f"❌ {attention_type.value} - FAILED: {e}")
            results[attention_type.value] = f"FAILED: {e}"
    
    # Summary
    print(f"\n=== ATTENTION COMPONENTS SUMMARY ===")
    passed = sum(1 for status in results.values() if status == "PASSED")
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for comp_type, status in results.items():
        icon = "✅" if status == "PASSED" else "❌"
        print(f"{icon} {comp_type}: {status}")
    
    return passed == total

if __name__ == "__main__":
    success = test_attention_components()
    sys.exit(0 if success else 1)
