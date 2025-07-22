#!/usr/bin/env python3
"""
Decomposition Components Test Script

Tests all decomposition mechanisms in the modular framework:
- SERIES_DECOMP
- STABLE_DECOMP
- LEARNABLE_DECOMP
- WAVELET_DECOMP
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

def test_decomposition_components():
    """Test all decomposition components."""
    print("=== DECOMPOSITION COMPONENTS TEST ===")
    
    # Ensure components are registered
    register_all_components()
    
    # Test parameters
    batch_size = 2
    seq_len = 96
    d_model = 64
    
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    decomp_types = [
        ComponentType.SERIES_DECOMP,
        ComponentType.STABLE_DECOMP,
        ComponentType.LEARNABLE_DECOMP,
        ComponentType.WAVELET_DECOMP
    ]
    
    results = {}
    
    for decomp_type in decomp_types:
        print(f"\nTesting {decomp_type.value}...")
        
        try:
            # Get component
            component_info = component_registry.get_component(decomp_type)
            assert component_info is not None
            
            # Create instance with appropriate parameters
            if decomp_type in [ComponentType.SERIES_DECOMP, ComponentType.STABLE_DECOMP]:
                params = {'kernel_size': 25}
            elif decomp_type == ComponentType.LEARNABLE_DECOMP:
                params = {'input_dim': d_model}
            elif decomp_type == ComponentType.WAVELET_DECOMP:
                params = {'seq_len': seq_len, 'd_model': d_model}
            
            component = component_info.factory(**params)
            
            # Test forward pass
            with torch.no_grad():
                seasonal, trend = component(test_input)
            
            # Validate
            assert seasonal.shape == test_input.shape
            assert trend.shape == test_input.shape
            assert not torch.isnan(seasonal).any()
            assert not torch.isnan(trend).any()
            
            # Test reconstruction
            reconstruction = seasonal + trend
            assert reconstruction.shape == test_input.shape
            
            print(f"✅ {decomp_type.value} - PASSED")
            results[decomp_type.value] = "PASSED"
            
        except Exception as e:
            print(f"❌ {decomp_type.value} - FAILED: {e}")
            results[decomp_type.value] = f"FAILED: {e}"
    
    # Summary
    print(f"\n=== DECOMPOSITION COMPONENTS SUMMARY ===")
    passed = sum(1 for status in results.values() if status == "PASSED")
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for comp_type, status in results.items():
        icon = "✅" if status == "PASSED" else "❌"
        print(f"{icon} {comp_type}: {status}")
    
    return passed == total

if __name__ == "__main__":
    success = test_decomposition_components()
    sys.exit(0 if success else 1)
