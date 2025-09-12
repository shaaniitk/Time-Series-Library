#!/usr/bin/env python3
"""Simple test script to verify fusion registry implementation without PyTorch."""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports and registry setup."""
    print("Testing basic imports...")
    
    try:
        # Import core components
        from layers.modular.core import unified_registry, ComponentFamily
        print("✓ Successfully imported core registry components")
        
        # Check if FUSION component family exists
        assert hasattr(ComponentFamily, 'FUSION'), "FUSION component family not found"
        print(f"✓ FUSION component family exists: {ComponentFamily.FUSION.value}")
        
        # Import component registration
        import layers.modular.core.register_components
        print("✓ Successfully imported component registration")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during basic imports: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registry_structure():
    """Test registry structure without creating components."""
    print("\nTesting registry structure...")
    
    try:
        from layers.modular.core import unified_registry, ComponentFamily
        import layers.modular.core.register_components
        
        # Check registered fusion components
        fusion_components = unified_registry.get_all_by_type(ComponentFamily.FUSION)
        print(f"✓ Found {len(fusion_components)} fusion components: {list(fusion_components.keys())}")
        
        # Check if HierarchicalFusion is registered
        if 'hierarchical_fusion' in fusion_components:
            print("✓ HierarchicalFusion is registered")
            component_info = fusion_components['hierarchical_fusion']
            print(f"✓ Component class: {component_info['class'].__name__}")
            print(f"✓ Component config keys: {list(component_info.get('config', {}).keys())}")
        else:
            print("✗ HierarchicalFusion not found in registry")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during registry structure test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_class_import():
    """Test importing the HierarchicalFusion class directly."""
    print("\nTesting component class import...")
    
    try:
        from layers.modular.fusion.hierarchical_fusion import HierarchicalFusion
        print("✓ Successfully imported HierarchicalFusion class")
        
        from layers.modular.fusion.base import BaseFusion
        print("✓ Successfully imported BaseFusion base class")
        
        # Check inheritance
        assert issubclass(HierarchicalFusion, BaseFusion), "HierarchicalFusion should inherit from BaseFusion"
        print("✓ HierarchicalFusion properly inherits from BaseFusion")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during component class import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test registry error handling."""
    print("\nTesting error handling...")
    
    try:
        from layers.modular.core import unified_registry, ComponentFamily
        import layers.modular.core.register_components
        
        # Test error handling for non-existent component
        try:
            unified_registry.create('non_existent_fusion', ComponentFamily.FUSION)
            print("✗ Should have raised error for non-existent component")
            return False
        except ValueError as e:
            print(f"✓ Proper error handling: {e}")
        except Exception as e:
            print(f"✓ Error handling works (different exception type): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during error handling test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("FUSION REGISTRY BASIC VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_registry_structure,
        test_component_class_import,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL BASIC TESTS PASSED! Fusion registry structure is correct.")
        print("Note: Full functionality testing requires PyTorch installation.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
        sys.exit(1)