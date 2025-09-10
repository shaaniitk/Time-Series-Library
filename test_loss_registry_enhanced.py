#!/usr/bin/env python3
"""
Simplified test for enhanced loss registry integration.

This script validates the registry structure and component registration
without requiring PyTorch dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_loss_registry_structure():
    """Test that ComponentFamily.LOSS exists and is properly defined."""
    print("\n=== Testing Loss Registry Structure ===")
    
    try:
        # Test basic enum structure without torch imports
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "registry", 
            "layers/modular/core/registry.py"
        )
        
        if spec is None or spec.loader is None:
            print("✗ Could not load registry module")
            return False
            
        # Read the file content to check for LOSS enum
        with open("layers/modular/core/registry.py", 'r') as f:
            content = f.read()
            
        if 'LOSS = "loss"' in content:
            print("✓ ComponentFamily.LOSS properly defined in registry.py")
        else:
            print("✗ ComponentFamily.LOSS not found in registry.py")
            return False
            
        if 'class ComponentFamily(Enum):' in content:
            print("✓ ComponentFamily enum class found")
        else:
            print("✗ ComponentFamily enum class not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing loss registry structure: {e}")
        return False

def test_register_components_structure():
    """Test that register_components.py has loss component registrations."""
    print("\n=== Testing Register Components Structure ===")
    
    try:
        with open("layers/modular/core/register_components.py", 'r') as f:
            content = f.read()
            
        # Check for loss component imports
        loss_imports = [
            "from ..loss.quantile_loss import PinballLoss",
            "from ..loss.standard_losses import StandardLossWrapper",
            "from ..loss.advanced_losses import",
            "from ..loss.adaptive_bayesian_losses import"
        ]
        
        missing_imports = []
        for import_line in loss_imports:
            if import_line not in content:
                missing_imports.append(import_line)
            else:
                print(f"✓ Found import: {import_line}")
        
        if missing_imports:
            print(f"✗ Missing imports: {missing_imports}")
            return False
            
        # Check for loss component registrations
        loss_registrations = [
            'name="quantile_loss"',
            'name="pinball_loss"',
            'name="mape_loss"',
            'name="focal_loss"',
            'component_type=ComponentFamily.LOSS'
        ]
        
        missing_registrations = []
        for registration in loss_registrations:
            if registration not in content:
                missing_registrations.append(registration)
            else:
                print(f"✓ Found registration: {registration}")
        
        if missing_registrations:
            print(f"✗ Missing registrations: {missing_registrations}")
            return False
            
        print("✓ All expected loss component registrations found")
        return True
        
    except Exception as e:
        print(f"✗ Error testing register components structure: {e}")
        return False

def test_core_init_exports():
    """Test that core/__init__.py exports unified_registry."""
    print("\n=== Testing Core Init Exports ===")
    
    try:
        with open("layers/modular/core/__init__.py", 'r') as f:
            content = f.read()
            
        # Check for unified_registry export
        if 'unified_registry' in content:
            print("✓ unified_registry found in __init__.py")
        else:
            print("✗ unified_registry not found in __init__.py")
            return False
            
        # Check for proper import statement
        if 'component_registry as unified_registry' in content:
            print("✓ Proper unified_registry import alias found")
        else:
            print("✗ Proper unified_registry import alias not found")
            return False
            
        # Check __all__ export
        if '"unified_registry"' in content:
            print("✓ unified_registry in __all__ exports")
        else:
            print("✗ unified_registry not in __all__ exports")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing core init exports: {e}")
        return False

def test_loss_registry_deprecation_shim():
    """Test that loss registry has proper deprecation shim."""
    print("\n=== Testing Loss Registry Deprecation Shim ===")
    
    try:
        with open("layers/modular/loss/registry.py", 'r') as f:
            content = f.read()
            
        # Check for deprecation warning
        if 'DeprecationWarning' in content:
            print("✓ DeprecationWarning found in loss registry")
        else:
            print("✗ DeprecationWarning not found in loss registry")
            return False
            
        # Check for unified registry import attempt
        if 'from layers.modular.core import unified_registry' in content:
            print("✓ Unified registry import found in deprecation shim")
        else:
            print("✗ Unified registry import not found in deprecation shim")
            return False
            
        # Check for shim functions
        if '_shim_get' in content and '_shim_list' in content:
            print("✓ Deprecation shim functions found")
        else:
            print("✗ Deprecation shim functions not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing loss registry deprecation shim: {e}")
        return False

def test_file_structure_integrity():
    """Test that all required files exist and are accessible."""
    print("\n=== Testing File Structure Integrity ===")
    
    required_files = [
        "layers/modular/core/registry.py",
        "layers/modular/core/__init__.py",
        "layers/modular/core/register_components.py",
        "layers/modular/loss/registry.py",
        "layers/modular/fusion/hierarchical_fusion.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} missing")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True

def test_component_count_validation():
    """Test that the expected number of loss components are registered."""
    print("\n=== Testing Component Count Validation ===")
    
    try:
        with open("layers/modular/core/register_components.py", 'r') as f:
            content = f.read()
            
        # Count component registrations for LOSS type
        loss_registrations = content.count('component_type=ComponentFamily.LOSS')
        
        expected_count = 12  # Based on our registration additions
        
        if loss_registrations >= expected_count:
            print(f"✓ Found {loss_registrations} loss component registrations (expected >= {expected_count})")
            return True
        else:
            print(f"✗ Found only {loss_registrations} loss component registrations (expected >= {expected_count})")
            return False
            
    except Exception as e:
        print(f"✗ Error testing component count: {e}")
        return False

def main():
    """Run all loss registry enhancement tests."""
    print("Enhanced Loss Registry Integration Test (Structure Only)")
    print("=" * 60)
    
    tests = [
        test_file_structure_integrity,
        test_loss_registry_structure,
        test_register_components_structure,
        test_core_init_exports,
        test_loss_registry_deprecation_shim,
        test_component_count_validation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! Loss registry enhancement is properly configured.")
        print("Note: Runtime tests require PyTorch installation.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Loss registry structure needs attention.")
        return 1

if __name__ == "__main__":
    exit(main())