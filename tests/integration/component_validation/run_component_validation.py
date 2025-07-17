#!/usr/bin/env python3
"""
Comprehensive Component Validation Test Runner

This script runs all component validation tests to ensure each component
works correctly with expected mathematical behaviors and properties.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def run_test_module(module_path, test_name):
    """Run a specific test module and return results"""
    print(f"\n🎯 Running {test_name}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Import and run the test module
        if module_path == "test_backbone_functionality":
            from tests.integration.component_validation.test_backbone_functionality import run_backbone_functionality_tests
            success = run_backbone_functionality_tests()
        elif module_path == "test_loss_functionality":
            from tests.integration.component_validation.test_loss_functionality import run_loss_functionality_tests
            success = run_loss_functionality_tests()
        elif module_path == "test_attention_functionality":
            from tests.integration.component_validation.test_attention_functionality import run_attention_functionality_tests
            success = run_attention_functionality_tests()
        elif module_path == "test_processor_functionality":
            from tests.integration.component_validation.test_processor_functionality import run_processor_functionality_tests
            success = run_processor_functionality_tests()
        elif module_path == "test_integration_functionality":
            from tests.integration.component_validation.test_integration_functionality import run_integration_functionality_tests
            success = run_integration_functionality_tests()
        else:
            print(f"❌ Unknown test module: {module_path}")
            return False, 0
            
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f"✅ {test_name} completed successfully in {duration:.2f}s")
        else:
            print(f"❌ {test_name} failed after {duration:.2f}s")
            
        return success, duration
        
    except ImportError as e:
        print(f"❌ Could not import {test_name}: {e}")
        return False, 0
    except Exception as e:
        print(f"❌ Error running {test_name}: {e}")
        return False, 0

def run_quick_validation():
    """Run a quick validation of all components"""
    print("🚀 Running Quick Component Validation")
    print("=" * 80)
    
    # Quick tests for each component type
    quick_tests = [
        ("Backbone Quick Test", "test_backbone_functionality"),
        ("Loss Quick Test", "test_loss_functionality"),
        ("Attention Quick Test", "test_attention_functionality"),
    ]
    
    passed = 0
    total = len(quick_tests)
    total_time = 0
    
    for test_name, module_path in quick_tests:
        success, duration = run_test_module(module_path, test_name)
        if success:
            passed += 1
        total_time += duration
    
    print(f"\n📊 Quick Validation Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    print(f"   Total Time: {total_time:.2f}s")
    
    return passed == total

def run_comprehensive_validation():
    """Run comprehensive validation of all components"""
    print("🚀 Running Comprehensive Component Validation")
    print("=" * 80)
    
    # All validation tests
    all_tests = [
        ("Backbone Component Functionality", "test_backbone_functionality"),
        ("Loss Function Functionality", "test_loss_functionality"),
        ("Attention Mechanism Functionality", "test_attention_functionality"),
        ("Processor Component Functionality", "test_processor_functionality"),
        ("Integration Component Functionality", "test_integration_functionality"),
    ]
    
    passed = 0
    total = len(all_tests)
    total_time = 0
    results = []
    
    for test_name, module_path in all_tests:
        success, duration = run_test_module(module_path, test_name)
        results.append((test_name, success, duration))
        if success:
            passed += 1
        total_time += duration
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("📊 Comprehensive Component Validation Results")
    print("=" * 80)
    
    print("\n📋 Test Results Summary:")
    for test_name, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} - {test_name} ({duration:.2f}s)")
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Time per Test: {total_time/total:.2f}s")
    
    if passed == total:
        print("\n🎉 All component validation tests passed!")
        print("✨ Components are working correctly with expected mathematical behaviors")
    else:
        print("\n⚠️ Some component validation tests failed")
        print("🔍 Review failed tests for component issues")
    
    return passed == total

def run_focused_validation(component_type):
    """Run validation for a specific component type"""
    print(f"🚀 Running Focused Validation: {component_type.title()}")
    print("=" * 80)
    
    # Map component types to test modules
    component_map = {
        'backbone': ("Backbone Component Functionality", "test_backbone_functionality"),
        'loss': ("Loss Function Functionality", "test_loss_functionality"),
        'attention': ("Attention Mechanism Functionality", "test_attention_functionality"),
        'processor': ("Processor Component Functionality", "test_processor_functionality"),
        'integration': ("Integration Component Functionality", "test_integration_functionality"),
    }
    
    if component_type not in component_map:
        print(f"❌ Unknown component type: {component_type}")
        print(f"Available types: {list(component_map.keys())}")
        return False
    
    test_name, module_path = component_map[component_type]
    success, duration = run_test_module(module_path, test_name)
    
    print(f"\n📊 Focused Validation Results:")
    print(f"   Component: {component_type.title()}")
    print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"   Duration: {duration:.2f}s")
    
    return success

def print_usage():
    """Print usage information"""
    print("Component Validation Test Runner")
    print("=" * 40)
    print("Usage: python run_component_validation.py [mode] [component]")
    print()
    print("Modes:")
    print("  quick         - Run quick validation (backbone, loss, attention)")
    print("  comprehensive - Run all validation tests (default)")
    print("  focused       - Run validation for specific component")
    print()
    print("Components (for focused mode):")
    print("  backbone      - Test backbone components")
    print("  loss          - Test loss functions")
    print("  attention     - Test attention mechanisms")
    print("  processor     - Test processor components")
    print("  integration   - Test component integration")
    print()
    print("Examples:")
    print("  python run_component_validation.py")
    print("  python run_component_validation.py quick")
    print("  python run_component_validation.py comprehensive")
    print("  python run_component_validation.py focused backbone")

def main():
    """Main function"""
    args = sys.argv[1:]
    
    if len(args) == 0:
        # Default: comprehensive validation
        success = run_comprehensive_validation()
    elif args[0] == "quick":
        success = run_quick_validation()
    elif args[0] == "comprehensive":
        success = run_comprehensive_validation()
    elif args[0] == "focused":
        if len(args) < 2:
            print("❌ Focused mode requires component type")
            print_usage()
            sys.exit(1)
        success = run_focused_validation(args[1])
    elif args[0] in ["help", "-h", "--help"]:
        print_usage()
        sys.exit(0)
    else:
        print(f"❌ Unknown mode: {args[0]}")
        print_usage()
        sys.exit(1)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
