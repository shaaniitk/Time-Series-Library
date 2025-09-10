"""Test runner for all registry tests."""

import sys
import os
import pytest
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_registry_tests():
    """Run all registry-related tests."""
    test_files = [
        'test_backbone_registry.py',
        'test_feedforward_registry.py', 
        'test_output_registry.py',
        'test_registry_integration.py'
    ]
    
    print("Running comprehensive registry tests...")
    print("=" * 50)
    
    all_passed = True
    results = {}
    
    for test_file in test_files:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            print(f"\nRunning {test_file}...")
            result = pytest.main([str(test_path), '-v'])
            results[test_file] = result
            if result != 0:
                all_passed = False
                print(f"❌ {test_file} FAILED")
            else:
                print(f"✅ {test_file} PASSED")
        else:
            print(f"⚠️  {test_file} not found")
            all_passed = False
    
    print("\n" + "=" * 50)
    print("REGISTRY TEST SUMMARY")
    print("=" * 50)
    
    for test_file, result in results.items():
        status = "PASSED" if result == 0 else "FAILED"
        emoji = "✅" if result == 0 else "❌"
        print(f"{emoji} {test_file}: {status}")
    
    if all_passed:
        print("\n🎉 All registry tests PASSED!")
        print("\nRegistry system is fully functional with:")
        print("  - Backbone components (chronos, t5, bert, simple_transformer)")
        print("  - Feedforward components (standard, gated, MoE, conv)")
        print("  - Output components (linear, forecasting, regression)")
        print("  - Full integration with unified registry system")
        print("  - Comprehensive error handling and validation")
        return 0
    else:
        print("\n❌ Some registry tests FAILED!")
        print("Please check the test output above for details.")
        return 1

def run_specific_test(test_name):
    """Run a specific test file."""
    test_path = Path(__file__).parent / f"test_{test_name}_registry.py"
    if test_path.exists():
        print(f"Running {test_name} registry tests...")
        return pytest.main([str(test_path), '-v'])
    else:
        print(f"Test file for {test_name} not found")
        return 1

def run_integration_tests():
    """Run only integration tests."""
    test_path = Path(__file__).parent / "test_registry_integration.py"
    if test_path.exists():
        print("Running registry integration tests...")
        return pytest.main([str(test_path), '-v'])
    else:
        print("Integration test file not found")
        return 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'integration':
            exit_code = run_integration_tests()
        elif command in ['backbone', 'feedforward', 'output']:
            exit_code = run_specific_test(command)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: integration, backbone, feedforward, output")
            exit_code = 1
    else:
        exit_code = run_all_registry_tests()
    
    sys.exit(exit_code)