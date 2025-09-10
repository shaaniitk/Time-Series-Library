#!/usr/bin/env python3
"""
Comprehensive test runner for all component tests in the Time Series Library.

This script runs tests for:
- Backbone components
- Feedforward components  
- Output components
- Integration tests

Usage:
    python tests/run_component_tests.py
    python tests/run_component_tests.py --component backbone
    python tests/run_component_tests.py --component feedforward
    python tests/run_component_tests.py --component output
    python tests/run_component_tests.py --verbose
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_pytest_command(test_path, verbose=False, capture=True):
    """
    Run pytest on a specific test path and return results.
    """
    cmd = ["python", "-m", "pytest", str(test_path)]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend(["--cov=layers.modular", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    if not capture:
        cmd.append("--capture=no")
    
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True, cwd=project_root)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def run_backbone_tests(verbose=False):
    """
    Run all backbone component tests.
    """
    print("\n" + "="*60)
    print("RUNNING BACKBONE COMPONENT TESTS")
    print("="*60)
    
    test_path = project_root / "tests" / "component" / "backbone"
    if not test_path.exists():
        print(f"❌ Backbone test directory not found: {test_path}")
        return False
    
    returncode, stdout, stderr = run_pytest_command(test_path, verbose)
    
    if returncode == 0:
        print("✅ All backbone tests passed!")
        if verbose and stdout:
            print(stdout)
        return True
    else:
        print("❌ Some backbone tests failed!")
        if stdout:
            print("STDOUT:", stdout)
        if stderr:
            print("STDERR:", stderr)
        return False

def run_feedforward_tests(verbose=False):
    """
    Run all feedforward component tests.
    """
    print("\n" + "="*60)
    print("RUNNING FEEDFORWARD COMPONENT TESTS")
    print("="*60)
    
    test_path = project_root / "tests" / "component" / "feedforward"
    if not test_path.exists():
        print(f"❌ Feedforward test directory not found: {test_path}")
        return False
    
    returncode, stdout, stderr = run_pytest_command(test_path, verbose)
    
    if returncode == 0:
        print("✅ All feedforward tests passed!")
        if verbose and stdout:
            print(stdout)
        return True
    else:
        print("❌ Some feedforward tests failed!")
        if stdout:
            print("STDOUT:", stdout)
        if stderr:
            print("STDERR:", stderr)
        return False

def run_output_tests(verbose=False):
    """
    Run all output component tests.
    """
    print("\n" + "="*60)
    print("RUNNING OUTPUT COMPONENT TESTS")
    print("="*60)
    
    test_path = project_root / "tests" / "component" / "output"
    if not test_path.exists():
        print(f"❌ Output test directory not found: {test_path}")
        return False
    
    returncode, stdout, stderr = run_pytest_command(test_path, verbose)
    
    if returncode == 0:
        print("✅ All output tests passed!")
        if verbose and stdout:
            print(stdout)
        return True
    else:
        print("❌ Some output tests failed!")
        if stdout:
            print("STDOUT:", stdout)
        if stderr:
            print("STDERR:", stderr)
        return False

def run_integration_tests(verbose=False):
    """
    Run component integration tests.
    """
    print("\n" + "="*60)
    print("RUNNING COMPONENT INTEGRATION TESTS")
    print("="*60)
    
    # Look for integration test files
    integration_files = [
        project_root / "tests" / "test_registry_integration.py",
        project_root / "tests" / "component" / "test_component_integration.py"
    ]
    
    all_passed = True
    for test_file in integration_files:
        if test_file.exists():
            print(f"\nRunning {test_file.name}...")
            returncode, stdout, stderr = run_pytest_command(test_file, verbose)
            
            if returncode == 0:
                print(f"✅ {test_file.name} passed!")
                if verbose and stdout:
                    print(stdout)
            else:
                print(f"❌ {test_file.name} failed!")
                if stdout:
                    print("STDOUT:", stdout)
                if stderr:
                    print("STDERR:", stderr)
                all_passed = False
    
    return all_passed

def run_all_component_tests(verbose=False):
    """
    Run all component tests in sequence.
    """
    print("🚀 Starting comprehensive component test suite...")
    
    results = {
        "backbone": run_backbone_tests(verbose),
        "feedforward": run_feedforward_tests(verbose),
        "output": run_output_tests(verbose),
        "integration": run_integration_tests(verbose)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("COMPONENT TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_type, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_type.upper():15} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL COMPONENT TESTS PASSED! 🎉")
        print("The component registry system is working correctly.")
    else:
        print("💥 SOME TESTS FAILED! 💥")
        print("Please check the output above for details.")
    print("="*60)
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Run component tests for Time Series Library")
    parser.add_argument(
        "--component", 
        choices=["backbone", "feedforward", "output", "integration", "all"],
        default="all",
        help="Which component tests to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    os.chdir(project_root)
    
    success = True
    
    if args.component == "all":
        success = run_all_component_tests(args.verbose)
    elif args.component == "backbone":
        success = run_backbone_tests(args.verbose)
    elif args.component == "feedforward":
        success = run_feedforward_tests(args.verbose)
    elif args.component == "output":
        success = run_output_tests(args.verbose)
    elif args.component == "integration":
        success = run_integration_tests(args.verbose)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()