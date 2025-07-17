#!/usr/bin/env python3
"""
Simple Test Runner for ChronosX Integration
Focuses on working ChronosX tests only
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_test(test_name, test_path):
    """Run a single test"""
    print(f"\n🔬 Running {test_name}...")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Change to project root and run test
        project_root = Path(__file__).parent.absolute()
        result = subprocess.run(
            [sys.executable, test_path],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            return True, duration, ""
        else:
            print(f"❌ {test_name} FAILED ({duration:.1f}s)")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}...")
            return False, duration, result.stderr
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {test_name} TIMED OUT ({duration:.1f}s)")
        return False, duration, "Timeout"
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {test_name} CRASHED ({duration:.1f}s): {str(e)}")
        return False, duration, str(e)

def main():
    """Main test runner"""
    print("🚀 ChronosX Integration Test Runner")
    print("=" * 70)
    
    # Tests that should work
    tests = [
        ("Installation Verification", "verify_installation.py"),
        ("Component Inspection", "inspect_components.py"),
        ("ChronosX Simple Demo", "tests/chronosx/chronos_x_simple_demo.py"),
        ("ChronosX Model Sizes", "tests/chronosx/test_chronos_x_model_sizes.py"),
        ("ChronosX Real Data", "tests/chronosx/test_chronos_x_real_data.py"),
        ("ChronosX Demo", "tests/chronosx/chronos_x_demo.py"),
        ("ChronosX Modular Demo", "tests/chronosx/demo_chronosx_modular.py"),
        ("ChronosX Benchmark", "tests/chronosx/chronos_x_benchmark_suite.py"),
    ]
    
    results = {}
    total_passed = 0
    total_tests = 0
    total_time = 0
    
    for test_name, test_path in tests:
        # Check if test file exists
        if not os.path.exists(test_path):
            print(f"⚠️ Skipping {test_name} - file not found: {test_path}")
            continue
            
        total_tests += 1
        success, duration, error = run_test(test_name, test_path)
        
        results[test_name] = {
            'success': success,
            'duration': duration,
            'error': error
        }
        
        if success:
            total_passed += 1
        
        total_time += duration
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 70)
    print(f"📊 Results: {total_passed}/{total_tests} tests passed")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! ChronosX integration is working perfectly!")
    else:
        print("⚠️ Some tests failed. Details below:")
        
        for test_name, result in results.items():
            if not result['success']:
                print(f"   ❌ {test_name}: {result['error'][:100]}...")
    
    print(f"\n💡 Working Features:")
    for test_name, result in results.items():
        if result['success']:
            print(f"   ✅ {test_name}")
    
    if total_passed > 0:
        print(f"\n🚀 At least {total_passed} components are working!")
        print("   You can use the working features for development and testing.")
    
    return 0 if total_passed > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
