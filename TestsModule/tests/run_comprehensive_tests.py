#!/usr/bin/env python3
"""
Comprehensive test runner for Autoformer models
Tests each critical point of the Autoformer workflow
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite():
    """Run comprehensive test suite covering all critical workflow points"""
    print("=" * 60)
    print("COMPREHENSIVE AUTOFORMER WORKFLOW TEST SUITE")
    print("=" * 60)
    
    # Test categories with descriptions
    test_categories = [
        {
            'name': 'Core Algorithm Tests',
            'files': [
                'core_algorithms/test_autocorrelation_mechanism.py',
                'core_algorithms/test_series_decomposition.py',
                'core_algorithms/test_attention_mechanisms.py'
            ],
            'description': 'Test core Autoformer mechanisms'
        },
        {
            'name': 'Training Validation Tests', 
            'files': [
                'training_validation/test_simple_training.py',
                'training_validation/test_performance_benchmarks.py',
                'training_validation/test_robustness.py'
            ],
            'description': 'Test training behavior and performance'
        },
        {
            'name': 'Integration Tests',
            'files': [
                'integration/test_end_to_end_workflows.py',
                'integration/test_model_comparison.py'
            ],
            'description': 'Test end-to-end workflows and model comparisons'
        },
        {
            'name': 'Utilities Tests',
            'files': [
                'utilities/test_modular_components.py',
                'utilities/test_configuration_robustness.py'
            ],
            'description': 'Test utilities and modular components'
        }
    ]
    
    results = {}
    total_start = time.time()
    
    for category in test_categories:
        print(f"\n{category['name'].upper()}")
        print(f"Description: {category['description']}")
        print("-" * 40)
        
        category_results = {}
        
        for test_file in category['files']:
            print(f"Running {test_file}...")
            start_time = time.time()
            
            try:
                # Check if test file exists
                test_path = Path(__file__).parent / test_file
                if not test_path.exists():
                    print(f"  SKIP: {test_file} not found")
                    category_results[test_file] = "SKIP"
                    continue
                
                # Run pytest
                result = subprocess.run([
                    sys.executable, "-m", "pytest", 
                    str(test_path), "-v", "--tb=short"
                ], capture_output=True, text=True, cwd=Path(__file__).parent)
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"  PASS: {test_file} ({duration:.2f}s)")
                    category_results[test_file] = "PASS"
                else:
                    print(f"  FAIL: {test_file} ({duration:.2f}s)")
                    category_results[test_file] = "FAIL"
                    
            except Exception as e:
                print(f"  ERROR: {test_file} - {e}")
                category_results[test_file] = "ERROR"
        
        results[category['name']] = category_results
    
    total_duration = time.time() - total_start
    
    # Summary Report
    print("\n" + "=" * 60)
    print("TEST SUMMARY REPORT")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    error_tests = 0
    
    for category_name, category_results in results.items():
        print(f"\n{category_name}:")
        for test_file, result in category_results.items():
            status_symbol = {
                "PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "ERROR": "[ERROR]"
            }.get(result, "[?]")
            print(f"  {status_symbol} {test_file}: {result}")
            
            total_tests += 1
            if result == "PASS":
                passed_tests += 1
            elif result == "FAIL":
                failed_tests += 1
            elif result == "SKIP":
                skipped_tests += 1
            elif result == "ERROR":
                error_tests += 1
    
    print(f"\nOVERALL RESULTS:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Skipped: {skipped_tests}")
    print(f"Errors: {error_tests}")
    print(f"Duration: {total_duration:.2f}s")
    
    # Final assessment
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.8 and failed_tests == 0:
        print(f"\nEXCELLENT: {success_rate:.1%} success rate - Models are production ready!")
        return True
    elif success_rate >= 0.6:
        print(f"\nGOOD: {success_rate:.1%} success rate - Some issues need attention")
        return False
    else:
        print(f"\nPOOR: {success_rate:.1%} success rate - Significant issues detected")
        return False

if __name__ == "__main__":
    print("Starting Comprehensive Autoformer Test Suite")
    success = run_test_suite()
    
    if success:
        print("\nCONCLUSION: Autoformer models are comprehensively tested and ready!")
        sys.exit(0)
    else:
        print("\nCONCLUSION: Some test failures detected. Review needed.")
        sys.exit(1)