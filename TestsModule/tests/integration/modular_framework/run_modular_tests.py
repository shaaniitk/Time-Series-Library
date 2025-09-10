#!/usr/bin/env python3
"""
Test runner for comprehensive modular framework tests.

This script runs all test suites for the modular framework:
1. Unified registry comprehensive tests
2. Fusion component tests
3. Loss component tests

Usage:
    python run_modular_tests.py [--verbose] [--specific TEST_NAME]
"""

import unittest
import sys
import os
import argparse
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

def discover_and_run_tests(test_pattern=None, verbosity=1):
    """
    Discover and run all modular framework tests.
    
    Args:
        test_pattern: Optional pattern to filter specific tests
        verbosity: Test output verbosity level (0-2)
    
    Returns:
        TestResult object
    """
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests in this directory
    if test_pattern:
        # Load specific test pattern
        suite = loader.loadTestsFromName(test_pattern)
    else:
        # Discover all tests in the modular framework directory
        suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Create test runner
    stream = StringIO() if verbosity == 0 else sys.stdout
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=verbosity,
        buffer=True,
        failfast=False
    )
    
    # Run tests
    result = runner.run(suite)
    
    return result

def print_test_summary(result):
    """
    Print a summary of test results.
    
    Args:
        result: TestResult object from test run
    """
    print("\n" + "="*60)
    print("MODULAR FRAMEWORK TEST SUMMARY")
    print("="*60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {total_tests - failures - errors - skipped}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if failures > 0:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
    
    if errors > 0:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See details above'}")
    
    if hasattr(result, 'skipped') and skipped > 0:
        print("\nSKIPPED:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    print("\n" + "="*60)
    
    # Overall result
    if failures == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False

def run_individual_test_suites(verbosity=1):
    """
    Run individual test suites and report on each.
    
    Args:
        verbosity: Test output verbosity level
    
    Returns:
        Dictionary with results for each test suite
    """
    test_suites = [
        ('test_unified_registry_comprehensive', 'Unified Registry Tests'),
        ('test_fusion_components', 'Fusion Component Tests'),
        ('test_loss_components', 'Loss Component Tests')
    ]
    
    results = {}
    
    for test_module, description in test_suites:
        print(f"\n{'='*20} {description} {'='*20}")
        
        try:
            result = discover_and_run_tests(test_module, verbosity)
            results[test_module] = result
            
            # Quick summary for this suite
            total = result.testsRun
            failed = len(result.failures) + len(result.errors)
            print(f"\n{description}: {total - failed}/{total} tests passed")
            
        except Exception as e:
            print(f"Error running {description}: {e}")
            results[test_module] = None
    
    return results

def main():
    """
    Main function to run modular framework tests.
    """
    parser = argparse.ArgumentParser(
        description='Run comprehensive tests for the modular framework'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run tests with verbose output'
    )
    parser.add_argument(
        '--specific', '-s',
        type=str,
        help='Run specific test (e.g., test_unified_registry_comprehensive)'
    )
    parser.add_argument(
        '--individual', '-i',
        action='store_true',
        help='Run test suites individually with separate reporting'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Run tests with minimal output'
    )
    
    args = parser.parse_args()
    
    # Determine verbosity level
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    print("Modular Framework Test Suite")
    print("============================")
    print("Testing unified registry, fusion components, and loss components...\n")
    
    try:
        if args.individual:
            # Run individual test suites
            results = run_individual_test_suites(verbosity)
            
            # Overall summary
            print("\n" + "="*60)
            print("OVERALL SUMMARY")
            print("="*60)
            
            total_passed = 0
            total_tests = 0
            
            for test_module, result in results.items():
                if result:
                    passed = result.testsRun - len(result.failures) - len(result.errors)
                    total_passed += passed
                    total_tests += result.testsRun
                    print(f"{test_module}: {passed}/{result.testsRun} passed")
                else:
                    print(f"{test_module}: ERROR")
            
            if total_tests > 0:
                overall_rate = (total_passed / total_tests * 100)
                print(f"\nOverall: {total_passed}/{total_tests} tests passed ({overall_rate:.1f}%)")
            
        else:
            # Run all tests together
            result = discover_and_run_tests(args.specific, verbosity)
            success = print_test_summary(result)
            
            # Exit with appropriate code
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tests: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()