#!/usr/bin/env python3
"""
Comprehensive Test Runner for Time Series Library

This runner organizes and executes all tests in the proper hierarchical structure:
- Unit tests for individual components
- Integration tests for end-to-end workflows  
- Dimension management tests for multi-time series scenarios
- Modular framework tests for component systems

Usage:
    python test_runner_comprehensive.py --suite unit
    python test_runner_comprehensive.py --suite integration
    python test_runner_comprehensive.py --suite dimension
    python test_runner_comprehensive.py --suite all
"""

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

class TestSuiteRunner:
    """Comprehensive test suite runner with proper organization"""
    
    def __init__(self):
        self.project_root = project_root
        self.tests_dir = self.project_root / "tests"
        
        # Test suite organization
        self.test_suites = {
            'unit': {
                'description': 'Unit tests for individual components',
                'paths': [
                    'tests/unit/components',
                    'tests/unit/models', 
                    'tests/unit/utils'
                ],
                'priority': 1
            },
            'integration': {
                'description': 'Integration tests for end-to-end workflows',
                'paths': [
                    'tests/integration/end_to_end',
                    'tests/integration/modular_framework'
                ],
                'priority': 2
            },
            'dimension': {
                'description': 'Dimension management tests for multi-time series',
                'paths': [
                    'tests/integration/dimension_tests'
                ],
                'priority': 3
            },
            'legacy': {
                'description': 'Legacy tests and standalone scripts',
                'paths': [
                    'tests/legacy',
                    '.'  # Root level test files
                ],
                'priority': 4
            }
        }
    
    def discover_tests(self, paths: List[str]) -> List[Path]:
        """Discover all test files in given paths"""
        test_files = set()  # Use set to avoid duplicates
        
        for path_str in paths:
            path = self.project_root / path_str
            if path.exists():
                # Find all Python test files
                for test_file in path.rglob("test_*.py"):
                    test_files.add(test_file)
                
                # Also check for direct Python files that can be run (but avoid duplicates)
                for py_file in path.rglob("*.py"):
                    if py_file.name.startswith('run_') and py_file.is_file():
                        test_files.add(py_file)
        
        return sorted(list(test_files))
    
    def run_test_file(self, test_file: Path) -> Tuple[bool, str, float]:
        """Run a single test file and return results"""
        print(f"\n  TEST Running: {test_file.relative_to(self.project_root)}")
        
        start_time = time.time()
        
        try:
            # Try to run as a module first, then as a script
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"    PASS PASSED ({elapsed:.2f}s)")
                return True, result.stdout, elapsed
            else:
                print(f"    FAIL FAILED ({elapsed:.2f}s)")
                print(f"    Error: {result.stderr[:200]}...")
                return False, result.stderr, elapsed
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"     TIMEOUT ({elapsed:.2f}s)")
            return False, "Test timed out", elapsed
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    FAIL ERROR ({elapsed:.2f}s): {e}")
            return False, str(e), elapsed
    
    def run_suite(self, suite_name: str) -> Dict:
        """Run a specific test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_config = self.test_suites[suite_name]
        print(f"\nTARGET Running {suite_name.upper()} Test Suite")
        print(f"   Description: {suite_config['description']}")
        print("=" * 70)
        
        # Discover tests
        test_files = self.discover_tests(suite_config['paths'])
        
        if not test_files:
            print(f"WARN  No test files found in {suite_name} suite")
            return {
                'suite': suite_name,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': []
            }
        
        print(f" Found {len(test_files)} test file(s)")
        
        # Run tests
        results = {
            'suite': suite_name,
            'total': len(test_files),
            'passed': 0,
            'failed': 0,
            'errors': [],
            'total_time': 0
        }
        
        for test_file in test_files:
            success, output, elapsed = self.run_test_file(test_file)
            results['total_time'] += elapsed
            
            if success:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'file': str(test_file.relative_to(self.project_root)),
                    'error': output
                })
        
        return results
    
    def print_summary(self, results: Dict):
        """Print a comprehensive summary of test results"""
        print("\n" + "=" * 70)
        print("CHART TEST SUITE SUMMARY")
        print("=" * 70)
        
        print(f"TARGET {results['suite'].upper()} Suite Results:")
        print(f"   Total Tests: {results['total']}")
        print(f"   Passed: {results['passed']} PASS")
        print(f"   Failed: {results['failed']} FAIL")
        if results['total'] > 0:
            print(f"   Success Rate: {(results['passed']/results['total']*100):.1f}%")
        print(f"   Total Time: {results.get('total_time', 0):.2f}s")
        
        if results['errors']:
            print(f"\nFAIL Failed Tests:")
            for error in results['errors']:
                print(f"   - {error['file']}")

def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description='Comprehensive Test Suite Runner')
    parser.add_argument(
        '--suite',
        choices=['unit', 'integration', 'dimension', 'legacy'],
        required=True,
        help='Test suite to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner()
    
    try:
        results = runner.run_suite(args.suite)
        runner.print_summary(results)
        
        # Exit with error code if any tests failed
        if results['failed'] > 0:
            sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nWARN  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFAIL Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
