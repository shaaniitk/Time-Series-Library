#!/usr/bin/env python3
"""
Comprehensive Test Runner for Time Series Library

This test runner provides organized execution of all tests with proper
categorization and detailed reporting. It handles the reorganized test
structure and provides specific focus on dimension management issues.

Test Categories:
1. Unit Tests - Individual component testing
2. Integration Tests - Component interaction testing
3. End-to-End Tests - Full pipeline testing
4. Dimension Tests - Specific dimension management validation
5. Modular Framework Tests - Modular architecture validation

Usage:
  python tests/test_runner_comprehensive.py [category] [options]
  
Examples:
  python tests/test_runner_comprehensive.py all                    # Run all tests
  python tests/test_runner_comprehensive.py unit                   # Run only unit tests
  python tests/test_runner_comprehensive.py dimension              # Run dimension tests
  python tests/test_runner_comprehensive.py integration            # Run integration tests
  python tests/test_runner_comprehensive.py quick                  # Run essential tests only
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


class TestCategory:
    """Test category configuration"""
    
    def __init__(self, name: str, description: str, paths: List[str], 
                 priority: int = 3, timeout: int = 300):
        self.name = name
        self.description = description
        self.paths = paths
        self.priority = priority  # 1=critical, 5=optional
        self.timeout = timeout


class ComprehensiveTestRunner:
    """
    Comprehensive test runner for the reorganized test structure
    """
    
    def __init__(self):
        self.project_root = project_root
        self.test_root = self.project_root / "tests"
        
        # Define test categories
        self.categories = {
            'dimension': TestCategory(
                name='Dimension Management',
                description='Tests for DimensionManager and multi-time series dimension handling',
                paths=[
                    'tests/integration/dimension_tests/test_dimension_manager_integration.py',
                    'tests/integration/end_to_end/test_multi_timeseries_dimensions.py',
                    'tests/integration/end_to_end/test_hf_autoformer_dimensions.py',
                    'tests/integration/dimension_tests/test_model_dimensions.py'
                ],
                priority=1,  # Critical
                timeout=600
            ),
            'unit': TestCategory(
                name='Unit Tests',
                description='Individual component and utility tests',
                paths=[
                    'tests/unit/models/',
                    'tests/unit/components/',
                    'tests/unit/utils/',
                    'tests/test_series_decomposition.py',
                    'tests/test_fft_lengths.py'
                ],
                priority=2,
                timeout=300
            ),
            'integration': TestCategory(
                name='Integration Tests',
                description='Component interaction and workflow tests',
                paths=[
                    'tests/integration/end_to_end/',
                    'tests/integration/',
                    'tests/test_integration.py'
                ],
                priority=2,
                timeout=600
            ),
            'modular': TestCategory(
                name='Modular Framework',
                description='Modular architecture and component registry tests',
                paths=[
                    'tests/integration/modular_framework/',
                ],
                priority=3,
                timeout=450
            ),
            'legacy': TestCategory(
                name='Legacy Tests',
                description='Existing model and feature tests',
                paths=[
                    'tests/test_enhanced_autoformer.py',
                    'tests/test_bayesian_loss_architecture.py',
                    'tests/test_quantile_bayesian.py',
                    'tests/models/'
                ],
                priority=4,
                timeout=400
            ),
            'training': TestCategory(
                name='Training & Validation',
                description='Training dynamics and validation tests',
                paths=[
                    'tests/training_validation/',
                    'tests/test_training_dynamics.py',
                    'tests/test_kl_training.py'
                ],
                priority=3,
                timeout=500
            )
        }
        
        # Quick test selection (most critical tests)
        self.quick_tests = [
            'tests/integration/dimension_tests/test_dimension_manager_integration.py',
            'tests/integration/end_to_end/test_multi_timeseries_dimensions.py',
            'tests/unit/models/test_autoformer.py',
            'tests/test_series_decomposition.py'
        ]
        
        self.results = {}
        self.start_time = None
        
    def run_test_file(self, test_file: str, timeout: int = 300) -> Tuple[bool, str, float]:
        """Run a single test file and return results"""
        
        if not os.path.exists(test_file):
            return False, f"Test file not found: {test_file}", 0.0
        
        print(f"    TEST Running: {os.path.basename(test_file)}")
        
        start_time = time.time()
        
        try:
            # Use pytest if available, otherwise direct execution
            try:
                import pytest
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
                ], capture_output=True, text=True, timeout=timeout, cwd=self.project_root)
            except ImportError:
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, timeout=timeout, cwd=self.project_root)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return True, "PASSED", duration
            else:
                error_output = result.stdout + result.stderr
                return False, f"FAILED: {error_output[-500:]}", duration  # Last 500 chars
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return False, f"TIMEOUT after {timeout}s", duration
            
        except Exception as e:
            duration = time.time() - start_time
            return False, f"ERROR: {str(e)}", duration
    
    def run_test_directory(self, test_dir: str, timeout: int = 300) -> Dict:
        """Run all tests in a directory"""
        
        if not os.path.exists(test_dir):
            return {'passed': 0, 'failed': 1, 'errors': [f"Directory not found: {test_dir}"]}
        
        results = {'passed': 0, 'failed': 0, 'errors': [], 'details': []}
        
        # Find all Python test files
        test_files = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            results['errors'].append(f"No test files found in {test_dir}")
            results['failed'] = 1
            return results
        
        for test_file in test_files:
            success, message, duration = self.run_test_file(test_file, timeout)
            
            if success:
                results['passed'] += 1
                status = "PASS"
            else:
                results['failed'] += 1
                results['errors'].append(f"{test_file}: {message}")
                status = "FAIL"
            
            results['details'].append({
                'file': test_file,
                'status': status,
                'message': message,
                'duration': duration
            })
        
        return results
    
    def run_category(self, category_name: str) -> Dict:
        """Run all tests in a category"""
        
        if category_name not in self.categories:
            return {'passed': 0, 'failed': 1, 'errors': [f"Unknown category: {category_name}"]}
        
        category = self.categories[category_name]
        print(f"\n Running {category.name} Tests")
        print(f"   {category.description}")
        print("   " + "="*60)
        
        total_results = {'passed': 0, 'failed': 0, 'errors': [], 'details': []}
        
        for path in category.paths:
            full_path = str(self.project_root / path)
            
            if os.path.isfile(full_path):
                # Single test file
                success, message, duration = self.run_test_file(full_path, category.timeout)
                
                if success:
                    total_results['passed'] += 1
                    status = "PASS"
                else:
                    total_results['failed'] += 1
                    total_results['errors'].append(f"{path}: {message}")
                    status = "FAIL"
                
                total_results['details'].append({
                    'file': path,
                    'status': status,
                    'message': message,
                    'duration': duration
                })
                
            elif os.path.isdir(full_path):
                # Directory of tests
                dir_results = self.run_test_directory(full_path, category.timeout)
                total_results['passed'] += dir_results['passed']
                total_results['failed'] += dir_results['failed']
                total_results['errors'].extend(dir_results['errors'])
                total_results['details'].extend(dir_results.get('details', []))
            
            else:
                print(f"    WARN Path not found: {path}")
                total_results['failed'] += 1
                total_results['errors'].append(f"Path not found: {path}")
        
        # Summary for category
        total_tests = total_results['passed'] + total_results['failed']
        if total_tests > 0:
            success_rate = (total_results['passed'] / total_tests) * 100
            print(f"\n   CHART {category.name} Summary: {total_results['passed']}/{total_tests} passed ({success_rate:.1f}%)")
        
        return total_results
    
    def run_quick_tests(self) -> Dict:
        """Run essential tests only"""
        
        print(f"\nLIGHTNING Running Quick Test Suite")
        print(f"   Essential tests for rapid validation")
        print("   " + "="*60)
        
        results = {'passed': 0, 'failed': 0, 'errors': [], 'details': []}
        
        for test_file in self.quick_tests:
            full_path = str(self.project_root / test_file)
            
            if os.path.exists(full_path):
                success, message, duration = self.run_test_file(full_path, 180)  # 3 min timeout
                
                if success:
                    results['passed'] += 1
                    status = "PASS"
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{test_file}: {message}")
                    status = "FAIL"
                
                results['details'].append({
                    'file': test_file,
                    'status': status,
                    'message': message,
                    'duration': duration
                })
            else:
                print(f"    WARN Quick test not found: {test_file}")
                results['failed'] += 1
                results['errors'].append(f"Test not found: {test_file}")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all test categories"""
        
        print(f"\nROCKET Running All Test Categories")
        print("="*80)
        
        all_results = {'passed': 0, 'failed': 0, 'errors': [], 'categories': {}}
        
        # Run categories by priority
        categories_by_priority = sorted(
            self.categories.items(), 
            key=lambda x: x[1].priority
        )
        
        for category_name, category in categories_by_priority:
            category_results = self.run_category(category_name)
            
            all_results['passed'] += category_results['passed']
            all_results['failed'] += category_results['failed']
            all_results['errors'].extend(category_results['errors'])
            all_results['categories'][category_name] = category_results
        
        return all_results
    
    def generate_report(self, results: Dict, test_type: str = "Test Run"):
        """Generate a comprehensive test report"""
        
        total_duration = time.time() - self.start_time if self.start_time else 0
        total_tests = results['passed'] + results['failed']
        success_rate = (results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nTARGET {test_type} Report")
        print("="*80)
        print(f"TIMER  Total Duration: {total_duration:.2f} seconds")
        print(f"TEST Total Tests: {total_tests}")
        print(f"PASS Passed: {results['passed']}")
        print(f"FAIL Failed: {results['failed']}")
        print(f"CHART Success Rate: {success_rate:.1f}%")
        
        if results['failed'] > 0:
            print(f"\n Failed Tests/Errors:")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"    {error}")
            
            if len(results['errors']) > 10:
                print(f"   ... and {len(results['errors']) - 10} more errors")
        
        # Category breakdown for all tests
        if 'categories' in results:
            print(f"\nCLIPBOARD Category Breakdown:")
            for cat_name, cat_results in results['categories'].items():
                cat_total = cat_results['passed'] + cat_results['failed']
                cat_rate = (cat_results['passed'] / cat_total * 100) if cat_total > 0 else 0
                priority = self.categories[cat_name].priority
                priority_icon = "" if priority == 1 else "" if priority == 2 else ""
                
                print(f"   {priority_icon} {cat_name}: {cat_results['passed']}/{cat_total} ({cat_rate:.1f}%)")
        
        # Recommendations
        print(f"\nIDEA Recommendations:")
        if success_rate >= 95:
            print("   PARTY Excellent! Test suite is in great shape.")
        elif success_rate >= 80:
            print("    Good test coverage. Address failing tests for improvement.")
        elif success_rate >= 60:
            print("   WARN Moderate issues. Focus on critical test failures first.")
        else:
            print("    Significant issues detected. Immediate attention required.")
        
        if results['failed'] > 0:
            print("   TOOL Focus on fixing dimension management tests first (highest priority)")
            print("    Check test logs for detailed error information")
            print("    Run 'python tests/test_runner_comprehensive.py quick' for rapid iteration")
        
        print("="*80)
    
    def run(self, test_target: str = 'all'):
        """Main test runner entry point"""
        
        self.start_time = time.time()
        
        print("TEST Time Series Library - Comprehensive Test Suite")
        print("="*80)
        print("TARGET Focus: Dimension Management & Multi-Time Series Support")
        print("  Architecture: Modular HFAutoformer Framework")
        print("="*80)
        
        if test_target == 'all':
            results = self.run_all_tests()
            self.generate_report(results, "Comprehensive Test Suite")
            
        elif test_target == 'quick':
            results = self.run_quick_tests()
            self.generate_report(results, "Quick Test Suite")
            
        elif test_target in self.categories:
            results = self.run_category(test_target)
            self.generate_report(results, f"{self.categories[test_target].name} Tests")
            
        else:
            print(f"FAIL Unknown test target: {test_target}")
            print(f"Available targets: all, quick, {', '.join(self.categories.keys())}")
            return False
        
        return results['failed'] == 0


def main():
    """Main entry point with command line argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Time Series Library"
    )
    
    parser.add_argument(
        'target',
        nargs='?',
        default='all',
        help='Test target: all, quick, or category name (dimension, unit, integration, modular, legacy, training)'
    )
    
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List available test categories'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what tests would run without executing them'
    )
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner()
    
    if args.list_categories:
        print("CLIPBOARD Available Test Categories:")
        print("="*50)
        for name, category in runner.categories.items():
            priority_icon = "" if category.priority == 1 else "" if category.priority == 2 else ""
            print(f"{priority_icon} {name}: {category.description}")
        print("\nLIGHTNING Special targets:")
        print("ROCKET all: Run all categories")
        print("LIGHTNING quick: Run essential tests only")
        return
    
    if args.dry_run:
        print("SEARCH Dry Run - Tests that would be executed:")
        print("="*50)
        
        if args.target == 'all':
            for name, category in runner.categories.items():
                print(f"\n {name}:")
                for path in category.paths:
                    print(f"    {path}")
        elif args.target == 'quick':
            print(f"\nLIGHTNING Quick Tests:")
            for test in runner.quick_tests:
                print(f"    {test}")
        elif args.target in runner.categories:
            category = runner.categories[args.target]
            print(f"\n {category.name}:")
            for path in category.paths:
                print(f"    {path}")
        return
    
    # Run tests
    success = runner.run(args.target)
    
    if success:
        print("\nPARTY All tests passed successfully!")
        sys.exit(0)
    else:
        print("\nFAIL Some tests failed. Check the report above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
