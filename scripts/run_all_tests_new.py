#!/usr/bin/env python3
"""
Comprehensive Test Runner for Time Series Library
Organizes and runs all tests with proper categorization and reporting.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestRunner:
    """Comprehensive test runner for the Time Series Library"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.tests_dir = self.project_root / "tests"
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
    
    def get_test_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """Define test categories and their corresponding files"""
        return {
            "chronosx": {
                "description": "ChronosX Integration Tests",
                "tests": [
                    "tests/chronosx/test_chronosx_simple.py",
                    "tests/chronosx/test_chronos_x_comprehensive.py", 
                    "tests/chronosx/test_chronos_x_model_sizes.py",
                    "tests/chronosx/test_chronos_x_real_data.py",
                    "tests/chronosx/test_modular_autoformer_chronosx.py"
                ],
                "demos": [
                    "demo_models/chronos_x_simple_demo.py --smoke",
                    "demo_models/chronos_x_demo.py --smoke"
                ]
            },
            "modular_framework": {
                "description": "Modular Framework Tests",
                "tests": [
                    "tests/modular_framework/test_modular_framework_comprehensive.py",
                    "tests/test_component_registry.py",
                    "test_complete_modular_framework.py",
                    "test_modular_system.py"
                ]
            },
            "enhanced_models": {
                "description": "Enhanced Model Tests",
                "tests": [
                    "test_enhanced_hf_core.py",
                    "test_hf_enhanced_models.py", 
                    "test_enhanced_bayesian_model.py",
                    "test_complete_hf_suite.py",
                    "tests/test_enhanced_autoformer.py",
                    "tests/test_enhanced_models_ultralight.py"
                ]
            },
            "bayesian": {
                "description": "Bayesian Model Tests", 
                "tests": [
                    "test_bayesian_fix.py",
                    "tests/test_bayesian_loss_architecture.py",
                    "tests/test_quantile_bayesian.py",
                    "tests/test_simple_quantile_bayesian.py",
                    "test_production_bayesian.py"
                ]
            },
            "core_algorithms": {
                "description": "Core Algorithm Tests",
                "tests": [
                    "tests/core_algorithms/test_autocorrelation_comprehensive.py",
                    "tests/core_algorithms/test_autocorrelation_core.py",
                    "tests/test_series_decomposition.py",
                    "tests/test_multiwavelet_integration.py"
                ]
            },
            "integration": {
                "description": "Integration Tests",
                "tests": [
                    "tests/integration/test_integration.py",
                    "tests/test_end_to_end_workflows.py",
                    "test_advanced_integration.py",
                    "test_covariate_wavelet_integration.py"
                ]
            },
            "unit": {
                "description": "Unit Tests",
                "tests": [
                    "tests/unit/test_*.py"
                ]
            },
            "quick": {
                "description": "Quick Smoke Tests",
                "tests": [
                    "tests/chronosx/test_chronosx_simple.py",
                    "tests/simple_test.py",
                    "tests/minimal_test.py",
                    "simple_test.py"
                ]
            }
        }
    
    def find_existing_tests(self, test_paths: List[str]) -> List[str]:
        """Find which test files actually exist"""
        existing_tests = []
        for test_path in test_paths:
            # Handle glob patterns
            if '*' in test_path:
                import glob
                pattern = str(self.project_root / test_path)
                matching_files = glob.glob(pattern)
                existing_tests.extend([str(Path(f).relative_to(self.project_root)) for f in matching_files])
            else:
                full_path = self.project_root / test_path
                if full_path.exists():
                    existing_tests.append(test_path)
                else:
                    # Try without tests/ prefix if it's already included
                    alt_path = self.project_root / test_path.replace('tests/', '')
                    if alt_path.exists():
                        existing_tests.append(test_path.replace('tests/', ''))
        return existing_tests
    
    def run_test_file(self, test_path: str, timeout: int = 300) -> Tuple[bool, str, float]:
        """Run a single test file and return (success, output, duration)"""
        start_time = time.time()
        
        try:
            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            # Run the test
            result = subprocess.run(
                [sys.executable, test_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return success, output, duration
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return False, f"Test timed out after {timeout} seconds", duration
        except Exception as e:
            duration = time.time() - start_time
            return False, f"Error running test: {str(e)}", duration
        finally:
            os.chdir(original_cwd)
    
    def run_category_tests(self, category: str, test_info: Dict) -> Dict:
        """Run all tests in a category"""
        print(f"\n{'='*60}")
        print(f"🧪 Running {test_info['description']}")
        print(f"{'='*60}")
        
        category_results = {
            'description': test_info['description'],
            'tests': [],
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'duration': 0
        }
        
        # Get existing test files
        test_files = test_info.get('tests', [])
        existing_tests = self.find_existing_tests(test_files)
        
        if not existing_tests:
            print(f"   ⚠️ No test files found for {category}")
            return category_results
        
        print(f"   📋 Found {len(existing_tests)} test files")
        
        category_start_time = time.time()
        
        for test_file in existing_tests:
            print(f"\n   🔍 Running: {test_file}")
            
            success, output, duration = self.run_test_file(test_file)
            
            test_result = {
                'file': test_file,
                'success': success,
                'duration': duration,
                'output': output[:1000] + "..." if len(output) > 1000 else output  # Truncate long output
            }
            
            category_results['tests'].append(test_result)
            category_results['total'] += 1
            category_results['duration'] += duration
            
            if success:
                print(f"      ✅ PASSED ({duration:.2f}s)")
                category_results['passed'] += 1
                self.passed_tests += 1
            else:
                print(f"      ❌ FAILED ({duration:.2f}s)")
                # Show last few lines of error
                error_lines = output.split('\n')[-5:] if output else ['No output']
                for line in error_lines:
                    if line.strip():
                        print(f"         {line.strip()}")
                category_results['failed'] += 1
                self.failed_tests += 1
            
            self.total_tests += 1
        
        category_results['duration'] = time.time() - category_start_time
        
        # Print category summary
        print(f"\n   📊 {test_info['description']} Summary:")
        print(f"      Total: {category_results['total']}")
        print(f"      Passed: {category_results['passed']} ✅")
        print(f"      Failed: {category_results['failed']} ❌")
        print(f"      Duration: {category_results['duration']:.2f}s")
        
        return category_results
    
    def run_demos(self, category: str, test_info: Dict) -> Dict:
        """Run demo scripts in a category"""
        if 'demos' not in test_info:
            return {}
        
        print(f"\n   🎭 Running {category} demos...")
        
        demo_results = {
            'demos': [],
            'total': 0,
            'passed': 0,
            'failed': 0
        }
        
        existing_demos = self.find_existing_tests([d.split()[0] for d in test_info['demos']])
        
        for demo_entry in test_info['demos']:
            parts = demo_entry.split()
            demo_file = parts[0]
            demo_args = parts[1:]
            if demo_file not in existing_demos:
                continue
            print(f"      🎬 Running demo: {' '.join(parts)}")
            
            # Build command with extra args (like --smoke)
            start_time = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, demo_file, *demo_args],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_root
                )
                duration = time.time() - start_time
                success = result.returncode == 0
                output = result.stdout + result.stderr
            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                success = False
                output = f"Demo timed out after 120 seconds"
            
            demo_results['demos'].append({
                'file': demo_file,
                'success': success,
                'duration': duration
            })
            
            demo_results['total'] += 1
            
            if success:
                print(f"         ✅ Demo completed ({duration:.2f}s)")
                demo_results['passed'] += 1
            else:
                print(f"         ❌ Demo failed ({duration:.2f}s)")
                demo_results['failed'] += 1
        
        return demo_results
    
    def generate_report(self, results: Dict, output_file: str = None):
        """Generate comprehensive test report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            'categories': results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Detailed report saved to: {output_file}")
        
        return report
    
    def print_final_summary(self, results: Dict):
        """Print final test summary"""
        print(f"\n{'='*70}")
        print(f"🎯 FINAL TEST SUMMARY")
        print(f"{'='*70}")
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests} ✅")
        print(f"   Failed: {self.failed_tests} ❌")
        
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            print(f"   Success Rate: {success_rate:.1f}%")
            
            if success_rate == 100:
                print(f"\n🎉 ALL TESTS PASSED! Your codebase is in excellent shape!")
            elif success_rate >= 80:
                print(f"\n✅ Most tests passed! Some minor issues to address.")
            elif success_rate >= 50:
                print(f"\n⚠️ Some tests failed. Review the failed tests above.")
            else:
                print(f"\n❌ Many tests failed. Significant issues need attention.")
        
        print(f"\n🔍 Category Breakdown:")
        for category, result in results.items():
            if result.get('total', 0) > 0:
                cat_success = (result['passed'] / result['total']) * 100
                status = "✅" if cat_success == 100 else "⚠️" if cat_success >= 50 else "❌"
                print(f"   {status} {result['description']}: {result['passed']}/{result['total']} ({cat_success:.0f}%)")
    
    def run_tests(self, categories: List[str] = None, include_demos: bool = False):
        """Run tests for specified categories or all categories"""
        print(f"🚀 Time Series Library Test Runner")
        print(f"{'='*70}")
        print(f"📁 Project root: {self.project_root}")
        print(f"🧪 Tests directory: {self.tests_dir}")
        
        test_categories = self.get_test_categories()
        
        # Filter categories if specified
        if categories:
            test_categories = {k: v for k, v in test_categories.items() if k in categories}
        
        print(f"🏃 Running categories: {list(test_categories.keys())}")
        
        results = {}
        
        for category, test_info in test_categories.items():
            # Run main tests
            category_results = self.run_category_tests(category, test_info)
            results[category] = category_results
            
            # Run demos if requested
            if include_demos:
                demo_results = self.run_demos(category, test_info)
                if demo_results:
                    category_results['demos'] = demo_results
        
        # Generate and print final report
        self.print_final_summary(results)
        
        # Save detailed report
        report_file = self.project_root / "test_results" / f"test_report_{int(time.time())}.json"
        report_file.parent.mkdir(exist_ok=True)
        self.generate_report(results, str(report_file))
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run Time Series Library tests')
    parser.add_argument('--categories', '-c', nargs='+', 
                       choices=['chronosx', 'modular_framework', 'enhanced_models', 'bayesian', 
                               'core_algorithms', 'integration', 'unit', 'quick'],
                       help='Test categories to run (default: all)')
    parser.add_argument('--demos', '-d', action='store_true', 
                       help='Include demo scripts')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run only quick smoke tests')
    parser.add_argument('--list-categories', '-l', action='store_true',
                       help='List available test categories')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.list_categories:
        print("📋 Available test categories:")
        for category, info in runner.get_test_categories().items():
            print(f"   {category}: {info['description']}")
        return
    
    # Determine which categories to run
    if args.quick:
        categories = ['quick']
    elif args.categories:
        categories = args.categories
    else:
        categories = None  # Run all
    
    # Run tests
    start_time = time.time()
    results = runner.run_tests(categories, include_demos=args.demos)
    total_duration = time.time() - start_time
    
    print(f"\n⏱️ Total execution time: {total_duration:.2f} seconds")
    
    # Exit with appropriate code
    sys.exit(0 if runner.failed_tests == 0 else 1)


if __name__ == "__main__":
    main()
