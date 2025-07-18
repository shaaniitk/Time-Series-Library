#!/usr/bin/env python3
"""
Comprehensive Test Runner for Reorganized Test Suite

This script runs all tests in the newly organized test structure, providing
comprehensive validation of the entire system with proper test categorization.

Test Organization:
- Unit Tests: Individual component and utility testing
- Integration Tests: End-to-end workflows and component interaction
- Dimension Tests: Specific focus on dimension management issues
- Modular Framework Tests: Modular component system validation
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class TestRunner:
    """Comprehensive test runner for the reorganized test suite"""
    
    def __init__(self):
        self.project_root = Path(project_root)
        self.tests_root = self.project_root / "tests"
        
        # Test categories and their paths
        self.test_categories = {
            'unit': {
                'path': self.tests_root / 'unit',
                'description': 'Unit tests for individual components',
                'priority': 1
            },
            'integration_dimension': {
                'path': self.tests_root / 'integration' / 'dimension_tests',
                'description': 'Dimension management integration tests',
                'priority': 2
            },
            'integration_end_to_end': {
                'path': self.tests_root / 'integration' / 'end_to_end',
                'description': 'End-to-end workflow tests',
                'priority': 3
            },
            'integration_modular': {
                'path': self.tests_root / 'integration' / 'modular_framework',
                'description': 'Modular framework tests',
                'priority': 4
            },
            'legacy': {
                'path': self.tests_root,
                'description': 'Legacy tests (root level)',
                'priority': 5
            }
        }
        
        self.results = {}
        self.start_time = None
        
    def discover_tests(self, category: str) -> List[Path]:
        """Discover test files in a category"""
        category_info = self.test_categories[category]
        test_path = category_info['path']
        
        if not test_path.exists():
            return []
        
        test_files = []
        
        if category == 'legacy':
            # For legacy, only get root-level test files
            for file_path in test_path.glob('test_*.py'):
                if file_path.is_file():
                    test_files.append(file_path)
        else:
            # For organized categories, search recursively
            for file_path in test_path.rglob('test_*.py'):
                if file_path.is_file():
                    test_files.append(file_path)
        
        return sorted(test_files)
    
    def run_test_file(self, test_file: Path, timeout: int = 300) -> Tuple[bool, str, float]:
        """Run a single test file"""
        print(f"    TEST Running {test_file.name}...")
        
        start_time = time.time()
        
        try:
            # Try to run as a module first
            if self._has_main_function(test_file):
                result = self._run_as_module(test_file, timeout)
            else:
                result = self._run_with_pytest(test_file, timeout)
                
            duration = time.time() - start_time
            
            if result['success']:
                print(f"      PASS {test_file.name} PASSED ({duration:.2f}s)")
                return True, result['output'], duration
            else:
                print(f"      FAIL {test_file.name} FAILED ({duration:.2f}s)")
                return False, result['output'], duration
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Exception running {test_file.name}: {e}"
            print(f"      FAIL {test_file.name} ERROR ({duration:.2f}s)")
            return False, error_msg, duration
    
    def _has_main_function(self, test_file: Path) -> bool:
        """Check if test file has a main function"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return '__main__' in content and 'def run_' in content
        except:
            return False
    
    def _run_as_module(self, test_file: Path, timeout: int) -> Dict:
        """Run test file as a Python module"""
        try:
            # Import and run the test module
            spec = importlib.util.spec_from_file_location("test_module", test_file)
            module = importlib.util.module_from_spec(spec)
            
            # Capture output
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                spec.loader.exec_module(module)
            
            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            
            return {'success': True, 'output': output}
            
        except Exception as e:
            return {'success': False, 'output': str(e)}
    
    def _run_with_pytest(self, test_file: Path, timeout: int) -> Dict:
        """Run test file with pytest"""
        try:
            cmd = [sys.executable, '-m', 'pytest', str(test_file), '-v', '--tb=short']
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            return {'success': success, 'output': output}
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'output': f'Test timed out after {timeout} seconds'}
        except Exception as e:
            return {'success': False, 'output': str(e)}
    
    def run_category(self, category: str) -> Dict:
        """Run all tests in a category"""
        category_info = self.test_categories[category]
        print(f"\n Running {category} tests: {category_info['description']}")
        print("=" * 70)
        
        test_files = self.discover_tests(category)
        
        if not test_files:
            print(f"  WARN No test files found in {category_info['path']}")
            return {
                'category': category,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'duration': 0,
                'details': []
            }
        
        print(f"  CHART Found {len(test_files)} test file(s)")
        
        passed = 0
        failed = 0
        total_duration = 0
        details = []
        
        for test_file in test_files:
            try:
                success, output, duration = self.run_test_file(test_file)
                total_duration += duration
                
                if success:
                    passed += 1
                else:
                    failed += 1
                
                details.append({
                    'file': test_file.name,
                    'success': success,
                    'duration': duration,
                    'output': output
                })
                
            except KeyboardInterrupt:
                print(f"\n  WARN Tests interrupted by user")
                break
            except Exception as e:
                print(f"  FAIL Error running {test_file.name}: {e}")
                failed += 1
                details.append({
                    'file': test_file.name,
                    'success': False,
                    'duration': 0,
                    'output': str(e)
                })
        
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n  CHART {category} Results:")
        print(f"     Total: {total}, Passed: {passed}, Failed: {failed}")
        print(f"     Success Rate: {success_rate:.1f}%, Duration: {total_duration:.2f}s")
        
        return {
            'category': category,
            'total': total,
            'passed': passed,
            'failed': failed,
            'duration': total_duration,
            'success_rate': success_rate,
            'details': details
        }
    
    def run_all(self, categories: Optional[List[str]] = None) -> Dict:
        """Run all test categories"""
        print("ROCKET Running Comprehensive Test Suite")
        print("=" * 80)
        print(f" Project Root: {self.project_root}")
        print(f"TEST Tests Root: {self.tests_root}")
        
        self.start_time = time.time()
        
        if categories is None:
            # Run in priority order
            categories = sorted(
                self.test_categories.keys(),
                key=lambda x: self.test_categories[x]['priority']
            )
        
        all_results = {}
        total_passed = 0
        total_failed = 0
        total_duration = 0
        
        for category in categories:
            if category not in self.test_categories:
                print(f"WARN Unknown category: {category}")
                continue
            
            try:
                result = self.run_category(category)
                all_results[category] = result
                
                total_passed += result['passed']
                total_failed += result['failed']
                total_duration += result['duration']
                
            except KeyboardInterrupt:
                print(f"\nWARN Test run interrupted by user")
                break
            except Exception as e:
                print(f"FAIL Error running category {category}: {e}")
                all_results[category] = {
                    'category': category,
                    'total': 0,
                    'passed': 0,
                    'failed': 1,
                    'duration': 0,
                    'error': str(e)
                }
                total_failed += 1
        
        # Summary
        total_tests = total_passed + total_failed
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 80)
        print("CHART COMPREHENSIVE TEST RESULTS")
        print("=" * 80)
        
        for category, result in all_results.items():
            status = "PASS" if result['failed'] == 0 else "FAIL"
            print(f"{status} {category:20} | "
                  f"Tests: {result['total']:3} | "
                  f"Passed: {result['passed']:3} | "
                  f"Failed: {result['failed']:3} | "
                  f"Rate: {result.get('success_rate', 0):5.1f}%")
        
        print("-" * 80)
        print(f"TARGET OVERALL SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Success Rate: {overall_success_rate:.1f}%")
        print(f"   Total Duration: {total_duration:.2f}s")
        
        if total_failed == 0:
            print("\nPARTY PARTY PARTY ALL TESTS PASSED! PARTY PARTY PARTY")
        else:
            print(f"\nWARN {total_failed} test(s) failed. Check individual outputs for details.")
        
        return {
            'overall': {
                'total': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': overall_success_rate,
                'duration': total_duration
            },
            'categories': all_results
        }
    
    def run_quick(self) -> Dict:
        """Run quick tests (unit tests only)"""
        print("LIGHTNING Running Quick Tests (Unit Tests Only)")
        return self.run_all(['unit'])
    
    def run_dimension_focus(self) -> Dict:
        """Run dimension-focused tests"""
        print(" Running Dimension-Focused Tests")
        return self.run_all(['unit', 'integration_dimension'])
    
    def run_integration_only(self) -> Dict:
        """Run integration tests only"""
        print(" Running Integration Tests Only")
        return self.run_all(['integration_dimension', 'integration_end_to_end', 'integration_modular'])
    
    def generate_report(self, results: Dict, output_file: Optional[str] = None):
        """Generate a detailed test report"""
        report_lines = []
        report_lines.append("# Comprehensive Test Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall summary
        overall = results['overall']
        report_lines.append("## Overall Results")
        report_lines.append(f"- Total Tests: {overall['total']}")
        report_lines.append(f"- Passed: {overall['passed']}")
        report_lines.append(f"- Failed: {overall['failed']}")
        report_lines.append(f"- Success Rate: {overall['success_rate']:.1f}%")
        report_lines.append(f"- Duration: {overall['duration']:.2f}s")
        report_lines.append("")
        
        # Category details
        report_lines.append("## Category Details")
        for category, result in results['categories'].items():
            report_lines.append(f"### {category}")
            report_lines.append(f"- Description: {self.test_categories[category]['description']}")
            report_lines.append(f"- Tests: {result['total']}")
            report_lines.append(f"- Passed: {result['passed']}")
            report_lines.append(f"- Failed: {result['failed']}")
            report_lines.append(f"- Success Rate: {result.get('success_rate', 0):.1f}%")
            report_lines.append("")
            
            # Individual test details
            if 'details' in result:
                report_lines.append("#### Individual Tests")
                for detail in result['details']:
                    status = "PASS" if detail['success'] else "FAIL"
                    report_lines.append(f"- {status} {detail['file']} ({detail['duration']:.2f}s)")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"PAGE Report saved to: {output_file}")
        
        return report_content


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument('--mode', choices=['all', 'quick', 'dimension', 'integration'], 
                       default='all', help='Test mode to run')
    parser.add_argument('--categories', nargs='+', 
                       choices=['unit', 'integration_dimension', 'integration_end_to_end', 
                               'integration_modular', 'legacy'],
                       help='Specific categories to run')
    parser.add_argument('--report', type=str, help='Generate report to file')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.categories:
            results = runner.run_all(args.categories)
        elif args.mode == 'quick':
            results = runner.run_quick()
        elif args.mode == 'dimension':
            results = runner.run_dimension_focus()
        elif args.mode == 'integration':
            results = runner.run_integration_only()
        else:
            results = runner.run_all()
        
        if args.report:
            runner.generate_report(results, args.report)
        
        # Exit with proper code
        exit_code = 0 if results['overall']['failed'] == 0 else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nWARN Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL Test runner error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
