"""
Test Discovery and Runner Configuration

This file provides test discovery, runner configuration, and comprehensive
test execution for the modular Autoformer framework.
"""

import pytest
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def collect_test_modules():
    """Collect all test modules in the tests directory"""
    test_modules = []
    tests_dir = Path(__file__).parent
    
    for test_file in tests_dir.glob("test_*.py"):
        if test_file.name != "test_runner.py":  # Exclude this file
            module_name = test_file.stem
            test_modules.append(module_name)
    
    return sorted(test_modules)


def run_test_suite(test_modules=None, verbose=True, capture='no'):
    """
    Run the complete test suite for the modular framework
    
    Args:
        test_modules: List of specific test modules to run (None for all)
        verbose: Whether to run in verbose mode
        capture: Capture mode ('no', 'sys', 'fd')
    """
    if test_modules is None:
        test_modules = collect_test_modules()
    
    pytest_args = ["-v" if verbose else "-q"]
    
    # Add capture mode
    if capture == 'no':
        pytest_args.append("-s")
    elif capture == 'sys':
        pytest_args.append("--capture=sys")
    elif capture == 'fd':
        pytest_args.append("--capture=fd")
    
    # Add test modules
    tests_dir = Path(__file__).parent
    for module in test_modules:
        test_file = tests_dir / f"{module}.py"
        if test_file.exists():
            pytest_args.append(str(test_file))
    
    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend(["--cov=models", "--cov=utils", "--cov-report=term-missing"])
    except ImportError:
        print("pytest-cov not available, skipping coverage reporting")
    
    # Run tests
    return pytest.main(pytest_args)


def run_quick_tests():
    """Run a subset of quick tests for development"""
    quick_modules = [
        "test_modular_framework_comprehensive",
        "test_component_registry"
    ]
    
    return run_test_suite(test_modules=quick_modules, verbose=True)


def run_integration_tests():
    """Run integration and end-to-end tests"""
    integration_modules = [
        "test_migration_strategy",
        "test_end_to_end_workflows"
    ]
    
    return run_test_suite(test_modules=integration_modules, verbose=True)


def run_performance_tests():
    """Run performance and benchmark tests"""
    performance_modules = [
        "test_performance_benchmarks"
    ]
    
    return run_test_suite(test_modules=performance_modules, verbose=True)


def create_test_report():
    """Generate comprehensive test report"""
    import json
    import time
    from datetime import datetime
    
    report = {
        "test_run_info": {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "project_root": str(project_root)
        },
        "test_modules": {},
        "summary": {}
    }
    
    # Run each test module individually to get detailed results
    all_modules = collect_test_modules()
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for module in all_modules:
        print(f"\nRunning {module}...")
        
        # Run individual module
        start_time = time.time()
        result = run_test_suite([module], verbose=False, capture='sys')
        end_time = time.time()
        
        # Categorize result
        if result == 0:
            status = "PASSED"
            total_passed += 1
        elif result == 5:  # No tests collected
            status = "SKIPPED"
            total_skipped += 1
        else:
            status = "FAILED"
            total_failed += 1
        
        report["test_modules"][module] = {
            "status": status,
            "exit_code": result,
            "duration": round(end_time - start_time, 2)
        }
    
    # Summary
    report["summary"] = {
        "total_modules": len(all_modules),
        "passed": total_passed,
        "failed": total_failed,
        "skipped": total_skipped,
        "success_rate": round(total_passed / len(all_modules) * 100, 1) if all_modules else 0
    }
    
    # Save report
    report_file = project_root / "test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"Total Modules: {report['summary']['total_modules']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Skipped: {report['summary']['skipped']}")
    print(f"Success Rate: {report['summary']['success_rate']}%")
    print(f"Report saved to: {report_file}")
    
    return report


def validate_test_environment():
    """Validate that the test environment is properly set up"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        issues.append(f"Python 3.7+ required, found {sys.version}")
    
    # Check required packages
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'pytest'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Required package '{package}' not found")
    
    # Check optional packages
    optional_packages = {
        'pytest_cov': 'Coverage reporting',
        'memory_profiler': 'Memory profiling',
        'psutil': 'System monitoring'
    }
    
    missing_optional = []
    for package, description in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(f"{package} ({description})")
    
    # Check project structure
    expected_dirs = ['models', 'utils', 'data_provider', 'exp']
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            issues.append(f"Expected directory '{dir_name}' not found")
    
    # Report validation results
    print("Test Environment Validation")
    print("=" * 40)
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("✅ All required components found")
    
    if missing_optional:
        print("\nOPTIONAL PACKAGES MISSING:")
        for package in missing_optional:
            print(f"  ⚠️  {package}")
        print("  (These packages provide additional functionality but are not required)")
    
    print(f"\nProject Root: {project_root}")
    print(f"Python Version: {sys.version}")
    
    return len(issues) == 0


def main():
    """Main CLI interface for test runner"""
    parser = argparse.ArgumentParser(description="Modular Autoformer Test Runner")
    
    parser.add_argument(
        "command",
        choices=["all", "quick", "integration", "performance", "report", "validate"],
        help="Test suite to run"
    )
    
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific test modules to run"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode"
    )
    
    parser.add_argument(
        "--capture",
        choices=["no", "sys", "fd"],
        default="no",
        help="Output capture mode"
    )
    
    args = parser.parse_args()
    
    # Validate environment first
    if not validate_test_environment():
        print("\n❌ Environment validation failed. Please fix issues before running tests.")
        return 1
    
    print("\n" + "=" * 60)
    print("MODULAR AUTOFORMER TEST SUITE")
    print("=" * 60)
    
    # Execute based on command
    if args.command == "validate":
        return 0  # Already validated above
    
    elif args.command == "all":
        if args.modules:
            result = run_test_suite(args.modules, not args.quiet, args.capture)
        else:
            result = run_test_suite(verbose=not args.quiet, capture=args.capture)
    
    elif args.command == "quick":
        result = run_quick_tests()
    
    elif args.command == "integration":
        result = run_integration_tests()
    
    elif args.command == "performance":
        result = run_performance_tests()
    
    elif args.command == "report":
        report = create_test_report()
        result = 0 if report["summary"]["failed"] == 0 else 1
    
    else:
        parser.print_help()
        return 1
    
    return result


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
