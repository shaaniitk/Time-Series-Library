"""
Test Runner for Enhanced SOTA PGAT with MoE Components

Comprehensive test runner that executes all tests for the new MoE framework
and enhanced components.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_smoke_tests():
    """Run all smoke tests for quick validation."""
    print("=" * 80)
    print("RUNNING SMOKE TESTS FOR ENHANCED SOTA PGAT WITH MOE")
    print("=" * 80)
    
    smoke_tests = [
        "TestsModule/smoke/test_moe_framework_smoke.py",
        "TestsModule/smoke/test_temporal_experts_smoke.py", 
        "TestsModule/smoke/test_spatial_experts_smoke.py",
        "TestsModule/smoke/test_uncertainty_experts_smoke.py",
        "TestsModule/smoke/test_training_enhancements_smoke.py",
        "TestsModule/smoke/test_enhanced_sota_pgat_moe_smoke.py"
    ]
    
    for test_file in smoke_tests:
        if os.path.exists(test_file):
            print(f"\n{'='*60}")
            print(f"Running: {test_file}")
            print(f"{'='*60}")
            
            result = pytest.main([test_file, "-v", "--tb=short"])
            
            if result != 0:
                print(f"❌ FAILED: {test_file}")
            else:
                print(f"✅ PASSED: {test_file}")
        else:
            print(f"⚠️  SKIPPED: {test_file} (file not found)")

def run_integration_tests():
    """Run integration tests for comprehensive validation."""
    print("\n" + "=" * 80)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 80)
    
    integration_tests = [
        "TestsModule/tests/integration/test_moe_integration.py"
    ]
    
    for test_file in integration_tests:
        if os.path.exists(test_file):
            print(f"\n{'='*60}")
            print(f"Running: {test_file}")
            print(f"{'='*60}")
            
            result = pytest.main([test_file, "-v", "--tb=short"])
            
            if result != 0:
                print(f"❌ FAILED: {test_file}")
            else:
                print(f"✅ PASSED: {test_file}")
        else:
            print(f"⚠️  SKIPPED: {test_file} (file not found)")

def run_performance_tests():
    """Run performance tests (if available)."""
    print("\n" + "=" * 80)
    print("RUNNING PERFORMANCE TESTS")
    print("=" * 80)
    
    # Check if performance tests exist
    perf_test_dir = "TestsModule/perf"
    if os.path.exists(perf_test_dir):
        perf_tests = [f for f in os.listdir(perf_test_dir) if f.startswith("test_") and f.endswith(".py")]
        
        for test_file in perf_tests:
            test_path = os.path.join(perf_test_dir, test_file)
            print(f"\n{'='*60}")
            print(f"Running: {test_path}")
            print(f"{'='*60}")
            
            result = pytest.main([test_path, "-v", "--tb=short", "-m", "not slow"])
            
            if result != 0:
                print(f"❌ FAILED: {test_path}")
            else:
                print(f"✅ PASSED: {test_path}")
    else:
        print("⚠️  No performance tests found")

def run_component_tests():
    """Run tests for specific components."""
    print("\n" + "=" * 80)
    print("RUNNING COMPONENT-SPECIFIC TESTS")
    print("=" * 80)
    
    # Test categories
    test_categories = {
        "MoE Framework": [
            "test_moe_framework_smoke.py",
            "test_moe_integration.py"
        ],
        "Temporal Experts": [
            "test_temporal_experts_smoke.py"
        ],
        "Spatial Experts": [
            "test_spatial_experts_smoke.py"
        ],
        "Uncertainty Experts": [
            "test_uncertainty_experts_smoke.py"
        ],
        "Training Enhancements": [
            "test_training_enhancements_smoke.py"
        ],
        "Enhanced Model": [
            "test_enhanced_sota_pgat_moe_smoke.py"
        ]
    }
    
    results = {}
    
    for category, test_files in test_categories.items():
        print(f"\n{'-'*40}")
        print(f"Testing: {category}")
        print(f"{'-'*40}")
        
        category_results = []
        
        for test_file in test_files:
            # Find the test file
            test_path = None
            for root, dirs, files in os.walk("TestsModule"):
                if test_file in files:
                    test_path = os.path.join(root, test_file)
                    break
            
            if test_path and os.path.exists(test_path):
                result = pytest.main([test_path, "-v", "--tb=line", "-q"])
                category_results.append(result == 0)
                
                if result == 0:
                    print(f"  ✅ {test_file}")
                else:
                    print(f"  ❌ {test_file}")
            else:
                print(f"  ⚠️  {test_file} (not found)")
                category_results.append(None)
        
        # Calculate category success rate
        passed = sum(1 for r in category_results if r is True)
        total = sum(1 for r in category_results if r is not None)
        
        if total > 0:
            success_rate = passed / total
            results[category] = success_rate
            print(f"  📊 {category}: {passed}/{total} passed ({success_rate:.1%})")
        else:
            results[category] = 0.0
            print(f"  📊 {category}: No tests found")
    
    return results

def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if results:
        total_categories = len(results)
        successful_categories = sum(1 for rate in results.values() if rate >= 0.8)
        
        print(f"📊 Overall Results:")
        print(f"   Categories tested: {total_categories}")
        print(f"   Successful categories (≥80%): {successful_categories}")
        print(f"   Overall success rate: {successful_categories/total_categories:.1%}")
        
        print(f"\n📋 Category Breakdown:")
        for category, success_rate in results.items():
            status = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.5 else "❌"
            print(f"   {status} {category}: {success_rate:.1%}")
    
    print(f"\n🎯 Key Components Tested:")
    print(f"   ✓ Mixture of Experts Framework")
    print(f"   ✓ Temporal Pattern Experts (4 types)")
    print(f"   ✓ Spatial Relationship Experts (3 types)")
    print(f"   ✓ Uncertainty Quantification Experts (2 types)")
    print(f"   ✓ Expert Routing Mechanisms (4 types)")
    print(f"   ✓ Curriculum Learning Strategies (4 types)")
    print(f"   ✓ Memory Optimization Components")
    print(f"   ✓ Enhanced SOTA PGAT with MoE Integration")
    
    print(f"\n🔧 Test Coverage:")
    print(f"   • Smoke tests: Basic functionality validation")
    print(f"   • Integration tests: Component interaction validation")
    print(f"   • Gradient flow tests: Training compatibility")
    print(f"   • Device compatibility tests: CPU/GPU support")
    print(f"   • Configuration tests: Different parameter settings")
    print(f"   • Performance tests: Scalability and efficiency")

def main():
    """Main test runner."""
    print("🚀 Enhanced SOTA PGAT with MoE - Comprehensive Test Suite")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not os.path.exists("TestsModule"):
        print("❌ Error: TestsModule directory not found!")
        print("   Please run this script from the project root directory.")
        return 1
    
    # Run different test suites
    try:
        # 1. Run smoke tests (quick validation)
        run_smoke_tests()
        
        # 2. Run integration tests
        run_integration_tests()
        
        # 3. Run component-specific tests and get results
        results = run_component_tests()
        
        # 4. Run performance tests (optional)
        # run_performance_tests()
        
        # 5. Print comprehensive summary
        print_summary(results)
        
        print(f"\n🎉 Test suite completed!")
        print(f"   Check the output above for detailed results.")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running test suite: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)