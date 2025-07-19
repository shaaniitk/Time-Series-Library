#!/usr/bin/env python3
"""
Modular Framework Test Suite Summary

This script provides an overview of all available tests and their purposes,
and can run specific test categories or the complete test suite.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestSuiteSummary:
    """Comprehensive test suite summary and runner."""
    
    def __init__(self):
        self.scripts_dir = Path(__file__).parent / "scripts"
        self.tests_dir = Path(__file__).parent
        
    def print_test_overview(self):
        """Print comprehensive overview of all available tests."""
        print("=" * 100)
        print("MODULAR AUTOFORMER FRAMEWORK - COMPLETE TEST SUITE OVERVIEW")
        print("=" * 100)
        
        print("\\n📋 TEST CATEGORIES:")
        print("-" * 50)
        
        # Component Level Tests
        print("\\n🔧 COMPONENT-LEVEL TESTS:")
        component_tests = [
            ("test_attention_components.py", "Tests all attention mechanisms (autocorrelation, adaptive, cross-resolution)"),
            ("test_decomposition_components.py", "Tests all decomposition methods (series, stable, learnable, wavelet)"),
            ("test_encoder_components.py", "Tests encoder components (standard, enhanced, hierarchical)"), 
            ("test_decoder_components.py", "Tests decoder components (standard, enhanced)"),
            ("test_sampling_components.py", "Tests sampling strategies (deterministic, Bayesian)"),
            ("test_output_head_components.py", "Tests output heads (standard, quantile)"),
            ("test_loss_components.py", "Tests loss functions (MSE, Bayesian, Bayesian-quantile)"),
            ("test_all_components.py", "Comprehensive test of ALL components together")
        ]
        
        for script, description in component_tests:
            status = "✅" if (self.scripts_dir / script).exists() else "⚠️"
            print(f"  {status} {script:<35} - {description}")
        
        # Integration Tests
        print("\\n🔗 INTEGRATION TESTS:")
        integration_tests = [
            ("test_gcli_configurations.py", "Tests all 7 GCLI custom configurations"),
            ("test_hf_integration.py", "Tests all 6 HF autoformer models"),
            ("test_unified_factory.py", "Tests unified factory functionality"),
            ("test_all_integrations.py", "Comprehensive integration test suite"),
            ("test_end_to_end.py", "Complete workflow testing"),
            ("test_model_comparison.py", "Model comparison and benchmarking")
        ]
        
        for script, description in integration_tests:
            status = "✅" if (self.scripts_dir / script).exists() else "⚠️"
            print(f"  {status} {script:<35} - {description}")
        
        # Performance Tests
        print("\\n⚡ PERFORMANCE & ROBUSTNESS TESTS:")
        performance_tests = [
            ("test_performance.py", "Performance benchmarks and memory usage"),
            ("test_robustness.py", "Edge cases and error handling"),
            ("run_smoke_tests.py", "Quick validation tests"),
            ("run_ci_tests.py", "CI/CD pipeline tests")
        ]
        
        for script, description in performance_tests:
            status = "✅" if (self.scripts_dir / script).exists() else "⚠️"
            print(f"  {status} {script:<35} - {description}")
        
        # Master Runners
        print("\\n🎯 MASTER TEST RUNNERS:")
        master_tests = [
            ("run_all_tests.py", "Runs complete test suite in sequence"),
            ("run_comprehensive_tests.py", "Legacy comprehensive test runner"),
            ("gcli_success_demo.py", "GCLI architecture success demonstration")
        ]
        
        for script, description in master_tests:
            status = "✅" if (self.scripts_dir / script).exists() or (project_root / script).exists() else "⚠️"
            print(f"  {status} {script:<35} - {description}")
        
        print("\\n📊 TEST STATISTICS:")
        print("-" * 50)
        
        # Count available tests
        component_count = sum(1 for script, _ in component_tests if (self.scripts_dir / script).exists())
        integration_count = sum(1 for script, _ in integration_tests if (self.scripts_dir / script).exists())
        performance_count = sum(1 for script, _ in performance_tests if (self.scripts_dir / script).exists())
        master_count = sum(1 for script, _ in master_tests 
                          if (self.scripts_dir / script).exists() or (project_root / script).exists())
        
        total_available = component_count + integration_count + performance_count + master_count
        total_planned = len(component_tests) + len(integration_tests) + len(performance_tests) + len(master_tests)
        
        print(f"Component Tests Available: {component_count}/{len(component_tests)}")
        print(f"Integration Tests Available: {integration_count}/{len(integration_tests)}")
        print(f"Performance Tests Available: {performance_count}/{len(performance_tests)}")
        print(f"Master Runners Available: {master_count}/{len(master_tests)}")
        print(f"Total Tests Available: {total_available}/{total_planned}")
        print(f"Completion: {(total_available/total_planned*100):.1f}%")
    
    def print_framework_coverage(self):
        """Print framework component coverage."""
        print("\\n🏗️ FRAMEWORK COMPONENT COVERAGE:")
        print("-" * 50)
        
        # GCLI Components
        print("\\n📦 GCLI Custom Components:")
        gcli_components = [
            ("Attention", ["AUTOCORRELATION_LAYER", "ADAPTIVE_AUTOCORRELATION_LAYER", "CROSS_RESOLUTION"]),
            ("Decomposition", ["SERIES_DECOMP", "STABLE_DECOMP", "LEARNABLE_DECOMP", "WAVELET_DECOMP"]),
            ("Encoders", ["STANDARD_ENCODER", "ENHANCED_ENCODER", "HIERARCHICAL_ENCODER"]),
            ("Decoders", ["STANDARD_DECODER", "ENHANCED_DECODER"]),
            ("Sampling", ["DETERMINISTIC", "BAYESIAN"]),
            ("Output Heads", ["STANDARD_HEAD", "QUANTILE"]),
            ("Loss Functions", ["MSE", "BAYESIAN", "BAYESIAN_QUANTILE"])
        ]
        
        for category, components in gcli_components:
            print(f"  {category}:")
            for component in components:
                print(f"    - {component}")
        
        # HF Models
        print("\\n🤗 HF Autoformer Models:")
        hf_models = [
            "hf_enhanced", "hf_bayesian", "hf_hierarchical", 
            "hf_quantile", "hf_enhanced_advanced", "hf_bayesian_production"
        ]
        
        for model in hf_models:
            print(f"  - {model}")
        
        # GCLI Configurations
        print("\\n⚙️ GCLI Configurations:")
        gcli_configs = [
            "standard", "enhanced", "fixed", "enhanced_fixed",
            "bayesian_enhanced", "hierarchical", "quantile_bayesian"
        ]
        
        for config in gcli_configs:
            print(f"  - {config}")
    
    def run_quick_validation(self):
        """Run quick validation to ensure basic functionality."""
        print("\\n🚀 QUICK VALIDATION TEST:")
        print("-" * 50)
        
        try:
            # Test import of key modules
            print("Testing module imports...")
            
            from models.unified_autoformer_factory import list_available_models
            available = list_available_models()
            print(f"✅ Unified factory - Custom: {len(available['custom'])}, HF: {len(available['hf'])}")
            
            from configs.modular_components import component_registry, register_all_components
            register_all_components()
            print("✅ Component registry initialized")
            
            from configs.schemas import ComponentType
            print(f"✅ Component types available: {len(list(ComponentType))}")
            
            print("\\n🎉 Quick validation PASSED - Framework is ready!")
            return True
            
        except Exception as e:
            print(f"\\n❌ Quick validation FAILED: {e}")
            return False
    
    def run_specific_test_category(self, category):
        """Run a specific test category."""
        category_scripts = {
            'components': [
                'test_attention_components.py',
                'test_decomposition_components.py',
                'test_all_components.py'
            ],
            'integration': [
                'test_unified_factory.py', 
                'test_all_integrations.py'
            ],
            'performance': [
                'test_performance.py',
                'run_smoke_tests.py'
            ],
            'all': [
                'run_all_tests.py'
            ]
        }
        
        if category not in category_scripts:
            print(f"❌ Unknown category: {category}")
            print(f"Available categories: {list(category_scripts.keys())}")
            return False
        
        scripts = category_scripts[category]
        print(f"\\n🏃 Running {category} tests...")
        
        success_count = 0
        total_count = 0
        
        for script in scripts:
            script_path = self.scripts_dir / script
            if script_path.exists():
                print(f"\\nRunning {script}...")
                try:
                    result = subprocess.run([sys.executable, str(script_path)], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ {script} - PASSED")
                        success_count += 1
                    else:
                        print(f"❌ {script} - FAILED")
                        if result.stderr:
                            print(f"Error: {result.stderr[:200]}...")
                    total_count += 1
                except Exception as e:
                    print(f"❌ {script} - ERROR: {e}")
                    total_count += 1
            else:
                print(f"⚠️  {script} - NOT FOUND")
        
        print(f"\\nCategory '{category}' results: {success_count}/{total_count} passed")
        return success_count == total_count
    
    def show_usage(self):
        """Show usage instructions."""
        print("\\n📖 USAGE:")
        print("-" * 50)
        print("python test_summary.py [command]")
        print("\\nCommands:")
        print("  overview     - Show complete test suite overview (default)")
        print("  quick        - Run quick validation test")
        print("  components   - Run component-level tests")
        print("  integration  - Run integration tests")
        print("  performance  - Run performance tests") 
        print("  all          - Run complete test suite")
        print("  help         - Show this help message")

def main():
    """Main function to handle command line arguments."""
    summary = TestSuiteSummary()
    
    command = sys.argv[1] if len(sys.argv) > 1 else "overview"
    
    if command == "overview":
        summary.print_test_overview()
        summary.print_framework_coverage()
        
    elif command == "quick":
        summary.print_test_overview()
        success = summary.run_quick_validation()
        return 0 if success else 1
        
    elif command in ["components", "integration", "performance", "all"]:
        success = summary.run_specific_test_category(command)
        return 0 if success else 1
        
    elif command == "help":
        summary.show_usage()
        
    else:
        print(f"❌ Unknown command: {command}")
        summary.show_usage()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
