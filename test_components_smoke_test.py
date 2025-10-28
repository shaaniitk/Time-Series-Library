#!/usr/bin/env python3
"""
SMOKE TEST: Quick 4-batch validation for all component configurations
Verifies no training errors without full training time
"""

import sys
import os
import yaml
import time
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_systematic_tests_production_workflow import ProductionWorkflowTester

class SmokeTestTester(ProductionWorkflowTester):
    """Modified tester that stops training after 4 batches"""
    
    def __init__(self):
        super().__init__(results_dir='results/smoke_test_results')
        
        # Override base config for ultra-fast smoke testing
        self.base_config.update({
            'train_epochs': 1,  # Just 1 epoch
            'batch_size': 4,    # Slightly larger batch for faster testing
            'max_train_batches': 4,  # Stop after 4 batches
            'patience': 1,      # No early stopping needed
            'log_interval': 1,  # Log every batch
            'save_checkpoints': False,  # No checkpoints needed
            'enable_memory_diagnostics': False,  # Clean output
        })
    
    def create_smoke_test_config(self, config_name: str, **component_overrides) -> str:
        """Create a smoke test config that stops after 4 batches"""
        config = self.base_config.copy()
        config.update(component_overrides)
        
        # Create a modified training script that stops after 4 batches
        config_path = self.results_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return str(config_path)
    
    def run_smoke_test(self, config_name: str, **component_overrides) -> dict:
        """Run a 4-batch smoke test"""
        print(f"\n🔥 Smoke Test: {config_name}")
        print("-" * 50)
        
        # Create config
        config_path = self.create_smoke_test_config(config_name, **component_overrides)
        
        start_time = time.time()
        try:
            # Import and patch the production training function
            from scripts.train.train_celestial_production import train_celestial_pgat_production
            
            # Monkey patch to stop after 4 batches
            original_train_epoch = None
            
            def limited_train_epoch(*args, **kwargs):
                """Modified train_epoch that stops after 4 batches"""
                # Get the original function if not already stored
                nonlocal original_train_epoch
                if original_train_epoch is None:
                    from scripts.train.train_celestial_production import train_epoch
                    original_train_epoch = train_epoch
                
                # Call original but with modified loader
                train_loader = args[1]  # Second argument is train_loader
                
                # Create a limited loader that only yields 4 batches
                class LimitedLoader:
                    def __init__(self, original_loader, max_batches=4):
                        self.original_loader = original_loader
                        self.max_batches = max_batches
                    
                    def __iter__(self):
                        count = 0
                        for batch in self.original_loader:
                            if count >= self.max_batches:
                                break
                            yield batch
                            count += 1
                    
                    def __len__(self):
                        return min(self.max_batches, len(self.original_loader))
                
                # Replace the train_loader with limited version
                limited_loader = LimitedLoader(train_loader, max_batches=4)
                modified_args = list(args)
                modified_args[1] = limited_loader
                
                return original_train_epoch(*modified_args, **kwargs)
            
            # Temporarily replace the train_epoch function
            import scripts.train.train_celestial_production as prod_module
            original_func = prod_module.train_epoch
            prod_module.train_epoch = limited_train_epoch
            
            try:
                success = train_celestial_pgat_production(config_path)
                training_time = time.time() - start_time
                
                if success:
                    print(f"✅ SUCCESS: {config_name} (4 batches completed)")
                    print(f"   • Training time: {training_time:.1f}s")
                    
                    result = {
                        'status': 'success',
                        'config_name': config_name,
                        'training_time': training_time,
                        'batches_completed': 4,
                        'config_path': config_path,
                        'smoke_test': True
                    }
                else:
                    print(f"❌ FAILED: {config_name}")
                    result = {
                        'status': 'failed',
                        'config_name': config_name,
                        'training_time': training_time,
                        'config_path': config_path,
                        'error': 'Production training returned False',
                        'smoke_test': True
                    }
            
            finally:
                # Restore original function
                prod_module.train_epoch = original_func
            
        except Exception as e:
            training_time = time.time() - start_time
            print(f"💥 ERROR: {config_name} - {e}")
            result = {
                'status': 'error',
                'config_name': config_name,
                'training_time': training_time,
                'config_path': config_path,
                'error': str(e),
                'smoke_test': True
            }
        
        return result
    
    def run_all_smoke_tests(self):
        """Run smoke tests for all configurations"""
        print("🔥 SMOKE TEST SUITE - 4 Batches Per Configuration")
        print("=" * 60)
        print("Quickly validates that all component combinations work without errors")
        print("Each test runs for exactly 4 training batches then stops")
        print("=" * 60)
        
        # All configurations to test
        test_configs = [
            ("01_Baseline", {
                "use_multi_scale_patching": False,
                "use_hierarchical_mapper": False,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("02_MultiScale_Only", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": False,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("03_MultiScale_Hierarchical", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": False,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("04_MultiScale_Hier_Stochastic", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": False,
                "use_mixture_decoder": False
            }),
            ("05_MultiScale_Hier_Stoch_Graph", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": False
            }),
            ("06_Full_Configuration", {
                "use_multi_scale_patching": True,
                "use_hierarchical_mapper": True,
                "use_stochastic_learner": True,
                "use_gated_graph_combiner": True,
                "use_mixture_decoder": True
            })
        ]
        
        results = {}
        successful_tests = []
        failed_tests = []
        
        for config_name, component_overrides in test_configs:
            result = self.run_smoke_test(config_name, **component_overrides)
            results[config_name] = result
            
            if result['status'] == 'success':
                successful_tests.append(config_name)
            else:
                failed_tests.append((config_name, result.get('error', 'Unknown')))
            
            # Brief pause between tests
            time.sleep(1)
        
        # Save results
        self.save_results(results, 'smoke_test_results.json')
        
        # Print summary
        print("\n" + "=" * 60)
        print("🔥 SMOKE TEST SUMMARY")
        print("=" * 60)
        
        print(f"\n✅ SUCCESSFUL TESTS ({len(successful_tests)}/{len(test_configs)}):")
        for test in successful_tests:
            print(f"   • {test}")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}/{len(test_configs)}):")
            for test, error in failed_tests:
                print(f"   • {test}: {error}")
        
        success_rate = len(successful_tests) / len(test_configs) * 100
        print(f"\n📊 SUCCESS RATE: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 ALL CONFIGURATIONS WORK! Ready for full systematic testing.")
        elif success_rate >= 80:
            print("✅ Most configurations work. Minor issues to fix.")
        else:
            print("⚠️  Multiple configuration issues detected. Need debugging.")
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        
        return results


def main():
    """Run smoke tests"""
    print("🔥 Enhanced SOTA PGAT - Component Smoke Testing")
    print("=" * 50)
    
    tester = SmokeTestTester()
    results = tester.run_all_smoke_tests()
    
    return results


if __name__ == "__main__":
    results = main()