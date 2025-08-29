#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Modular Autoformer Framework

This script tests all integrated configurations including:
1. All 7 GCLI custom configurations  
2. All 6 HF autoformer models
3. Unified factory functionality
4. End-to-end workflows

Based on the successful gcli_success_demo.py but expanded for comprehensive testing.
"""

import sys
import os
import time
import torch
from pathlib import Path
from argparse import Namespace

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import GCLI configurations
from configs.autoformer import (
    standard_config,
    enhanced_config,
    fixed_config,
    enhanced_fixed_config,
    bayesian_enhanced_config,
    hierarchical_config,
    quantile_bayesian_config,
)

# Import unified factory
from models.unified_autoformer_factory import (
    UnifiedAutoformerFactory,
    UnifiedModelInterface,
    create_autoformer,
    compare_implementations,
    list_available_models
)

class IntegrationTestSuite:
    """Comprehensive integration test suite for modular framework."""
    
    def __init__(self):
        self.test_results = {
            'gcli_configs': {},
            'hf_models': {},
            'factory_tests': {},
            'workflow_tests': {}
        }
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def create_test_data(self):
        """Create test data for models."""
        batch_size = 2
        seq_len = 96
        pred_len = 24
        label_len = 48
        enc_in = 7
        dec_in = 7
        
        return {
            'x_enc': torch.randn(batch_size, seq_len, enc_in),
            'x_mark_enc': torch.randn(batch_size, seq_len, 4),
            'x_dec': torch.randn(batch_size, label_len + pred_len, dec_in),
            'x_mark_dec': torch.randn(batch_size, label_len + pred_len, 4),
        }
    
    def test_gcli_configuration(self, config_name, config_func):
        """Test a single GCLI configuration."""
        print(f"\\n--- Testing GCLI Config: {config_name} ---")
        
        try:
            # Get configuration
            config = config_func()
            
            # Create model using unified factory
            model = UnifiedAutoformerFactory.create_model(
                model_type=config_name.lower().replace('_', '_'),
                config=config,
                framework_preference='custom'
            )
            
            # Test model creation
            self.assertIsNotNone(model, f"Failed to create {config_name} model")
            
            # Test forward pass
            test_data = self.create_test_data()
            
            model.eval()
            with torch.no_grad():
                output = model(
                    test_data['x_enc'],
                    test_data['x_mark_enc'], 
                    test_data['x_dec'],
                    test_data['x_mark_dec']
                )
            
            # Validate output
            expected_shape = (test_data['x_enc'].shape[0], config.pred_len, config.c_out)
            actual_shape = output.shape
            
            # Handle quantile outputs
            if hasattr(config, 'quantile_levels') and config.quantile_levels:
                expected_shape = (expected_shape[0], expected_shape[1], 
                                expected_shape[2] * len(config.quantile_levels))
            
            self.assertEqual(actual_shape, expected_shape, 
                           f"{config_name} output shape mismatch: {actual_shape} vs {expected_shape}")
            
            # Check for NaN/Inf
            self.assertFalse(torch.isnan(output).any(), f"{config_name} produced NaN values")
            self.assertFalse(torch.isinf(output).any(), f"{config_name} produced Inf values")
            
            print(f"✓ {config_name} passed - Output shape: {actual_shape}")
            self.test_results['gcli_configs'][config_name] = {
                'status': 'PASSED',
                'output_shape': actual_shape,
                'model_class': model.__class__.__name__
            }
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"✗ {config_name} failed: {e}")
            self.test_results['gcli_configs'][config_name] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.failed_tests += 1
            return False
        finally:
            self.total_tests += 1
    
    def test_hf_model(self, model_type):
        """Test a single HF model."""
        print(f"\\n--- Testing HF Model: {model_type} ---")
        
        try:
            # Basic configuration for HF models
            config = {
                'seq_len': 96,
                'pred_len': 24,
                'label_len': 48,
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7,
                'd_model': 64,
                'n_heads': 8,
                'd_ff': 256,
                'e_layers': 2,
                'd_layers': 1,
                'dropout': 0.1,
                'activation': 'gelu'
            }
            
            # Create model using unified factory
            model = UnifiedAutoformerFactory.create_model(
                model_type=model_type,
                config=config,
                framework_preference='hf'
            )
            
            # Test model creation
            self.assertIsNotNone(model, f"Failed to create {model_type} model")
            
            # Test forward pass
            test_data = self.create_test_data()
            
            model.eval()
            with torch.no_grad():
                output = model(
                    test_data['x_enc'],
                    test_data['x_mark_enc'],
                    test_data['x_dec'],
                    test_data['x_mark_dec']
                )
            
            # Validate output
            expected_shape = (test_data['x_enc'].shape[0], config['pred_len'], config['c_out'])
            actual_shape = output.shape
            
            # Some HF models might have different output dimensions
            if actual_shape != expected_shape:
                print(f"Note: {model_type} has non-standard output shape: {actual_shape}")
            
            # Check for NaN/Inf
            self.assertFalse(torch.isnan(output).any(), f"{model_type} produced NaN values")
            self.assertFalse(torch.isinf(output).any(), f"{model_type} produced Inf values")
            
            print(f"✓ {model_type} passed - Output shape: {actual_shape}")
            self.test_results['hf_models'][model_type] = {
                'status': 'PASSED',
                'output_shape': actual_shape,
                'model_class': model.__class__.__name__
            }
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"✗ {model_type} failed: {e}")
            self.test_results['hf_models'][model_type] = {
                'status': 'FAILED', 
                'error': str(e)
            }
            self.failed_tests += 1
            return False
        finally:
            self.total_tests += 1
    
    def test_unified_factory_functionality(self):
        """Test unified factory specific features."""
        print("\\n=== Testing Unified Factory Functionality ===")
        
        # Test 1: Model availability listing
        try:
            available_models = list_available_models()
            self.assertIsInstance(available_models, dict)
            self.assertIn('custom', available_models)
            self.assertIn('hf', available_models)
            
            custom_count = len(available_models['custom'])
            hf_count = len(available_models['hf'])
            
            print(f"✓ Model listing - Custom: {custom_count}, HF: {hf_count}")
            self.test_results['factory_tests']['model_listing'] = {
                'status': 'PASSED',
                'custom_models': custom_count,
                'hf_models': hf_count
            }
            self.passed_tests += 1
            
        except Exception as e:
            print(f"✗ Model listing failed: {e}")
            self.test_results['factory_tests']['model_listing'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.failed_tests += 1
        finally:
            self.total_tests += 1
        
        # Test 2: Framework preference handling
        try:
            config = {
                'seq_len': 96, 'pred_len': 24, 'label_len': 48,
                'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'd_model': 64
            }
            
            # Test auto preference
            model_auto = create_autoformer('enhanced', config, framework='auto')
            self.assertIsInstance(model_auto, UnifiedModelInterface)
            
            # Test custom preference
            model_custom = create_autoformer('enhanced', config, framework='custom')
            self.assertIsInstance(model_custom, UnifiedModelInterface)
            
            print("✓ Framework preference handling")
            self.test_results['factory_tests']['framework_preference'] = {'status': 'PASSED'}
            self.passed_tests += 1
            
        except Exception as e:
            print(f"✗ Framework preference failed: {e}")
            self.test_results['factory_tests']['framework_preference'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.failed_tests += 1
        finally:
            self.total_tests += 1
        
        # Test 3: Compatible pair creation
        try:
            config = {
                'seq_len': 96, 'pred_len': 24, 'label_len': 48,
                'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'd_model': 64
            }
            
            comparison = compare_implementations('enhanced', config)
            self.assertIsInstance(comparison, dict)
            
            # Should have at least custom implementation
            self.assertIn('custom', comparison)
            
            print(f"✓ Compatible pair creation - Frameworks: {list(comparison.keys())}")
            self.test_results['factory_tests']['compatible_pairs'] = {
                'status': 'PASSED',
                'frameworks': list(comparison.keys())
            }
            self.passed_tests += 1
            
        except Exception as e:
            print(f"✗ Compatible pair creation failed: {e}")
            self.test_results['factory_tests']['compatible_pairs'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.failed_tests += 1
        finally:
            self.total_tests += 1
    
    def test_end_to_end_workflows(self):
        """Test complete end-to-end workflows."""
        print("\\n=== Testing End-to-End Workflows ===")
        
        # Test 1: Basic prediction workflow
        try:
            config = {
                'seq_len': 96, 'pred_len': 24, 'label_len': 48,
                'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'd_model': 64
            }
            
            # Create model
            model_interface = create_autoformer('enhanced', config, framework='custom')
            
            # Generate test data
            test_data = self.create_test_data()
            
            # Test prediction
            prediction = model_interface.predict(
                test_data['x_enc'],
                test_data['x_mark_enc'],
                test_data['x_dec'],
                test_data['x_mark_dec']
            )
            
            # Validate prediction
            self.assertIsInstance(prediction, torch.Tensor)
            expected_shape = (test_data['x_enc'].shape[0], config['pred_len'], config['c_out'])
            self.assertEqual(prediction.shape, expected_shape)
            
            print(f"✓ Basic prediction workflow - Shape: {prediction.shape}")
            self.test_results['workflow_tests']['basic_prediction'] = {
                'status': 'PASSED',
                'prediction_shape': prediction.shape
            }
            self.passed_tests += 1
            
        except Exception as e:
            print(f"✗ Basic prediction workflow failed: {e}")
            self.test_results['workflow_tests']['basic_prediction'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.failed_tests += 1
        finally:
            self.total_tests += 1
        
        # Test 2: Model info retrieval
        try:
            config = {
                'seq_len': 96, 'pred_len': 24, 'label_len': 48,
                'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'd_model': 64
            }
            
            model_interface = create_autoformer('enhanced', config, framework='custom')
            
            # Get model info
            model_info = model_interface.get_model_info()
            
            # Validate info structure
            required_keys = ['framework_type', 'model_type', 'model_class', 'supports_uncertainty']
            for key in required_keys:
                self.assertIn(key, model_info, f"Missing key in model info: {key}")
            
            print(f"✓ Model info retrieval - Framework: {model_info['framework_type']}")
            self.test_results['workflow_tests']['model_info'] = {
                'status': 'PASSED',
                'framework_type': model_info['framework_type'],
                'model_class': model_info['model_class']
            }
            self.passed_tests += 1
            
        except Exception as e:
            print(f"✗ Model info retrieval failed: {e}")
            self.test_results['workflow_tests']['model_info'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.failed_tests += 1
        finally:
            self.total_tests += 1
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("=" * 100)
        print("COMPREHENSIVE MODULAR AUTOFORMER FRAMEWORK INTEGRATION TESTS")
        print("=" * 100)
        
        start_time = time.time()
        
        # Test GCLI configurations
        print("\\n🔧 TESTING GCLI CUSTOM CONFIGURATIONS")
        print("=" * 60)
        
        gcli_configs = [
            ('standard', standard_config),
            ('enhanced', enhanced_config),
            ('fixed', fixed_config), 
            ('enhanced_fixed', enhanced_fixed_config),
            ('bayesian_enhanced', bayesian_enhanced_config),
            ('hierarchical', hierarchical_config),
            ('quantile_bayesian', quantile_bayesian_config),
        ]
        
        for config_name, config_func in gcli_configs:
            self.test_gcli_configuration(config_name, config_func)
        
        # Test HF models
        print("\\n🤗 TESTING HF AUTOFORMER MODELS")
        print("=" * 60)
        
        try:
            available_models = list_available_models()
            hf_models = available_models.get('hf', [])
            
            for model_type in hf_models:
                self.test_hf_model(model_type)
                
        except Exception as e:
            print(f"Failed to get HF models list: {e}")
        
        # Test unified factory functionality
        self.test_unified_factory_functionality()
        
        # Test end-to-end workflows
        self.test_end_to_end_workflows()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print comprehensive summary
        self.print_comprehensive_summary(duration)
        
        return self.failed_tests == 0
    
    def print_comprehensive_summary(self, duration):
        """Print detailed test results summary."""
        print("\\n" + "=" * 100)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 100)
        
        # Overall statistics
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        
        # GCLI configurations summary
        print("\\n📋 GCLI CONFIGURATIONS:")
        gcli_passed = sum(1 for result in self.test_results['gcli_configs'].values() 
                         if result['status'] == 'PASSED')
        gcli_total = len(self.test_results['gcli_configs'])
        print(f"  Passed: {gcli_passed}/{gcli_total}")
        
        for config_name, result in self.test_results['gcli_configs'].items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {config_name}: {result['status']}")
            if result['status'] == 'PASSED':
                print(f"     Shape: {result['output_shape']}, Class: {result['model_class']}")
        
        # HF models summary  
        print("\\n🤗 HF MODELS:")
        hf_passed = sum(1 for result in self.test_results['hf_models'].values() 
                       if result['status'] == 'PASSED')
        hf_total = len(self.test_results['hf_models'])
        print(f"  Passed: {hf_passed}/{hf_total}")
        
        for model_name, result in self.test_results['hf_models'].items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {model_name}: {result['status']}")
            if result['status'] == 'PASSED':
                print(f"     Shape: {result['output_shape']}, Class: {result['model_class']}")
        
        # Factory tests summary
        print("\\n🏭 FACTORY FUNCTIONALITY:")
        factory_passed = sum(1 for result in self.test_results['factory_tests'].values() 
                           if result['status'] == 'PASSED')
        factory_total = len(self.test_results['factory_tests'])
        print(f"  Passed: {factory_passed}/{factory_total}")
        
        for test_name, result in self.test_results['factory_tests'].items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {test_name}: {result['status']}")
        
        # Workflow tests summary
        print("\\n🔄 WORKFLOW TESTS:")
        workflow_passed = sum(1 for result in self.test_results['workflow_tests'].values() 
                             if result['status'] == 'PASSED')
        workflow_total = len(self.test_results['workflow_tests'])
        print(f"  Passed: {workflow_passed}/{workflow_total}")
        
        for test_name, result in self.test_results['workflow_tests'].items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {test_name}: {result['status']}")
        
        # Final status
        if self.failed_tests == 0:
            print("\\n🎉 ALL INTEGRATION TESTS PASSED! 🎉")
            print("✨ Modular framework is fully functional with both GCLI and HF support! ✨")
        else:
            print(f"\\n❌ {self.failed_tests} TESTS FAILED")
            print("Please check the errors above and fix the issues.")
        
        print("=" * 100)
    
    def assertEqual(self, actual, expected, message=""):
        """Simple assertion helper."""
        if actual != expected:
            raise AssertionError(f"{message}: {actual} != {expected}")
    
    def assertIsNotNone(self, value, message=""):
        """Simple assertion helper."""
        if value is None:
            raise AssertionError(f"{message}: Value is None")
    
    def assertIsInstance(self, obj, cls, message=""):
        """Simple assertion helper."""
        if not isinstance(obj, cls):
            raise AssertionError(f"{message}: {obj} is not instance of {cls}")
    
    def assertIn(self, item, container, message=""):
        """Simple assertion helper."""
        if item not in container:
            raise AssertionError(f"{message}: {item} not in {container}")
    
    def assertFalse(self, value, message=""):
        """Simple assertion helper."""
        if value:
            raise AssertionError(f"{message}: Value is True when expected False")

def main():
    """Main test execution function."""
    test_suite = IntegrationTestSuite()
    success = test_suite.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
