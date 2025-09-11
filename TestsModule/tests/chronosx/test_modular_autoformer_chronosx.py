#!/usr/bin/env python3
"""
Comprehensive Test Suite for ModularAutoformer with ChronosX Integration
Tests the modular backbone functionality and component integration.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
import time
import traceback

# Add the project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.modular_autoformer import ModularAutoformer
from layers.modular.core.registry import component_registry
# from layers.modular.core.example_components import register_example_components  # Not needed


class ModularAutoformerTester:
    """Test suite for ModularAutoformer with ChronosX backbone"""
    
    def __init__(self):
        self.results = {}
        self.test_data = None
        
    def generate_test_data(self, seq_len=96, pred_len=24, num_features=1, batch_size=2):
        """Generate synthetic time series data for testing"""
        print("CHART Generating synthetic test data...")
        
        # Generate more realistic time series with trends and seasonality
        t = np.linspace(0, 4*np.pi, seq_len + pred_len)
        
        # Create multiple time series in batch
        data = []
        for b in range(batch_size):
            # Add trend, seasonality, and noise
            trend = 0.1 * t
            seasonal = 0.5 * np.sin(t) + 0.3 * np.cos(2*t)
            noise = 0.1 * np.random.randn(len(t))
            series = trend + seasonal + noise + b * 0.2  # Slight offset per batch
            
            if num_features > 1:
                # Add additional features
                extra_features = np.random.randn(len(t), num_features - 1) * 0.1
                series = np.column_stack([series, extra_features])
            else:
                series = series.reshape(-1, 1)
                
            data.append(series)
        
        data = np.stack(data)  # Shape: (batch_size, seq_len + pred_len, num_features)
        
        # Split into encoder and decoder parts
        x_enc = data[:, :seq_len, :]
        x_dec = data[:, seq_len-pred_len:, :]  # Overlapping for autoformer style
        
        # Create time marks (dummy)
        x_mark_enc = np.zeros((batch_size, seq_len, 4))  # Common time features
        x_mark_dec = np.zeros((batch_size, pred_len*2, 4))
        
        # Convert to tensors
        self.test_data = {
            'x_enc': torch.FloatTensor(x_enc),
            'x_mark_enc': torch.FloatTensor(x_mark_enc),
            'x_dec': torch.FloatTensor(x_dec),
            'x_mark_dec': torch.FloatTensor(x_mark_dec),
            'true_future': torch.FloatTensor(data[:, seq_len:, :])
        }
        
        print(f"   PASS Generated data shapes:")
        print(f"      x_enc: {self.test_data['x_enc'].shape}")
        print(f"      x_dec: {self.test_data['x_dec'].shape}")
        print(f"      true_future: {self.test_data['true_future'].shape}")
        
        return self.test_data
    
    def create_test_config(self, use_backbone=False, backbone_type=None):
        """Create test configuration for ModularAutoformer"""
        
        config = Namespace()
        
        # Basic task configuration
        config.task_name = 'long_term_forecast'
        config.seq_len = 96
        config.label_len = 48
        config.pred_len = 24
        
        # Model dimensions
        config.d_model = 64
        config.enc_in = 1
        config.dec_in = 1
        config.c_out = 1
        config.c_out_evaluation = 1
        
        # Embedding configuration
        config.embed = 'timeF'
        config.freq = 'h'
        config.dropout = 0.1
        
        # Backbone configuration
        config.use_backbone_component = use_backbone
        config.backbone_type = backbone_type
        
        if use_backbone and backbone_type:
            # ChronosX specific parameters
            config.backbone_params = {
                'model_size': 'tiny',  # tiny, small, base, large
                'use_uncertainty': True,
                'num_samples': 20
            }
            
            # Processor for backbone output
            config.processor_type = 'time_domain'  # Use a simple processor
        
        # Traditional autoformer components (used when not using backbone)
        config.encoder_type = 'standard'
        config.decoder_type = 'standard'
        config.attention_type = 'autocorrelation_layer'
        config.decomposition_type = 'series_decomp'
        config.sampling_type = 'deterministic'
        config.output_head_type = 'standard'
        config.loss_function_type = 'mse'
        
        # Component parameters
        config.encoder_params = {
            'd_model': config.d_model,
            'n_heads': 4,
            'e_layers': 2,
            'd_ff': 128,
            'factor': 1,
            'dropout': 0.1,
            'activation': 'gelu'
        }
        
        config.decoder_params = {
            'd_model': config.d_model,
            'n_heads': 4,
            'd_layers': 1,
            'd_ff': 128,
            'factor': 1,
            'dropout': 0.1,
            'activation': 'gelu'
        }
        
        config.attention_params = {
            'factor': 1,
            'dropout': 0.1,
            'd_model': config.d_model,
            'n_heads': 4
        }
        
        config.decomposition_params = {
            'kernel_size': 25
        }
        
        config.init_decomposition_params = {
            'kernel_size': 25
        }
        
        config.sampling_params = {}
        
        config.output_head_params = {
            'd_model': config.d_model,
            'c_out': config.c_out,
            'dropout': 0.1
        }
        
        config.loss_params = {}
        
        return config
    
    def test_traditional_autoformer(self):
        """Test traditional autoformer functionality"""
        print("\nTOOL Testing Traditional Autoformer...")
        
        try:
            # Create traditional config
            config = self.create_test_config(use_backbone=False)
            
            # Initialize registry
            registry = component_registry
            
            # Create model
            model = ModularAutoformer(config)
            model.eval()
            
            print(f"   PASS Model created successfully")
            print(f"      Architecture: {model.get_component_info()['architecture']}")
            
            # Test forward pass
            with torch.no_grad():
                start_time = time.time()
                output = model(
                    self.test_data['x_enc'],
                    self.test_data['x_mark_enc'],
                    self.test_data['x_dec'],
                    self.test_data['x_mark_dec']
                )
                inference_time = time.time() - start_time
            
            print(f"   PASS Forward pass successful")
            print(f"      Output shape: {output.shape}")
            print(f"      Inference time: {inference_time:.3f}s")
            print(f"      Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Calculate simple metrics
            if output.shape == self.test_data['true_future'].shape:
                mse = torch.nn.functional.mse_loss(output, self.test_data['true_future'])
                mae = torch.nn.functional.l1_loss(output, self.test_data['true_future'])
                print(f"      MSE: {mse.item():.6f}")
                print(f"      MAE: {mae.item():.6f}")
            
            self.results['traditional'] = {
                'success': True,
                'output_shape': output.shape,
                'inference_time': inference_time,
                'component_info': model.get_component_info()
            }
            
            return True
            
        except Exception as e:
            print(f"   FAIL Traditional autoformer test failed: {str(e)}")
            print(f"      Traceback: {traceback.format_exc()}")
            self.results['traditional'] = {'success': False, 'error': str(e)}
            return False
    
    def test_chronosx_backbone(self):
        """Test ChronosX backbone integration"""
        print("\nCRYSTAL Testing ChronosX Backbone Integration...")
        
        try:
            # Create ChronosX config
            config = self.create_test_config(use_backbone=True, backbone_type='chronos_x')
            
            # Initialize registry and register components
            registry = unified_registry
            register_example_components(registry)
            
            # Create model with ChronosX backbone
            model = ModularAutoformer(config)
            model.eval()
            
            print(f"   PASS Model with ChronosX backbone created successfully")
            
            # Get backbone info
            backbone_info = model.get_backbone_info()
            print(f"      Backbone type: {backbone_info['backbone_type']}")
            print(f"      Supports uncertainty: {backbone_info['supports_uncertainty']}")
            
            # Test forward pass
            with torch.no_grad():
                start_time = time.time()
                output = model(
                    self.test_data['x_enc'],
                    self.test_data['x_mark_enc'],
                    self.test_data['x_dec'],
                    self.test_data['x_mark_dec']
                )
                inference_time = time.time() - start_time
            
            print(f"   PASS Forward pass with ChronosX successful")
            print(f"      Output shape: {output.shape}")
            print(f"      Inference time: {inference_time:.3f}s")
            print(f"      Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Test uncertainty if available
            uncertainty_results = model.get_uncertainty_results()
            if uncertainty_results:
                print(f"   TARGET Uncertainty quantification available:")
                print(f"      Prediction shape: {uncertainty_results['prediction'].shape}")
                if 'std' in uncertainty_results:
                    print(f"      Std shape: {uncertainty_results['std'].shape}")
                    print(f"      Avg uncertainty: {uncertainty_results['std'].mean().item():.6f}")
            
            # Calculate metrics if shapes match
            if output.shape == self.test_data['true_future'].shape:
                mse = torch.nn.functional.mse_loss(output, self.test_data['true_future'])
                mae = torch.nn.functional.l1_loss(output, self.test_data['true_future'])
                print(f"      MSE: {mse.item():.6f}")
                print(f"      MAE: {mae.item():.6f}")
            
            self.results['chronosx'] = {
                'success': True,
                'output_shape': output.shape,
                'inference_time': inference_time,
                'backbone_info': backbone_info,
                'uncertainty_available': uncertainty_results is not None,
                'component_info': model.get_component_info()
            }
            
            return True
            
        except Exception as e:
            print(f"   FAIL ChronosX backbone test failed: {str(e)}")
            print(f"      Traceback: {traceback.format_exc()}")
            self.results['chronosx'] = {'success': False, 'error': str(e)}
            return False
    
    def test_uncertainty_variants(self):
        """Test different ChronosX uncertainty configurations"""
        print("\nDICE Testing ChronosX Uncertainty Variants...")
        
        variants = [
            ('chronos_x_tiny', 'Tiny model with uncertainty'),
            ('chronos_x_uncertainty', 'Uncertainty-focused model'),
        ]
        
        for variant_type, description in variants:
            print(f"\n   Testing {description} ({variant_type})...")
            
            try:
                # Create variant config
                config = self.create_test_config(use_backbone=True, backbone_type=variant_type)
                
                # Initialize registry
                registry = unified_registry
                register_example_components(registry)
                
                # Create model
                model = ModularAutoformer(config)
                model.eval()
                
                # Test forward pass
                with torch.no_grad():
                    start_time = time.time()
                    output = model(
                        self.test_data['x_enc'],
                        self.test_data['x_mark_enc'],
                        self.test_data['x_dec'],
                        self.test_data['x_mark_dec']
                    )
                    inference_time = time.time() - start_time
                
                print(f"      PASS {description} successful")
                print(f"         Output shape: {output.shape}")
                print(f"         Inference time: {inference_time:.3f}s")
                
                # Check uncertainty
                uncertainty_results = model.get_uncertainty_results()
                if uncertainty_results:
                    print(f"         TARGET Uncertainty available")
                    if 'std' in uncertainty_results:
                        avg_uncertainty = uncertainty_results['std'].mean().item()
                        print(f"         CHART Avg uncertainty: {avg_uncertainty:.6f}")
                
                self.results[f'variant_{variant_type}'] = {
                    'success': True,
                    'output_shape': output.shape,
                    'inference_time': inference_time,
                    'uncertainty_available': uncertainty_results is not None
                }
                
            except Exception as e:
                print(f"      FAIL {description} failed: {str(e)}")
                self.results[f'variant_{variant_type}'] = {'success': False, 'error': str(e)}
    
    def test_performance_comparison(self):
        """Compare performance between traditional and ChronosX approaches"""
        print("\nLIGHTNING Performance Comparison...")
        
        if 'traditional' in self.results and 'chronosx' in self.results:
            trad_time = self.results['traditional'].get('inference_time', 0)
            chronos_time = self.results['chronosx'].get('inference_time', 0)
            
            print(f"   CHART Inference Time Comparison:")
            print(f"      Traditional: {trad_time:.3f}s")
            print(f"      ChronosX: {chronos_time:.3f}s")
            
            if trad_time > 0 and chronos_time > 0:
                speedup = trad_time / chronos_time
                print(f"      Speedup: {speedup:.2f}x {'(ChronosX faster)' if speedup > 1 else '(Traditional faster)'}")
        
        print(f"\n   BRAIN Model Capabilities:")
        for test_name, result in self.results.items():
            if result.get('success'):
                uncertainty = result.get('uncertainty_available', False)
                print(f"      {test_name}: PASS {'+ Uncertainty' if uncertainty else 'Basic'}")
            else:
                print(f"      {test_name}: FAIL Failed")
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("ROCKET ModularAutoformer + ChronosX Integration Test Suite")
        print("=" * 70)
        
        # Generate test data
        self.generate_test_data()
        
        # Run individual tests
        success_count = 0
        total_tests = 0
        
        tests = [
            ('Traditional Autoformer', self.test_traditional_autoformer),
            ('ChronosX Backbone', self.test_chronosx_backbone),
            ('Uncertainty Variants', self.test_uncertainty_variants),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                if result:
                    success_count += 1
                total_tests += 1
            except Exception as e:
                print(f"FAIL Test {test_name} crashed: {str(e)}")
                total_tests += 1
        
        # Performance comparison
        self.test_performance_comparison()
        
        # Final summary
        print(f"\nPARTY TEST SUITE COMPLETE!")
        print("=" * 70)
        print(f"CHART Results: {success_count}/{total_tests} tests passed")
        
        if success_count == total_tests:
            print("PASS ALL TESTS PASSED! ModularAutoformer + ChronosX integration is working perfectly!")
        elif success_count > 0:
            print("WARN PARTIAL SUCCESS: Some tests passed, check failed tests above")
        else:
            print("FAIL ALL TESTS FAILED: Check installation and dependencies")
        
        print(f"\nIDEA Key Achievements:")
        for test_name, result in self.results.items():
            if result.get('success'):
                print(f"   PASS {test_name}: Working")
                if result.get('uncertainty_available'):
                    print(f"      TARGET Uncertainty quantification available")
        
        return success_count == total_tests


if __name__ == "__main__":
    print("MICROSCOPE Starting ModularAutoformer + ChronosX Integration Tests...\n")
    
    tester = ModularAutoformerTester()
    success = tester.run_all_tests()
    
    if success:
        print(f"\nROCKET Ready for production! Your modular architecture with ChronosX is fully functional.")
    else:
        print(f"\nTOOL Some issues found. Check the test results above for details.")
    
    sys.exit(0 if success else 1)
