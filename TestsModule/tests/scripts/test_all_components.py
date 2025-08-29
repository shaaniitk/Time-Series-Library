#!/usr/bin/env python3
"""
Comprehensive Modular Framework Component Tests

This script tests all individual components of the modular autoformer framework,
including both GCLI-based custom components and HF integration components.

Test Categories:
1. Attention Components
2. Decomposition Components  
3. Encoder Components
4. Decoder Components
5. Sampling Components
6. Output Head Components
7. Loss Function Components
"""

import sys
import os
import unittest
import torch
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import component registries and factories
from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType
from models.unified_autoformer_factory import UnifiedAutoformerFactory, list_available_models

class TestModularComponents(unittest.TestCase):
    """Test suite for all modular framework components."""
    
    def setUp(self):
        """Set up test parameters and data."""
        # Ensure all components are registered
        register_all_components()
        
        # Test parameters
        self.batch_size = 2
        self.seq_len = 96
        self.pred_len = 24
        self.label_len = 48
        self.d_model = 64
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.n_heads = 8
        self.d_ff = 256
        self.e_layers = 2
        self.d_layers = 1
        
        # Create test tensors
        self.x_enc = torch.randn(self.batch_size, self.seq_len, self.enc_in)
        self.x_mark_enc = torch.randn(self.batch_size, self.seq_len, 4)  # time features
        self.x_dec = torch.randn(self.batch_size, self.label_len + self.pred_len, self.dec_in)
        self.x_mark_dec = torch.randn(self.batch_size, self.label_len + self.pred_len, 4)
        
        # Model tensors (after embedding)
        self.model_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.model_cross = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_attention_components(self):
        """Test all attention components."""
        print("\\n=== Testing Attention Components ===")
        
        attention_types = [
            ComponentType.AUTOCORRELATION_LAYER,
            ComponentType.ADAPTIVE_AUTOCORRELATION_LAYER,
            ComponentType.CROSS_RESOLUTION
        ]
        
        for attention_type in attention_types:
            with self.subTest(attention_type=attention_type):
                print(f"Testing {attention_type.value}...")
                
                # Get component from registry
                component_info = component_registry.get_component(attention_type)
                self.assertIsNotNone(component_info, f"Component {attention_type.value} not found in registry")
                
                # Create component instance
                params = {
                    'd_model': self.d_model,
                    'n_heads': self.n_heads,
                    'dropout': 0.1,
                    'factor': 1
                }
                
                try:
                    component = component_info.factory(**params)
                    
                    # Test forward pass
                    with torch.no_grad():
                        if attention_type == ComponentType.CROSS_RESOLUTION:
                            # Cross attention needs different inputs
                            output, weights = component(
                                self.model_input, self.model_cross, self.model_cross
                            )
                        else:
                            # Self attention
                            output, weights = component(
                                self.model_input, self.model_input, self.model_input
                            )
                    
                    # Validate output shape
                    expected_shape = (self.batch_size, self.seq_len, self.d_model)
                    self.assertEqual(output.shape, expected_shape, 
                                   f"{attention_type.value} output shape mismatch")
                    
                    # Check for NaN or Inf
                    self.assertFalse(torch.isnan(output).any(), 
                                   f"{attention_type.value} produced NaN values")
                    self.assertFalse(torch.isinf(output).any(), 
                                   f"{attention_type.value} produced Inf values")
                    
                    print(f"✓ {attention_type.value} passed")
                    
                except Exception as e:
                    print(f"✗ {attention_type.value} failed: {e}")
                    raise

    def test_decomposition_components(self):
        """Test all decomposition components."""
        print("\\n=== Testing Decomposition Components ===")
        
        decomp_types = [
            ComponentType.SERIES_DECOMP,
            ComponentType.STABLE_DECOMP,
            ComponentType.LEARNABLE_DECOMP,
            ComponentType.WAVELET_DECOMP
        ]
        
        for decomp_type in decomp_types:
            with self.subTest(decomp_type=decomp_type):
                print(f"Testing {decomp_type.value}...")
                
                # Get component from registry
                component_info = component_registry.get_component(decomp_type)
                self.assertIsNotNone(component_info, f"Component {decomp_type.value} not found in registry")
                
                # Create component instance with appropriate parameters
                if decomp_type == ComponentType.SERIES_DECOMP:
                    params = {'kernel_size': 25}
                elif decomp_type == ComponentType.STABLE_DECOMP:
                    params = {'kernel_size': 25}
                elif decomp_type == ComponentType.LEARNABLE_DECOMP:
                    params = {'input_dim': self.d_model}
                elif decomp_type == ComponentType.WAVELET_DECOMP:
                    params = {'seq_len': self.seq_len, 'd_model': self.d_model}
                
                try:
                    component = component_info.factory(**params)
                    
                    # Test forward pass
                    with torch.no_grad():
                        seasonal, trend = component(self.model_input)
                    
                    # Validate output shapes
                    expected_shape = self.model_input.shape
                    self.assertEqual(seasonal.shape, expected_shape, 
                                   f"{decomp_type.value} seasonal shape mismatch")
                    self.assertEqual(trend.shape, expected_shape, 
                                   f"{decomp_type.value} trend shape mismatch")
                    
                    # Test reconstruction (seasonal + trend should approximate original)
                    reconstruction = seasonal + trend
                    self.assertEqual(reconstruction.shape, expected_shape,
                                   f"{decomp_type.value} reconstruction shape mismatch")
                    
                    # Check for NaN or Inf
                    self.assertFalse(torch.isnan(seasonal).any(), 
                                   f"{decomp_type.value} seasonal produced NaN")
                    self.assertFalse(torch.isnan(trend).any(), 
                                   f"{decomp_type.value} trend produced NaN")
                    
                    print(f"✓ {decomp_type.value} passed")
                    
                except Exception as e:
                    print(f"✗ {decomp_type.value} failed: {e}")
                    raise

    def test_encoder_components(self):
        """Test all encoder components."""
        print("\\n=== Testing Encoder Components ===")
        
        encoder_types = [
            ComponentType.STANDARD_ENCODER,
            ComponentType.ENHANCED_ENCODER,
            ComponentType.HIERARCHICAL_ENCODER
        ]
        
        for encoder_type in encoder_types:
            with self.subTest(encoder_type=encoder_type):
                print(f"Testing {encoder_type.value}...")
                
                # Get component from registry
                component_info = component_registry.get_component(encoder_type)
                self.assertIsNotNone(component_info, f"Component {encoder_type.value} not found in registry")
                
                # Create component instance
                params = {
                    'd_model': self.d_model,
                    'n_heads': self.n_heads,
                    'd_ff': self.d_ff,
                    'e_layers': self.e_layers,
                    'dropout': 0.1,
                    'activation': 'gelu'
                }
                
                try:
                    component = component_info.factory(**params)
                    
                    # Test forward pass
                    with torch.no_grad():
                        if encoder_type == ComponentType.HIERARCHICAL_ENCODER:
                            # Hierarchical encoder may return multiple scales
                            result = component(self.model_input, attn_mask=None)
                            if isinstance(result, tuple):
                                output, scales = result
                            else:
                                output = result
                        else:
                            output, attns = component(self.model_input, attn_mask=None)
                    
                    # Validate output shape
                    expected_shape = self.model_input.shape
                    self.assertEqual(output.shape, expected_shape, 
                                   f"{encoder_type.value} output shape mismatch")
                    
                    # Check for NaN or Inf
                    self.assertFalse(torch.isnan(output).any(), 
                                   f"{encoder_type.value} produced NaN values")
                    self.assertFalse(torch.isinf(output).any(), 
                                   f"{encoder_type.value} produced Inf values")
                    
                    print(f"✓ {encoder_type.value} passed")
                    
                except Exception as e:
                    print(f"✗ {encoder_type.value} failed: {e}")
                    raise

    def test_decoder_components(self):
        """Test all decoder components."""
        print("\\n=== Testing Decoder Components ===")
        
        decoder_types = [
            ComponentType.STANDARD_DECODER,
            ComponentType.ENHANCED_DECODER
        ]
        
        for decoder_type in decoder_types:
            with self.subTest(decoder_type=decoder_type):
                print(f"Testing {decoder_type.value}...")
                
                # Get component from registry
                component_info = component_registry.get_component(decoder_type)
                self.assertIsNotNone(component_info, f"Component {decoder_type.value} not found in registry")
                
                # Create component instance
                params = {
                    'd_model': self.d_model,
                    'n_heads': self.n_heads,
                    'd_ff': self.d_ff,
                    'd_layers': self.d_layers,
                    'dropout': 0.1,
                    'activation': 'gelu'
                }
                
                try:
                    component = component_info.factory(**params)
                    
                    # Create decoder input (target sequence)
                    dec_input = torch.randn(self.batch_size, self.label_len + self.pred_len, self.d_model)
                    
                    # Test forward pass
                    with torch.no_grad():
                        output, attns = component(
                            dec_input, self.model_input, 
                            x_mask=None, cross_mask=None, 
                            tau=None, delta=None
                        )
                    
                    # Validate output shape
                    expected_shape = dec_input.shape
                    self.assertEqual(output.shape, expected_shape, 
                                   f"{decoder_type.value} output shape mismatch")
                    
                    # Check for NaN or Inf
                    self.assertFalse(torch.isnan(output).any(), 
                                   f"{decoder_type.value} produced NaN values")
                    self.assertFalse(torch.isinf(output).any(), 
                                   f"{decoder_type.value} produced Inf values")
                    
                    print(f"✓ {decoder_type.value} passed")
                    
                except Exception as e:
                    print(f"✗ {decoder_type.value} failed: {e}")
                    raise

    def test_sampling_components(self):
        """Test all sampling components."""
        print("\\n=== Testing Sampling Components ===")
        
        sampling_types = [
            ComponentType.DETERMINISTIC,
            ComponentType.BAYESIAN
        ]
        
        for sampling_type in sampling_types:
            with self.subTest(sampling_type=sampling_type):
                print(f"Testing {sampling_type.value}...")
                
                # Get component from registry
                component_info = component_registry.get_component(sampling_type)
                self.assertIsNotNone(component_info, f"Component {sampling_type.value} not found in registry")
                
                # Create component instance
                params = {}
                if sampling_type == ComponentType.BAYESIAN:
                    params = {'n_samples': 10}
                
                try:
                    component = component_info.factory(**params)
                    
                    # Test sampling
                    input_tensor = torch.randn(self.batch_size, self.pred_len, self.d_model)
                    
                    with torch.no_grad():
                        if sampling_type == ComponentType.BAYESIAN:
                            samples, uncertainty = component(input_tensor)
                            # Validate shapes
                            self.assertEqual(samples.shape[0], params['n_samples'])
                            self.assertEqual(uncertainty.shape, input_tensor.shape)
                        else:
                            output = component(input_tensor)
                            # Validate shape
                            self.assertEqual(output.shape, input_tensor.shape)
                    
                    print(f"✓ {sampling_type.value} passed")
                    
                except Exception as e:
                    print(f"✗ {sampling_type.value} failed: {e}")
                    raise

    def test_output_head_components(self):
        """Test all output head components."""
        print("\\n=== Testing Output Head Components ===")
        
        head_types = [
            ComponentType.STANDARD_HEAD,
            ComponentType.QUANTILE
        ]
        
        for head_type in head_types:
            with self.subTest(head_type=head_type):
                print(f"Testing {head_type.value}...")
                
                # Get component from registry
                component_info = component_registry.get_component(head_type)
                self.assertIsNotNone(component_info, f"Component {head_type.value} not found in registry")
                
                # Create component instance
                if head_type == ComponentType.STANDARD_HEAD:
                    params = {'d_model': self.d_model, 'c_out': self.c_out}
                    expected_out_features = self.c_out
                elif head_type == ComponentType.QUANTILE:
                    num_quantiles = 3
                    params = {'d_model': self.d_model, 'c_out': self.c_out, 'num_quantiles': num_quantiles}
                    expected_out_features = self.c_out * num_quantiles
                
                try:
                    component = component_info.factory(**params)
                    
                    # Test forward pass
                    input_tensor = torch.randn(self.batch_size, self.pred_len, self.d_model)
                    
                    with torch.no_grad():
                        output = component(input_tensor)
                    
                    # Validate output shape
                    expected_shape = (self.batch_size, self.pred_len, expected_out_features)
                    self.assertEqual(output.shape, expected_shape, 
                                   f"{head_type.value} output shape mismatch")
                    
                    # Check for NaN or Inf
                    self.assertFalse(torch.isnan(output).any(), 
                                   f"{head_type.value} produced NaN values")
                    self.assertFalse(torch.isinf(output).any(), 
                                   f"{head_type.value} produced Inf values")
                    
                    print(f"✓ {head_type.value} passed")
                    
                except Exception as e:
                    print(f"✗ {head_type.value} failed: {e}")
                    raise

    def test_loss_function_components(self):
        """Test all loss function components."""
        print("\\n=== Testing Loss Function Components ===")
        
        loss_types = [
            ComponentType.MSE,
            ComponentType.BAYESIAN,
            ComponentType.BAYESIAN_QUANTILE
        ]
        
        for loss_type in loss_types:
            with self.subTest(loss_type=loss_type):
                print(f"Testing {loss_type.value}...")
                
                # Get component from registry
                component_info = component_registry.get_component(loss_type)
                self.assertIsNotNone(component_info, f"Component {loss_type.value} not found in registry")
                
                # Create component instance
                params = {}
                if loss_type == ComponentType.BAYESIAN:
                    params = {'kl_weight': 1e-5}
                elif loss_type == ComponentType.BAYESIAN_QUANTILE:
                    params = {'quantile_levels': [0.1, 0.5, 0.9], 'kl_weight': 1e-5}
                
                try:
                    component = component_info.factory(**params)
                    
                    # Create test predictions and targets
                    if loss_type == ComponentType.BAYESIAN_QUANTILE:
                        # Quantile predictions (batch, seq, features * quantiles)
                        pred = torch.randn(self.batch_size, self.pred_len, self.c_out * 3)
                    else:
                        pred = torch.randn(self.batch_size, self.pred_len, self.c_out)
                    
                    target = torch.randn(self.batch_size, self.pred_len, self.c_out)
                    
                    # Test loss computation
                    if loss_type == ComponentType.BAYESIAN:
                        # Mock KL divergence
                        kl_div = torch.tensor(0.1)
                        loss = component(pred, target, kl_div)
                    elif loss_type == ComponentType.BAYESIAN_QUANTILE:
                        # Mock KL divergence
                        kl_div = torch.tensor(0.1)
                        loss = component(pred, target, kl_div)
                    else:
                        loss = component(pred, target)
                    
                    # Validate loss properties
                    self.assertIsInstance(loss, torch.Tensor, f"{loss_type.value} should return tensor")
                    self.assertEqual(loss.ndim, 0, f"{loss_type.value} should return scalar")
                    self.assertGreaterEqual(loss.item(), 0, f"{loss_type.value} should be non-negative")
                    self.assertFalse(torch.isnan(loss), f"{loss_type.value} produced NaN")
                    self.assertFalse(torch.isinf(loss), f"{loss_type.value} produced Inf")
                    
                    print(f"✓ {loss_type.value} passed")
                    
                except Exception as e:
                    print(f"✗ {loss_type.value} failed: {e}")
                    raise

    def test_component_registry_completeness(self):
        """Test that all expected components are registered."""
        print("\\n=== Testing Component Registry Completeness ===")
        
        expected_components = [
            # Attention components
            ComponentType.AUTOCORRELATION_LAYER,
            ComponentType.ADAPTIVE_AUTOCORRELATION_LAYER,
            ComponentType.CROSS_RESOLUTION,
            # Decomposition components
            ComponentType.SERIES_DECOMP,
            ComponentType.STABLE_DECOMP,
            ComponentType.LEARNABLE_DECOMP,
            ComponentType.WAVELET_DECOMP,
            # Encoder components
            ComponentType.STANDARD_ENCODER,
            ComponentType.ENHANCED_ENCODER,
            ComponentType.HIERARCHICAL_ENCODER,
            # Decoder components
            ComponentType.STANDARD_DECODER,
            ComponentType.ENHANCED_DECODER,
            # Sampling components
            ComponentType.DETERMINISTIC,
            ComponentType.BAYESIAN,
            # Output head components
            ComponentType.STANDARD_HEAD,
            ComponentType.QUANTILE,
            # Loss function components
            ComponentType.MSE,
            ComponentType.BAYESIAN,
            ComponentType.BAYESIAN_QUANTILE,
        ]
        
        for component_type in expected_components:
            component_info = component_registry.get_component(component_type)
            self.assertIsNotNone(component_info, 
                               f"Component {component_type.value} not found in registry")
            print(f"✓ {component_type.value} registered")
        
        print(f"✓ All {len(expected_components)} expected components are registered")

def run_component_tests():
    """Run all component tests with timing and summary."""
    print("=" * 80)
    print("COMPREHENSIVE MODULAR FRAMEWORK COMPONENT TESTS")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModularComponents)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Duration: {duration:.2f} seconds")
    
    if result.failures:
        print("\\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Test HF models availability
    print("\\n" + "=" * 80)
    print("HF MODELS AVAILABILITY")
    print("=" * 80)
    
    try:
        available_models = list_available_models()
        hf_models = available_models.get('hf', [])
        custom_models = available_models.get('custom', [])
        
        print(f"Custom models available: {len(custom_models)}")
        for model in custom_models:
            print(f"  - {model}")
            
        print(f"\\nHF models available: {len(hf_models)}")
        for model in hf_models:
            print(f"  - {model}")
            
    except Exception as e:
        print(f"Error checking model availability: {e}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\\n🎉 ALL COMPONENT TESTS PASSED! 🎉")
    else:
        print("\\n❌ SOME TESTS FAILED")
    
    return success

if __name__ == "__main__":
    success = run_component_tests()
    sys.exit(0 if success else 1)
