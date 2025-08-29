"""
Unified Framework Integration Test

This test demonstrates the unified framework supporting:
1. Backward compatibility with existing ModularAutoformer (custom framework)
2. HF integration with HFAutoformerSuite
3. New ChronosX integration
4. Seamless switching between frameworks
"""

import unittest
import torch
import sys
import os
from argparse import Namespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import unified framework
from models.base_forecaster import create_unified_model, BaseTimeSeriesForecaster
from models.modular_autoformer import ModularAutoformer
from models.chronosx_autoformer import ChronosXAutoformer
from models.HFAutoformerSuite import HFEnhancedAutoformer

# Import configurations
from configs.autoformer.bayesian_enhanced_config import get_bayesian_enhanced_autoformer_config
from configs.autoformer.chronosx_config import (
    get_chronosx_autoformer_config,
    get_unified_framework_config,
    create_chronosx_config
)


class TestUnifiedFrameworkIntegration(unittest.TestCase):
    """Test unified framework with backward compatibility."""
    
    def setUp(self):
        """Set up test parameters."""
        self.num_targets = 7
        self.num_covariates = 3
        self.seq_len = 96
        self.pred_len = 24
        self.label_len = 48
        self.batch_size = 2
        
        # Create test data
        self.test_data = self._create_test_data()
    
    def _create_test_data(self):
        """Create test data tensors."""
        return {
            'x_enc': torch.randn(self.batch_size, self.seq_len, self.num_targets + self.num_covariates),
            'x_mark_enc': torch.randn(self.batch_size, self.seq_len, 4),
            'x_dec': torch.randn(self.batch_size, self.label_len + self.pred_len, self.num_targets + self.num_covariates),
            'x_mark_dec': torch.randn(self.batch_size, self.label_len + self.pred_len, 4)
        }
    
    def test_backward_compatibility_custom_framework(self):
        """Test that existing custom framework (ModularAutoformer) still works."""
        print("\n=== Testing Backward Compatibility: Custom Framework ===")
        
        # Use existing bayesian config (should work unchanged)
        config = get_bayesian_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len
        )
        
        # Create model using traditional approach
        model = ModularAutoformer(config)
        model.eval()
        
        print(f"✅ Traditional ModularAutoformer created successfully")
        print(f"   Framework type: {model.get_framework_type()}")
        print(f"   Model type: {model.get_model_type()}")
        print(f"   Supports uncertainty: {model.supports_uncertainty()}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(**self.test_data)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: {(self.batch_size, self.pred_len, config.c_out)}")
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.pred_len, config.c_out))
        
        # Test unified framework methods
        model_info = model.get_model_info()
        self.assertEqual(model_info['framework_type'], 'custom')
        self.assertEqual(model_info['model_type'], 'modular_autoformer')
        
        print("✅ Custom framework backward compatibility verified")
    
    def test_chronosx_framework(self):
        """Test new ChronosX framework integration."""
        print("\n=== Testing ChronosX Framework ===")
        
        # Create ChronosX configuration
        config = get_chronosx_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            chronosx_model_size='tiny',  # Use tiny for testing
            uncertainty_enabled=True,
            num_samples=10
        )
        
        # Create model
        model = ChronosXAutoformer(config)
        model.eval()
        
        print(f"✅ ChronosXAutoformer created successfully")
        print(f"   Framework type: {model.get_framework_type()}")
        print(f"   Model type: {model.get_model_type()}")
        print(f"   Supports uncertainty: {model.supports_uncertainty()}")
        print(f"   ChronosX info: {model.get_chronos_info()}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(**self.test_data)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: {(self.batch_size, self.pred_len, config.c_out)}")
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.pred_len, config.c_out))
        
        # Test uncertainty results
        uncertainty_results = model.get_uncertainty_results()
        if uncertainty_results:
            print(f"   Uncertainty available: {list(uncertainty_results.keys())}")
            print(f"   Prediction shape: {uncertainty_results['prediction'].shape}")
            if 'std' in uncertainty_results:
                print(f"   Std shape: {uncertainty_results['std'].shape}")
        
        # Test unified framework methods
        model_info = model.get_model_info()
        self.assertEqual(model_info['framework_type'], 'chronosx')
        self.assertEqual(model_info['model_type'], 'chronosx_autoformer')
        
        print("✅ ChronosX framework integration verified")
    
    def test_hf_framework(self):
        """Test HF framework integration."""
        print("\n=== Testing HF Framework ===")
        
        # Create HF configuration
        config = Namespace(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            enc_in=self.num_targets + self.num_covariates,
            dec_in=self.num_targets + self.num_covariates,
            c_out=self.num_targets,
            d_model=64,  # Smaller for testing
            task_name='long_term_forecast'
        )
        
        # Create model
        model = HFEnhancedAutoformer(config)
        model.eval()
        
        print(f"✅ HFEnhancedAutoformer created successfully")
        print(f"   Framework type: {model.get_framework_type()}")
        print(f"   Model type: {model.get_model_type()}")
        print(f"   Supports uncertainty: {model.supports_uncertainty()}")
        print(f"   HF model info: {model.get_hf_model_info()}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(**self.test_data)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: {(self.batch_size, self.pred_len, config.c_out)}")
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.pred_len, config.c_out))
        
        # Test unified framework methods
        model_info = model.get_model_info()
        self.assertEqual(model_info['framework_type'], 'hf')
        self.assertEqual(model_info['model_type'], 'hf_enhanced_autoformer')
        
        print("✅ HF framework integration verified")
    
    def test_unified_factory_function(self):
        """Test the unified factory function."""
        print("\n=== Testing Unified Factory Function ===")
        
        # Test auto-detection for custom framework
        custom_config = get_bayesian_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len
        )
        
        model_custom = create_unified_model(custom_config, framework_type='custom')
        self.assertIsInstance(model_custom, ModularAutoformer)
        self.assertEqual(model_custom.get_framework_type(), 'custom')
        print("✅ Unified factory creates custom models correctly")
        
        # Test ChronosX creation
        chronosx_config = create_chronosx_config(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.num_targets + self.num_covariates,
            c_out=self.num_targets
        )
        
        model_chronosx = create_unified_model(chronosx_config, framework_type='chronosx')
        self.assertIsInstance(model_chronosx, ChronosXAutoformer)
        self.assertEqual(model_chronosx.get_framework_type(), 'chronosx')
        print("✅ Unified factory creates ChronosX models correctly")
        
        print("✅ Unified factory function verified")
    
    def test_framework_compatibility(self):
        """Test that all frameworks produce compatible outputs."""
        print("\n=== Testing Framework Compatibility ===")
        
        # Create models from different frameworks
        models = {}
        
        # Custom framework
        custom_config = get_bayesian_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len
        )
        models['custom'] = ModularAutoformer(custom_config)
        
        # ChronosX framework
        chronosx_config = create_chronosx_config(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.num_targets + self.num_covariates,
            c_out=self.num_targets
        )
        models['chronosx'] = ChronosXAutoformer(chronosx_config)
        
        # HF framework
        hf_config = Namespace(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            enc_in=self.num_targets + self.num_covariates,
            dec_in=self.num_targets + self.num_covariates,
            c_out=self.num_targets,
            d_model=64,
            task_name='long_term_forecast'
        )
        models['hf'] = HFEnhancedAutoformer(hf_config)
        
        # Test all models
        outputs = {}
        for framework, model in models.items():
            model.eval()
            with torch.no_grad():
                output = model(**self.test_data)
            
            outputs[framework] = output
            print(f"   {framework:10}: {output.shape} - {model.get_framework_type()}")
            
            # Verify they all implement BaseTimeSeriesForecaster
            self.assertIsInstance(model, BaseTimeSeriesForecaster)
            
            # Verify common interface methods
            self.assertTrue(hasattr(model, 'get_model_info'))
            self.assertTrue(hasattr(model, 'supports_uncertainty'))
            self.assertTrue(hasattr(model, 'supports_quantiles'))
            
            # All should produce same shape outputs
            expected_shape = (self.batch_size, self.pred_len, self.num_targets)
            self.assertEqual(output.shape, expected_shape)
        
        print("✅ All frameworks produce compatible outputs")
        print("✅ All frameworks implement unified interface")
    
    def test_seamless_switching(self):
        """Test seamless switching between frameworks."""
        print("\n=== Testing Seamless Framework Switching ===")
        
        # Define a common evaluation function
        def evaluate_model(model, data):
            model.eval()
            with torch.no_grad():
                output = model(**data)
            
            info = model.get_model_info()
            return {
                'output_shape': output.shape,
                'framework_type': info['framework_type'],
                'model_type': info['model_type'],
                'supports_uncertainty': info['supports_uncertainty'],
                'parameter_count': info['parameter_count']
            }
        
        # Test different framework configs
        framework_configs = [
            ('custom', get_bayesian_enhanced_autoformer_config(
                num_targets=self.num_targets,
                num_covariates=self.num_covariates,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                label_len=self.label_len
            )),
            ('chronosx', create_chronosx_config(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                enc_in=self.num_targets + self.num_covariates,
                c_out=self.num_targets
            ))
        ]
        
        results = {}
        for framework_type, config in framework_configs:
            model = create_unified_model(config, framework_type=framework_type)
            results[framework_type] = evaluate_model(model, self.test_data)
            
            print(f"   {framework_type:10}: {results[framework_type]['framework_type']} "
                  f"- {results[framework_type]['output_shape']} "
                  f"- Uncertainty: {results[framework_type]['supports_uncertainty']}")
        
        # Verify they all work with the same evaluation function
        for framework_type, result in results.items():
            expected_shape = (self.batch_size, self.pred_len, self.num_targets)
            self.assertEqual(result['output_shape'], expected_shape)
            self.assertEqual(result['framework_type'], framework_type)
        
        print("✅ Seamless framework switching verified")


if __name__ == '__main__':
    print("🚀 Starting Unified Framework Integration Tests")
    print("=" * 60)
    
    # Run the tests
    unittest.main(verbosity=2)
