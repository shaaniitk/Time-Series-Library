"""
Test suite for the Unified Autoformer Architecture with GCLI Integration

This test validates:
1. The integration between HF-based and custom modular autoformers
2. GCLI structured configuration system
3. ModularComponent base class and "dumb assembler" pattern
4. Component registry and systematic Bayesian handling

As requested: "remember whatever changes you make they have to be incorporated in the test script"
"""

import unittest
import torch
from argparse import Namespace
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Original unified factory imports
from models.unified_autoformer_factory import (
    UnifiedAutoformerFactory,
    UnifiedModelInterface,
    create_autoformer,
    compare_implementations,
    list_available_models
)

# Legacy config imports
from configs.autoformer import (
    standard_config,
    enhanced_config,
    bayesian_enhanced_config,
    hierarchical_config,
    quantile_bayesian_config,
)

# GCLI Architecture imports
from configs.schemas import (
    ModularAutoformerConfig, ComponentType,
    create_enhanced_config, create_bayesian_enhanced_config, 
    create_quantile_bayesian_config, create_hierarchical_config
)
from configs.modular_components import (
    ModularAssembler, component_registry, register_all_components
)
from models.modular_autoformer import ModularAutoformer


class TestUnifiedAutoformerArchitecture(unittest.TestCase):
    """Test suite for unified autoformer architecture"""
    
    def setUp(self):
        """Set up common test parameters"""
        self.num_targets = 7
        self.num_covariates = 0
        self.seq_len = 96
        self.pred_len = 24
        self.label_len = 48
        self.batch_size = 2
        
        # Test data
        self.x_enc = torch.randn(self.batch_size, self.seq_len, self.num_targets)
        self.x_mark_enc = torch.randn(self.batch_size, self.seq_len, 4)
        self.x_dec = torch.randn(self.batch_size, self.label_len + self.pred_len, self.num_targets)
        self.x_mark_dec = torch.randn(self.batch_size, self.label_len + self.pred_len, 4)
        
        self.base_config = {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'label_len': self.label_len,
            'enc_in': self.num_targets,
            'dec_in': self.num_targets,
            'c_out': self.num_targets,
            'd_model': 512,
            'task_name': 'long_term_forecast'
        }
    
    def test_factory_model_creation(self):
        """Test that the factory can create models of different types"""
        
        # Test custom model types
        custom_types = ['enhanced', 'bayesian_enhanced', 'hierarchical']
        
        for model_type in custom_types:
            with self.subTest(model_type=model_type):
                print(f"Testing factory creation of custom {model_type}")
                
                # Get appropriate config
                if model_type == 'enhanced':
                    config = enhanced_config.get_enhanced_autoformer_config(
                        num_targets=self.num_targets, 
                        num_covariates=self.num_covariates,
                        **self.base_config
                    )
                elif model_type == 'bayesian_enhanced':
                    config = bayesian_enhanced_config.get_bayesian_enhanced_autoformer_config(
                        num_targets=self.num_targets, 
                        num_covariates=self.num_covariates,
                        **self.base_config
                    )
                elif model_type == 'hierarchical':
                    config = hierarchical_config.get_hierarchical_autoformer_config(
                        num_targets=self.num_targets, 
                        num_covariates=self.num_covariates,
                        **self.base_config
                    )
                
                # Test factory creation
                model = UnifiedAutoformerFactory.create_model(
                    model_type=model_type,
                    config=config,
                    framework_preference='custom'
                )
                
                self.assertIsNotNone(model)
                self.assertEqual(model.framework_type, 'custom')
                print(f"✅ Custom {model_type} created successfully")
    
    def test_unified_interface_consistency(self):
        """Test that the unified interface provides consistent results"""
        
        print("Testing unified interface consistency")
        
        # Create model through factory
        config = enhanced_config.get_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            **self.base_config
        )
        model = create_autoformer('enhanced', config, framework='custom')
        
        # Test basic prediction
        with torch.no_grad():
            prediction = model.predict(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        self.assertEqual(prediction.shape, (self.batch_size, self.pred_len, self.num_targets))
        print(f"✅ Prediction shape correct: {prediction.shape}")
        
        # Test model info
        info = model.get_model_info()
        self.assertIn('framework_type', info)
        self.assertIn('model_type', info)
        self.assertIn('model_class', info)
        print(f"✅ Model info: {info['framework_type']}/{info['model_type']}")
    
    def test_uncertainty_quantification_interface(self):
        """Test unified uncertainty quantification interface"""
        
        print("Testing uncertainty quantification interface")
        
        # Test with Bayesian model that supports uncertainty
        config = bayesian_enhanced_config.get_bayesian_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            **self.base_config
        )
        model = create_autoformer('bayesian_enhanced', config, framework='custom')
        
        # Test uncertainty prediction
        with torch.no_grad():
            result = model.predict_with_uncertainty(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        self.assertIn('prediction', result)
        self.assertIsNotNone(result['prediction'])
        self.assertEqual(result['prediction'].shape, (self.batch_size, self.pred_len, self.num_targets))
        print(f"✅ Uncertainty prediction works: {result['prediction'].shape}")
    
    def test_quantile_models_through_factory(self):
        """Test quantile models through the unified factory"""
        
        print("Testing quantile models through factory")
        
        config = quantile_bayesian_config.get_quantile_bayesian_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            quantile_levels=[0.1, 0.5, 0.9],
            **self.base_config
        )
        
        model = create_autoformer('quantile_bayesian', config, framework='custom')
        
        with torch.no_grad():
            prediction = model.predict(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        # Should output 7 targets * 3 quantiles = 21 dimensions
        expected_dims = self.num_targets * len(config.quantile_levels)
        self.assertEqual(prediction.shape, (self.batch_size, self.pred_len, expected_dims))
        print(f"✅ Quantile model output correct: {prediction.shape}")
    
    def test_framework_switching(self):
        """Test switching between custom and HF frameworks"""
        
        print("Testing framework switching")
        
        config = enhanced_config.get_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            **self.base_config
        )
        
        # Create custom model
        custom_model = create_autoformer('enhanced', config, framework='custom')
        self.assertEqual(custom_model.framework_type, 'custom')
        print("✅ Custom framework selection works")
        
        # Test HF framework (may not be available, should handle gracefully)
        try:
            hf_model = create_autoformer('enhanced', config, framework='hf')
            print("✅ HF framework available and working")
        except Exception as e:
            print(f"⚠️ HF framework not available: {e}")
    
    def test_model_comparison_interface(self):
        """Test the model comparison functionality"""
        
        print("Testing model comparison interface")
        
        config = enhanced_config.get_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            **self.base_config
        )
        
        # Create comparison models
        comparison = compare_implementations('enhanced', config)
        
        # Should at least have custom implementation
        self.assertIn('custom', comparison)
        print(f"✅ Comparison models available: {list(comparison.keys())}")
        
        # Test that both implementations give same shape output
        for framework, model in comparison.items():
            with torch.no_grad():
                prediction = model.predict(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            self.assertEqual(prediction.shape, (self.batch_size, self.pred_len, self.num_targets))
            print(f"✅ {framework} model output shape correct")
    
    def test_available_models_listing(self):
        """Test listing of available models"""
        
        print("Testing available models listing")
        
        available = list_available_models()
        
        self.assertIn('custom', available)
        self.assertIsInstance(available['custom'], list)
        self.assertGreater(len(available['custom']), 0)
        
        print(f"✅ Available models: {available}")
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        
        print("Testing edge cases and error handling")
        
        # Test invalid model type
        with self.assertRaises(Exception):
            UnifiedAutoformerFactory.create_model(
                model_type='nonexistent_model',
                config=self.base_config,
                framework_preference='custom'
            )
        print("✅ Invalid model type handled correctly")
        
        # Test dict to Namespace conversion
        model = create_autoformer('enhanced', self.base_config, framework='custom')
        self.assertIsNotNone(model)
        print("✅ Dict config conversion works")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        
        print("Testing end-to-end workflow")
        
        # 1. List available models
        available = list_available_models()
        print(f"Step 1 - Available models: {list(available['custom'])}")
        
        # 2. Create model using factory
        config = enhanced_config.get_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            **self.base_config
        )
        model = create_autoformer('enhanced', config, framework='auto')
        print(f"Step 2 - Model created: {model.get_model_info()['model_class']}")
        
        # 3. Make prediction
        with torch.no_grad():
            prediction = model.predict(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        print(f"Step 3 - Prediction shape: {prediction.shape}")
        
        # 4. Test uncertainty if supported
        uncertainty_result = model.predict_with_uncertainty(
            self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec
        )
        print(f"Step 4 - Uncertainty prediction: {'✅' if uncertainty_result['prediction'] is not None else '❌'}")
        
        # 5. Get model information
        info = model.get_model_info()
        print(f"Step 5 - Model info: {info['framework_type']}/{info['model_type']}")
        
        print("✅ End-to-end workflow completed successfully")


class TestUnifiedFactoryPerformance(unittest.TestCase):
    """Performance and stress tests for the unified factory"""
    
    def setUp(self):
        self.base_config = {
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 256  # Smaller for faster tests
        }
    
    def test_multiple_model_creation(self):
        """Test creating multiple models efficiently"""
        
        print("Testing multiple model creation performance")
        
        model_types = ['enhanced', 'bayesian_enhanced', 'hierarchical']
        models = []
        
        import time
        start_time = time.time()
        
        for model_type in model_types:
            if model_type == 'enhanced':
                config = enhanced_config.get_enhanced_autoformer_config(
                    num_targets=7, num_covariates=0, **self.base_config
                )
            elif model_type == 'bayesian_enhanced':
                config = bayesian_enhanced_config.get_bayesian_enhanced_autoformer_config(
                    num_targets=7, num_covariates=0, **self.base_config
                )
            elif model_type == 'hierarchical':
                config = hierarchical_config.get_hierarchical_autoformer_config(
                    num_targets=7, num_covariates=0, **self.base_config
                )
            
            model = create_autoformer(model_type, config, framework='custom')
            models.append(model)
        
        creation_time = time.time() - start_time
        print(f"Created {len(models)} models in {creation_time:.2f} seconds")
        
        # Verify all models work
        x_enc = torch.randn(1, 96, 7)
        x_mark_enc = torch.randn(1, 96, 4)
        x_dec = torch.randn(1, 72, 7)
        x_mark_dec = torch.randn(1, 72, 4)
        
        for i, model in enumerate(models):
            with torch.no_grad():
                prediction = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
            self.assertEqual(prediction.shape[1:], (24, 7))
            print(f"✅ Model {i+1} prediction successful")
    
    # =====================================================
    # GCLI ARCHITECTURE TESTS - New Implementation Tests
    # =====================================================
    
    def test_gcli_structured_configuration(self):
        """Test GCLI structured configuration with Pydantic schemas"""
        
        print("Testing GCLI structured configuration")
        
        # Test structured config creation
        structured_config = create_enhanced_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=512
        )
        
        # Verify config structure
        self.assertIsInstance(structured_config, ModularAutoformerConfig)
        self.assertEqual(structured_config.attention.type, ComponentType.ADAPTIVE_AUTOCORRELATION)
        self.assertEqual(structured_config.c_out, self.num_targets)
        print(f"✅ Structured config created: {structured_config.attention.type}")
        
        # Test conversion to legacy format
        legacy_ns = structured_config.to_namespace()
        self.assertIsInstance(legacy_ns, Namespace)
        self.assertEqual(legacy_ns.attention_type, 'adaptive_autocorrelation_layer')
        print("✅ Legacy conversion successful")
    
    def test_gcli_component_registry(self):
        """Test GCLI component registry and ModularComponent base class"""
        
        print("Testing GCLI component registry")
        
        # Register components
        register_all_components()
        
        # Test component creation
        from configs.schemas import AttentionConfig
        attention_config = AttentionConfig(
            type=ComponentType.AUTOCORRELATION,
            d_model=512,
            n_heads=8
        )
        
        attention_comp = component_registry.create_component(
            ComponentType.AUTOCORRELATION,
            attention_config
        )
        
        # Verify component interface
        self.assertTrue(hasattr(attention_comp, 'get_component_info'))
        self.assertTrue(attention_comp._validated)
        
        component_info = attention_comp.get_component_info()
        self.assertIn('name', component_info)
        self.assertIn('parameters', component_info)
        print(f"✅ Component created: {component_info['name']}")
    
    def test_gcli_dumb_assembler(self):
        """Test GCLI dumb assembler pattern"""
        
        print("Testing GCLI dumb assembler")
        
        # Create structured configuration
        config = create_enhanced_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            seq_len=self.seq_len,
            pred_len=self.pred_len
        )
        
        # Create assembler and assemble model
        assembler = ModularAssembler(component_registry)
        assembled_model = assembler.assemble_model(config)
        
        # Verify assembly
        self.assertIsNotNone(assembled_model)
        self.assertTrue(hasattr(assembled_model, 'components'))
        self.assertTrue(hasattr(assembled_model, 'attention'))
        
        # Test component summary
        summary = assembler.get_component_summary()
        self.assertIn('attention', summary)
        self.assertIn('encoder', summary)
        print(f"✅ Model assembled with {len(summary)} components")
    
    def test_gcli_modular_autoformer_integration(self):
        """Test GCLI integration with ModularAutoformer"""
        
        print("Testing GCLI ModularAutoformer integration")
        
        # Test with structured configuration
        structured_config = create_enhanced_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            seq_len=self.seq_len,
            pred_len=self.pred_len
        )
        
        # Create model with structured config
        model = ModularAutoformer(structured_config)
        
        # Verify GCLI features
        self.assertEqual(model.framework_type, 'custom')
        self.assertTrue(hasattr(model, 'assembled_model'))
        self.assertTrue(hasattr(model, 'structured_config'))
        
        # Test forward pass
        with torch.no_grad():
            output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        expected_shape = (self.batch_size, self.pred_len, self.num_targets)
        self.assertEqual(output.shape, expected_shape)
        print(f"✅ GCLI forward pass: {output.shape}")
        
        # Test legacy compatibility
        legacy_ns = structured_config.to_namespace()
        legacy_model = ModularAutoformer(legacy_ns)
        self.assertTrue(hasattr(legacy_model, 'structured_config'))
        print("✅ Legacy compatibility maintained")
    
    def test_gcli_configuration_variants(self):
        """Test GCLI with different configuration variants"""
        
        print("Testing GCLI configuration variants")
        
        configs_to_test = [
            ("Enhanced", create_enhanced_config),
            ("Bayesian Enhanced", create_bayesian_enhanced_config),
            ("Quantile Bayesian", lambda nt, nc, **kw: create_quantile_bayesian_config(nt, nc, quantile_levels=[0.1, 0.5, 0.9], **kw)),
            ("Hierarchical", create_hierarchical_config),
        ]
        
        for config_name, config_func in configs_to_test:
            with self.subTest(config=config_name):
                config = config_func(
                    num_targets=self.num_targets,
                    num_covariates=self.num_covariates,
                    seq_len=48,  # Smaller for faster testing
                    pred_len=12,
                    d_model=256
                )
                
                # Test model creation
                model = ModularAutoformer(config)
                self.assertIsNotNone(model)
                
                # Test basic forward pass
                x_enc = torch.randn(1, config.seq_len, config.enc_in)
                x_mark_enc = torch.randn(1, config.seq_len, 4)
                x_dec = torch.randn(1, config.label_len + config.pred_len, config.dec_in)
                x_mark_dec = torch.randn(1, config.label_len + config.pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # Verify output shape
                if "Quantile" in config_name:
                    expected_c_out = config.c_out  # Should be targets * quantiles
                else:
                    expected_c_out = config.c_out_evaluation
                
                expected_shape = (1, config.pred_len, expected_c_out)
                self.assertEqual(output.shape, expected_shape)
                print(f"✅ {config_name}: {output.shape}")
    
    def test_gcli_unified_factory_integration(self):
        """Test integration between GCLI and unified factory"""
        
        print("Testing GCLI unified factory integration")
        
        # Create both GCLI and legacy models for comparison
        structured_config = create_enhanced_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            **self.base_config
        )
        
        legacy_config = enhanced_config.get_enhanced_autoformer_config(
            num_targets=self.num_targets,
            num_covariates=self.num_covariates,
            **self.base_config
        )
        
        # Create models
        gcli_model = ModularAutoformer(structured_config)
        legacy_factory_model = create_autoformer('enhanced', legacy_config, framework='custom')
        
        # Test both models produce similar outputs (shapes should match)
        with torch.no_grad():
            gcli_output = gcli_model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            legacy_output = legacy_factory_model.predict(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        self.assertEqual(gcli_output.shape, legacy_output.shape)
        print(f"✅ GCLI and legacy outputs compatible: {gcli_output.shape}")


if __name__ == '__main__':
    print("=" * 60)
    print("UNIFIED AUTOFORMER ARCHITECTURE TEST SUITE WITH GCLI")
    print("=" * 60)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)
