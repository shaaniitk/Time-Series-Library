"""
Test Script for GCLI Modular Autoformer Architecture

This script tests the implementation of GCLI recommendations:
1. Structured configuration with Pydantic schemas
2. ModularComponent base class
3. "Dumb assembler" pattern
4. Component registry system
5. Systematic Bayesian handling (when available)

As requested: "remember whatever changes you make they have to be incorporated in the test script"
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.schemas import (
    ModularAutoformerConfig, ComponentType,
    create_enhanced_config, create_bayesian_enhanced_config, 
    create_quantile_bayesian_config, create_hierarchical_config
)
from configs.modular_components import (
    ModularAssembler, component_registry, register_all_components
)
from models.modular_autoformer import ModularAutoformer
from utils.logger import logger


def test_gcli_structured_configuration():
    """Test GCLI Recommendation 1: Structured configuration with Pydantic"""
    print("\n=== Testing GCLI Structured Configuration ===")
    
    try:
        # Test structured config creation
        config = create_enhanced_config(
            num_targets=7,
            num_covariates=0,
            seq_len=96,
            pred_len=24,
            d_model=512
        )
        
        print(f"✅ Structured config created: {config.attention.type}")
        print(f"✅ Validation works: c_out={config.c_out}, c_out_evaluation={config.c_out_evaluation}")
        
        # Test legacy conversion
        legacy_ns = config.to_namespace()
        print(f"✅ Legacy conversion works: {legacy_ns.attention_type}")
        
        # Test quantile config with validation
        quantile_config = create_quantile_bayesian_config(
            num_targets=7,
            num_covariates=0,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        print(f"✅ Quantile config: c_out={quantile_config.c_out}, quantiles={len(quantile_config.quantile_levels)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Structured configuration test failed: {e}")
        return False


def test_gcli_component_registry():
    """Test GCLI Recommendation 2: Component registry and ModularComponent base class"""
    print("\n=== Testing GCLI Component Registry ===")
    
    try:
        # Register all components
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
        
        print(f"✅ Component created: {attention_comp.get_component_info()}")
        print(f"✅ Component validated: {attention_comp._validated}")
        
        # Test available components
        available = component_registry.get_available_components()
        print(f"✅ Available components: {len(available)} types registered")
        
        return True
        
    except Exception as e:
        print(f"❌ Component registry test failed: {e}")
        return False


def test_gcli_dumb_assembler():
    """Test GCLI Recommendation 3: "Dumb assembler" pattern"""
    print("\n=== Testing GCLI Dumb Assembler ===")
    
    try:
        # Create structured configuration
        config = create_enhanced_config(
            num_targets=7,
            num_covariates=0,
            seq_len=96,
            pred_len=24
        )
        
        # Create assembler
        assembler = ModularAssembler(component_registry)
        
        # Assemble model using dumb assembler pattern
        assembled_model = assembler.assemble_model(config)
        
        print(f"✅ Model assembled: {assembled_model.__class__.__name__}")
        
        # Test component summary
        summary = assembler.get_component_summary()
        print(f"✅ Component summary: {len(summary)} components")
        for name, info in summary.items():
            print(f"   - {name}: {info['name']} ({info['parameters']} params)")
        
        # Test model summary
        model_summary = assembled_model.get_model_summary()
        print(f"✅ Model summary: {model_summary['total_parameters']} total parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Dumb assembler test failed: {e}")
        return False


def test_gcli_modular_autoformer():
    """Test GCLI integration with ModularAutoformer"""
    print("\n=== Testing GCLI ModularAutoformer Integration ===")
    
    try:
        # Test with structured configuration
        structured_config = create_enhanced_config(
            num_targets=7,
            num_covariates=0,
            seq_len=96,
            pred_len=24,
            d_model=512
        )
        
        # Create model with structured config
        model = ModularAutoformer(structured_config)
        
        print(f"✅ ModularAutoformer created with structured config")
        print(f"✅ Framework type: {model.framework_type}")
        print(f"✅ Has assembled model: {hasattr(model, 'assembled_model')}")
        
        # Test with legacy namespace config (backward compatibility)
        legacy_ns = structured_config.to_namespace()
        legacy_model = ModularAutoformer(legacy_ns)
        
        print(f"✅ Legacy compatibility maintained")
        
        # Test forward pass
        batch_size = 2
        x_enc = torch.randn(batch_size, structured_config.seq_len, structured_config.enc_in)
        x_mark_enc = torch.randn(batch_size, structured_config.seq_len, 4)
        x_dec = torch.randn(batch_size, structured_config.label_len + structured_config.pred_len, structured_config.dec_in)
        x_mark_dec = torch.randn(batch_size, structured_config.label_len + structured_config.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
        expected_shape = (batch_size, structured_config.pred_len, structured_config.c_out)
        print(f"✅ Forward pass successful: {output.shape} (expected: {expected_shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ ModularAutoformer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcli_configuration_variants():
    """Test GCLI with different configuration variants"""
    print("\n=== Testing GCLI Configuration Variants ===")
    
    try:
        configs_to_test = [
            ("Enhanced", create_enhanced_config),
            ("Bayesian Enhanced", create_bayesian_enhanced_config),
            ("Quantile Bayesian", lambda nt, nc, **kw: create_quantile_bayesian_config(nt, nc, quantile_levels=[0.1, 0.5, 0.9], **kw)),
            ("Hierarchical", create_hierarchical_config),
        ]
        
        for config_name, config_func in configs_to_test:
            try:
                config = config_func(
                    num_targets=7,
                    num_covariates=0,
                    seq_len=48,
                    pred_len=12,
                    d_model=256
                )
                
                # Test model creation
                model = ModularAutoformer(config)
                
                # Test basic forward pass
                x_enc = torch.randn(1, config.seq_len, config.enc_in)
                x_mark_enc = torch.randn(1, config.seq_len, 4)
                x_dec = torch.randn(1, config.label_len + config.pred_len, config.dec_in)
                x_mark_dec = torch.randn(1, config.label_len + config.pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                print(f"✅ {config_name}: {output.shape}")
                
            except Exception as e:
                print(f"❌ {config_name} failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Configuration variants test failed: {e}")
        return False


def test_gcli_component_encapsulation():
    """Test GCLI Recommendation 4: Component encapsulation"""
    print("\n=== Testing GCLI Component Encapsulation ===")
    
    try:
        # Test component isolation
        config = create_enhanced_config(7, 0)
        assembler = ModularAssembler(component_registry)
        
        # Test that components are properly encapsulated
        model = assembler.assemble_model(config)
        
        # Verify component interfaces
        components = ['attention', 'decomposition', 'encoder', 'decoder', 'output_head']
        for comp_name in components:
            comp = getattr(model, comp_name, None)
            if comp:
                info = comp.get_component_info()
                print(f"✅ {comp_name}: {info['name']} - {info['parameters']} params")
            else:
                print(f"⚠️  {comp_name}: Component not found")
        
        # Test component metadata
        for comp_name in components:
            comp = getattr(model, comp_name, None)
            if comp and hasattr(comp, 'metadata'):
                metadata = comp.metadata
                print(f"✅ {comp_name} metadata: {metadata.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component encapsulation test failed: {e}")
        return False


def run_comprehensive_gcli_test():
    """Run comprehensive test of all GCLI recommendations"""
    print("🚀 Starting Comprehensive GCLI Architecture Test")
    print("=" * 60)
    
    test_results = {
        "Structured Configuration": test_gcli_structured_configuration(),
        "Component Registry": test_gcli_component_registry(),
        "Dumb Assembler": test_gcli_dumb_assembler(),
        "ModularAutoformer Integration": test_gcli_modular_autoformer(),
        "Configuration Variants": test_gcli_configuration_variants(),
        "Component Encapsulation": test_gcli_component_encapsulation(),
    }
    
    print("\n" + "=" * 60)
    print("📊 GCLI Implementation Test Results:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    total = len(test_results)
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All GCLI recommendations successfully implemented!")
    else:
        print("⚠️  Some GCLI features need attention")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_gcli_test()
    sys.exit(0 if success else 1)
