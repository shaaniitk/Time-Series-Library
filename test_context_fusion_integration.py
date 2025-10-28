#!/usr/bin/env python3
"""
Test script to verify that the multi-scale context fusion is properly integrated
into the Celestial Enhanced PGAT modular model.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class TestConfig:
    # Core parameters
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    enc_in: int = 118
    dec_in: int = 4
    c_out: int = 4
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 3
    d_layers: int = 2
    dropout: float = 0.1
    embed: str = 'timeF'
    freq: str = 'h'
    
    # Celestial system
    use_celestial_graph: bool = True
    celestial_fusion_layers: int = 3
    num_celestial_bodies: int = 13
    
    # Enhanced features
    use_mixture_decoder: bool = False
    use_stochastic_learner: bool = False
    use_hierarchical_mapping: bool = False
    use_efficient_covariate_interaction: bool = False
    
    # Petri Net
    use_petri_net_combiner: bool = True
    num_message_passing_steps: int = 2
    edge_feature_dim: int = 6
    use_temporal_attention: bool = True
    use_spatial_attention: bool = True
    bypass_spatiotemporal_with_petri: bool = True
    
    # Other features
    enable_adaptive_topk: bool = False
    use_stochastic_control: bool = False
    enable_mdn_decoder: bool = False
    use_target_autocorrelation: bool = True
    use_calendar_effects: bool = True
    use_celestial_target_attention: bool = True
    celestial_target_use_gated_fusion: bool = True
    celestial_target_diagnostics: bool = True
    use_c2t_edge_bias: bool = False
    aggregate_waves_to_celestial: bool = True
    num_input_waves: int = 118
    target_wave_indices: List[int] = None
    use_dynamic_spatiotemporal_encoder: bool = True
    
    # Logging
    verbose_logging: bool = True
    enable_memory_debug: bool = False
    collect_diagnostics: bool = True
    enable_fusion_diagnostics: bool = True
    
    # Multi-Scale Context Fusion - KEY PARAMETERS
    use_multi_scale_context: bool = True
    context_fusion_mode: str = 'multi_scale'
    short_term_kernel_size: int = 5
    medium_term_kernel_size: int = 15
    long_term_kernel_size: int = 0
    context_fusion_dropout: float = 0.1
    enable_context_diagnostics: bool = True
    
    def __post_init__(self):
        if self.target_wave_indices is None:
            self.target_wave_indices = [0, 1, 2, 3]

def test_context_fusion_integration():
    """Test that context fusion is properly integrated and working."""
    print("🧪 Testing Multi-Scale Context Fusion Integration")
    print("=" * 60)
    
    try:
        # Import the modular model
        from models.Celestial_Enhanced_PGAT_Modular import Model
        from models.celestial_modules.context_fusion import ContextFusionFactory
        
        print("✅ Successfully imported modular model and context fusion")
        
        # Test different fusion modes
        fusion_modes = ['simple', 'gated', 'attention', 'multi_scale']
        
        for mode in fusion_modes:
            print(f"\n🔧 Testing {mode.upper()} fusion mode...")
            
            # Create configuration
            config = TestConfig()
            config.context_fusion_mode = mode
            config.enable_context_diagnostics = True
            
            # Create model
            model = Model(config)
            
            # Check that context fusion module exists and is correct type
            if config.use_multi_scale_context:
                assert model.context_fusion is not None, f"Context fusion module should exist for mode {mode}"
                print(f"   ✅ Context fusion module instantiated: {type(model.context_fusion).__name__}")
                
                # Check fusion mode
                actual_mode = model.get_context_fusion_mode()
                assert actual_mode == mode, f"Expected mode {mode}, got {actual_mode}"
                print(f"   ✅ Fusion mode correctly set: {actual_mode}")
            else:
                assert model.context_fusion is None, "Context fusion should be None when disabled"
            
            # Test forward pass
            batch_size = 2
            x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
            x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
            x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
            x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Check output format
            if isinstance(output, tuple):
                predictions, metadata = output[0], output[-1]
            else:
                predictions, metadata = output, {}
            
            print(f"   ✅ Forward pass successful: {predictions.shape}")
            
            # Check for context fusion diagnostics in metadata
            if metadata and 'context_fusion_diagnostics' in metadata:
                context_diag = metadata['context_fusion_diagnostics']
                print(f"   ✅ Context diagnostics available: {len(context_diag)} metrics")
                
                # Check for expected diagnostic keys
                expected_keys = ['input_norm', 'output_norm', 'norm_ratio', 'fusion_mode']
                for key in expected_keys:
                    if key in context_diag:
                        print(f"      • {key}: {context_diag[key]}")
            else:
                print("   ℹ️  No context diagnostics in metadata (may be disabled)")
            
            # Test diagnostic methods
            try:
                model.print_context_fusion_diagnostics()
                print("   ✅ Diagnostic printing works")
            except Exception as e:
                print(f"   ⚠️  Diagnostic printing failed: {e}")
        
        print(f"\n🎉 All fusion modes tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_fusion_disabled():
    """Test that the model works correctly when context fusion is disabled."""
    print("\n🔒 Testing Context Fusion Disabled")
    print("-" * 40)
    
    try:
        from models.Celestial_Enhanced_PGAT_Modular import Model
        
        # Create configuration with context fusion disabled
        config = TestConfig()
        config.use_multi_scale_context = False
        
        model = Model(config)
        
        # Check that context fusion is None
        assert model.context_fusion is None, "Context fusion should be None when disabled"
        print("✅ Context fusion properly disabled")
        
        # Check fusion mode
        mode = model.get_context_fusion_mode()
        assert mode == "disabled", f"Expected 'disabled', got '{mode}'"
        print(f"✅ Fusion mode correctly reported: {mode}")
        
        # Test forward pass still works
        batch_size = 2
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output
            
        print(f"✅ Forward pass works with disabled context fusion: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Disabled test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_fusion_factory():
    """Test the context fusion factory directly."""
    print("\n🏭 Testing Context Fusion Factory")
    print("-" * 40)
    
    try:
        from models.celestial_modules.context_fusion import ContextFusionFactory
        from models.celestial_modules.config import CelestialPGATConfig
        
        # Test factory methods
        supported_modes = ContextFusionFactory.get_supported_modes()
        print(f"✅ Supported modes: {supported_modes}")
        
        descriptions = ContextFusionFactory.get_mode_descriptions()
        print(f"✅ Mode descriptions: {len(descriptions)} modes")
        
        # Test configuration validation
        config = TestConfig()
        celestial_config = CelestialPGATConfig.from_original_configs(config)
        
        is_valid = ContextFusionFactory.validate_config(celestial_config)
        print(f"✅ Configuration validation: {is_valid}")
        
        # Test module creation
        fusion_module = ContextFusionFactory.create_context_fusion(celestial_config)
        assert fusion_module is not None, "Factory should create fusion module"
        print(f"✅ Factory created module: {type(fusion_module).__name__}")
        
        # Test recommended configurations
        try:
            financial_config = ContextFusionFactory.get_recommended_config('financial_timeseries')
            print(f"✅ Financial config: {financial_config['context_fusion_mode']}")
        except Exception as e:
            print(f"⚠️  Recommended config failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 Multi-Scale Context Fusion Integration Tests")
    print("=" * 80)
    
    tests = [
        ("Context Fusion Integration", test_context_fusion_integration),
        ("Context Fusion Disabled", test_context_fusion_disabled),
        ("Context Fusion Factory", test_context_fusion_factory),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 80)
    print("📊 Test Results Summary:")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Context fusion is properly integrated.")
        print("✨ The multi-scale context fusion is working correctly in the modular model.")
    else:
        print("⚠️  Some tests failed. Please review the integration.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)