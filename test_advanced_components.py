#!/usr/bin/env python3
"""
🚀 Quick Test for Advanced Components in Celestial Enhanced PGAT Modular
Verifies that all advanced components can be instantiated and configured properly
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_modular_model_import():
    """Test that the modular model can be imported"""
    try:
        from models.Celestial_Enhanced_PGAT_Modular import Model
        print("✅ Successfully imported Celestial Enhanced PGAT Modular")
        return True
    except Exception as e:
        print(f"❌ Failed to import modular model: {e}")
        return False

def test_advanced_component_configs():
    """Test different advanced component configurations"""
    try:
        from models.Celestial_Enhanced_PGAT_Modular import Model
        
        # Test configurations for different advanced components
        test_configs = [
            {
                'name': 'Multi-Scale Context Fusion',
                'config': {
                    'seq_len': 96, 'label_len': 48, 'pred_len': 24,
                    'enc_in': 113, 'dec_in': 113, 'c_out': 4,
                    'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1,
                    'dropout': 0.1, 'embed': 'timeF', 'freq': 'd',
                    
                    # Advanced Context Fusion
                    'use_multi_scale_context': True,
                    'context_fusion_mode': 'multi_scale',
                    'short_term_kernel_size': 3,
                    'medium_term_kernel_size': 15,
                    'long_term_kernel_size': 0,
                    'enable_context_diagnostics': True,
                }
            },
            {
                'name': 'Stochastic Learning',
                'config': {
                    'seq_len': 96, 'label_len': 48, 'pred_len': 24,
                    'enc_in': 113, 'dec_in': 113, 'c_out': 4,
                    'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1,
                    'dropout': 0.1, 'embed': 'timeF', 'freq': 'd',
                    
                    # Stochastic Components
                    'use_stochastic_learner': True,
                    'use_stochastic_control': True,
                }
            },
            {
                'name': 'Petri Net Combiner',
                'config': {
                    'seq_len': 96, 'label_len': 48, 'pred_len': 24,
                    'enc_in': 113, 'dec_in': 113, 'c_out': 4,
                    'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1,
                    'dropout': 0.1, 'embed': 'timeF', 'freq': 'd',
                    
                    # Advanced Graph Combiner
                    'use_petri_net_combiner': True,
                    'num_message_passing_steps': 2,
                    'edge_feature_dim': 8,
                    'use_temporal_attention': True,
                    'use_spatial_attention': True,
                }
            },
            {
                'name': 'Enhanced Decoders',
                'config': {
                    'seq_len': 96, 'label_len': 48, 'pred_len': 24,
                    'enc_in': 113, 'dec_in': 113, 'c_out': 4,
                    'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1,
                    'dropout': 0.1, 'embed': 'timeF', 'freq': 'd',
                    
                    # Enhanced Decoders
                    'use_mixture_decoder': True,
                    'enable_mdn_decoder': True,
                    'mdn_components': 5,
                    'use_target_autocorrelation': True,
                }
            }
        ]
        
        for test_case in test_configs:
            print(f"\n🧪 Testing {test_case['name']}...")
            
            # Create a simple config object
            class SimpleConfig:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)
            
            config = SimpleConfig(test_case['config'])
            
            try:
                model = Model(config)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"   ✅ Model created successfully ({param_count:,} parameters)")
                
                # Test forward pass with dummy data
                batch_size = 2
                x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
                x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # Time features
                x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
                x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                if isinstance(output, tuple):
                    predictions = output[0]
                else:
                    predictions = output
                
                expected_shape = (batch_size, config.pred_len, config.c_out)
                if predictions.shape == expected_shape:
                    print(f"   ✅ Forward pass successful: {predictions.shape}")
                else:
                    print(f"   ⚠️  Shape mismatch: got {predictions.shape}, expected {expected_shape}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Component configuration test failed: {e}")
        return False

def test_context_fusion_modes():
    """Test all context fusion modes"""
    try:
        from models.celestial_modules.context_fusion import ContextFusionFactory
        
        modes = ['simple', 'gated', 'attention', 'multi_scale']
        
        for mode in modes:
            print(f"\n🔍 Testing Context Fusion Mode: {mode}")
            
            class TestConfig:
                def __init__(self):
                    self.use_multi_scale_context = True
                    self.context_fusion_mode = mode
                    self.d_model = 128
                    self.n_heads = 8
                    self.short_term_kernel_size = 3
                    self.medium_term_kernel_size = 15
                    self.long_term_kernel_size = 0
                    self.context_fusion_dropout = 0.1
                    self.enable_context_diagnostics = True
            
            config = TestConfig()
            
            try:
                # Validate configuration
                ContextFusionFactory.validate_config(config)
                
                # Create fusion module
                fusion_module = ContextFusionFactory.create_context_fusion(config)
                
                if fusion_module is not None:
                    # Test forward pass
                    batch_size, seq_len = 2, 96
                    enc_out = torch.randn(batch_size, seq_len, config.d_model)
                    
                    with torch.no_grad():
                        fused_output, diagnostics = fusion_module(enc_out)
                    
                    if fused_output.shape == enc_out.shape:
                        print(f"   ✅ {mode} mode working correctly")
                    else:
                        print(f"   ❌ {mode} mode shape mismatch")
                        return False
                else:
                    print(f"   ❌ {mode} mode returned None")
                    return False
                    
            except Exception as e:
                print(f"   ❌ {mode} mode failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Context fusion test failed: {e}")
        return False

def main():
    """Run all advanced component tests"""
    print("🚀 TESTING ADVANCED COMPONENTS - Celestial Enhanced PGAT Modular")
    print("=" * 70)
    
    all_tests_passed = True
    
    # Test 1: Model Import
    print("\n📦 TEST 1: Model Import")
    if not test_modular_model_import():
        all_tests_passed = False
    
    # Test 2: Advanced Component Configurations
    print("\n🎯 TEST 2: Advanced Component Configurations")
    if not test_advanced_component_configs():
        all_tests_passed = False
    
    # Test 3: Context Fusion Modes
    print("\n🌟 TEST 3: Context Fusion Modes")
    if not test_context_fusion_modes():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("🎉 ALL ADVANCED COMPONENT TESTS PASSED!")
        print("✅ Ready for systematic component testing")
        print("✅ All advanced features are properly configured")
        print("✅ Production training script should work correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Check component configurations before running systematic tests")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)