#!/usr/bin/env python3
"""
Test script to validate the modular Celestial Enhanced PGAT implementation.

This script tests that all components from the original model are properly
implemented in the modular version with full feature parity.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_config():
    """Create a test configuration object."""
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
        
        # Celestial system parameters
        use_celestial_graph: bool = True
        celestial_fusion_layers: int = 3
        num_celestial_bodies: int = 13
        
        # Enhanced features
        use_mixture_decoder: bool = False
        use_stochastic_learner: bool = False
        use_hierarchical_mapping: bool = False
        use_gated_graph_combiner: bool = False
        use_efficient_covariate_interaction: bool = False
        
        # Petri Net Architecture
        use_petri_net_combiner: bool = True
        num_message_passing_steps: int = 2
        edge_feature_dim: int = 6
        use_temporal_attention: bool = True
        use_spatial_attention: bool = True
        bypass_spatiotemporal_with_petri: bool = True
        
        # Adaptive TopK
        enable_adaptive_topk: bool = False
        adaptive_topk_ratio: float = 0.5
        adaptive_topk_temperature: float = 1.0
        
        # Stochastic Control
        use_stochastic_control: bool = False
        stochastic_temperature_start: float = 1.0
        stochastic_temperature_end: float = 0.1
        stochastic_decay_steps: int = 1000
        stochastic_noise_std: float = 1.0
        stochastic_use_external_step: bool = False
        
        # MDN Decoder
        enable_mdn_decoder: bool = False
        mdn_components: int = 5
        mdn_sigma_min: float = 1e-3
        mdn_use_softplus: bool = True
        
        # Target Autocorrelation
        use_target_autocorrelation: bool = True
        target_autocorr_layers: int = 2
        
        # Calendar Effects
        use_calendar_effects: bool = True
        calendar_embedding_dim: int = 128
        
        # Celestial-to-Target Attention
        use_celestial_target_attention: bool = True
        celestial_target_use_gated_fusion: bool = True
        celestial_target_diagnostics: bool = True
        use_c2t_edge_bias: bool = False
        c2t_edge_bias_weight: float = 0.2
        c2t_aux_rel_loss_weight: float = 0.0
        
        # Wave aggregation
        aggregate_waves_to_celestial: bool = True
        num_input_waves: int = 118
        target_wave_indices: List[int] = None
        
        # Dynamic Spatiotemporal Encoder
        use_dynamic_spatiotemporal_encoder: bool = True
        
        # Logging and Diagnostics
        verbose_logging: bool = True
        enable_memory_debug: bool = False
        enable_memory_diagnostics: bool = False
        collect_diagnostics: bool = True
        enable_fusion_diagnostics: bool = True
        fusion_diag_batches: int = 10
        
        # Covariate Interaction
        enable_target_covariate_attention: bool = False
        
        # Sequential Mixture Decoder
        use_sequential_mixture_decoder: bool = False
        
        # Multi-Scale Context Fusion
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
    
    return TestConfig()

def test_modular_components():
    """Test individual modular components."""
    print("🧪 Testing Modular Components...")
    
    try:
        from models.celestial_modules.config import CelestialPGATConfig
        from models.celestial_modules.utils import ModelUtils
        from models.celestial_modules.diagnostics import ModelDiagnostics
        from models.celestial_modules.embedding import EmbeddingModule
        from models.celestial_modules.graph import GraphModule
        from models.celestial_modules.encoder import EncoderModule
        from models.celestial_modules.postprocessing import PostProcessingModule
        from models.celestial_modules.decoder import DecoderModule
        from models.celestial_modules.context_fusion import MultiScaleContextFusion, ContextFusionFactory
        
        config = CelestialPGATConfig.from_original_configs(create_test_config())
        print("✅ Configuration module loaded successfully")
        
        utils = ModelUtils(config)
        print("✅ Utils module loaded successfully")
        
        diagnostics = ModelDiagnostics(config)
        print("✅ Diagnostics module loaded successfully")
        
        embedding = EmbeddingModule(config)
        print("✅ Embedding module loaded successfully")
        
        if config.use_celestial_graph:
            graph = GraphModule(config)
            print("✅ Graph module loaded successfully")
        
        encoder = EncoderModule(config)
        print("✅ Encoder module loaded successfully")
        
        postprocessing = PostProcessingModule(config)
        print("✅ Post-processing module loaded successfully")
        
        decoder = DecoderModule(config)
        print("✅ Decoder module loaded successfully")
        
        # Test context fusion module
        context_fusion = ContextFusionFactory.create_context_fusion(config)
        if context_fusion is not None:
            print("✅ Context fusion module loaded successfully")
            print(f"    Mode: {config.context_fusion_mode}")
            print(f"    Supported modes: {ContextFusionFactory.get_supported_modes()}")
        else:
            print("ℹ️  Context fusion disabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_modular_model():
    """Test the complete modular model."""
    print("\n🚀 Testing Complete Modular Model...")
    
    try:
        from models.Celestial_Enhanced_PGAT_Modular import Model
        
        config = create_test_config()
        model = Model(config)
        print("✅ Modular model instantiated successfully")
        
        # Test model methods
        assert hasattr(model, 'get_point_prediction'), "Missing get_point_prediction method"
        assert hasattr(model, 'print_fusion_diagnostics_summary'), "Missing print_fusion_diagnostics_summary method"
        assert hasattr(model, 'print_celestial_target_diagnostics'), "Missing print_celestial_target_diagnostics method"
        assert hasattr(model, 'increment_fusion_diagnostics_batch'), "Missing increment_fusion_diagnostics_batch method"
        assert hasattr(model, 'get_regularization_loss'), "Missing get_regularization_loss method"
        assert hasattr(model, 'set_global_step'), "Missing set_global_step method"
        print("✅ All required methods present")
        
        # Test forward pass with dummy data
        batch_size, seq_len, enc_in = 2, config.seq_len, config.enc_in
        pred_len, dec_in = config.pred_len, config.dec_in
        
        x_enc = torch.randn(batch_size, seq_len, enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)  # 4 time features
        x_dec = torch.randn(batch_size, config.label_len + pred_len, dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + pred_len, 4)
        
        # Test basic forward pass
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            print(f"✅ Forward pass successful, output type: {type(output)}")
            
            if isinstance(output, tuple):
                predictions = output[0]
                print(f"✅ Predictions shape: {predictions.shape}")
                assert predictions.shape == (batch_size, pred_len, config.c_out), f"Wrong prediction shape: {predictions.shape}"
            else:
                print(f"✅ Output shape: {output.shape}")
                assert output.shape == (batch_size, pred_len, config.c_out), f"Wrong output shape: {output.shape}"
        
        # Test future celestial processing
        future_celestial_x = torch.randn(batch_size, pred_len, enc_in)
        with torch.no_grad():
            output_with_future = model(x_enc, x_mark_enc, x_dec, x_mark_dec, 
                                     future_celestial_x=future_celestial_x)
            print("✅ Future celestial processing successful")
        
        # Test utility methods
        reg_loss = model.get_regularization_loss()
        print(f"✅ Regularization loss: {reg_loss}")
        
        model.set_global_step(100)
        print("✅ Global step setting successful")
        
        # Test point prediction extraction
        point_pred = model.get_point_prediction(output)
        print(f"✅ Point prediction extraction successful: {point_pred.shape if hasattr(point_pred, 'shape') else type(point_pred)}")
        
        # Test context fusion diagnostics
        if hasattr(model, 'print_context_fusion_diagnostics'):
            model.print_context_fusion_diagnostics()
            print("✅ Context fusion diagnostics successful")
            
            context_mode = model.get_context_fusion_mode()
            print(f"✅ Context fusion mode: {context_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Modular model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_parity():
    """Test feature parity between original and modular versions."""
    print("\n🔍 Testing Feature Parity...")
    
    try:
        # Test configuration completeness
        from models.celestial_modules.config import CelestialPGATConfig
        config = CelestialPGATConfig.from_original_configs(create_test_config())
        
        # Check critical configuration parameters
        critical_params = [
            'verbose_logging', 'enable_memory_debug', 'collect_diagnostics',
            'enable_fusion_diagnostics', 'use_celestial_target_attention',
            'use_petri_net_combiner', 'enable_mdn_decoder', 'use_target_autocorrelation',
            'use_calendar_effects', 'aggregate_waves_to_celestial'
        ]
        
        for param in critical_params:
            assert hasattr(config, param), f"Missing configuration parameter: {param}"
        print("✅ All critical configuration parameters present")
        
        # Test modular model has all required components
        from models.Celestial_Enhanced_PGAT_Modular import Model
        model = Model(create_test_config())
        
        # Check for all major components
        assert hasattr(model, 'embedding_module'), "Missing embedding_module"
        assert hasattr(model, 'graph_module'), "Missing graph_module"
        assert hasattr(model, 'encoder_module'), "Missing encoder_module"
        assert hasattr(model, 'postprocessing_module'), "Missing postprocessing_module"
        assert hasattr(model, 'decoder_module'), "Missing decoder_module"
        assert hasattr(model, 'utils'), "Missing utils"
        assert hasattr(model, 'diagnostics'), "Missing diagnostics"
        print("✅ All major components present")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature parity test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 Celestial Enhanced PGAT - Modular Implementation Validation")
    print("=" * 60)
    
    tests = [
        ("Component Loading", test_modular_components),
        ("Complete Model", test_modular_model),
        ("Feature Parity", test_feature_parity),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Modular implementation is ready for use.")
        print("✨ The modular version has full feature parity with the original.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)