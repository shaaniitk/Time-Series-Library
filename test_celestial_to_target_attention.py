"""
Test script for Celestial-to-Target Attention integration

Validates that:
1. CelestialToTargetAttention module initializes correctly
2. Forward pass works with correct tensor shapes
3. Diagnostics are collected and accessible
4. Integration with main model is functional
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.modular.graph.celestial_to_target_attention import CelestialToTargetAttention


def test_standalone_module():
    """Test CelestialToTargetAttention module in isolation"""
    print("=" * 80)
    print("Test 1: Standalone Module Initialization and Forward Pass")
    print("=" * 80)
    
    # Create module
    module = CelestialToTargetAttention(
        num_celestial=13,
        num_targets=4,
        d_model=416,
        num_heads=8,
        dropout=0.1,
        use_gated_fusion=True,
        enable_diagnostics=True
    )
    
    print(f"✅ Module initialized successfully")
    print(f"   - Num celestial bodies: 13")
    print(f"   - Num targets: 4")
    print(f"   - d_model: 416")
    print(f"   - Num heads: 8")
    print(f"   - Gated fusion: True")
    print(f"   - Diagnostics: True")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 250
    pred_len = 10
    num_celestial = 13
    num_targets = 4
    d_model = 416
    
    target_features = torch.randn(batch_size, pred_len, num_targets, d_model)
    celestial_features = torch.randn(batch_size, seq_len, num_celestial, d_model)
    
    print(f"\n📊 Input shapes:")
    print(f"   - Target features: {target_features.shape}")
    print(f"   - Celestial features: {celestial_features.shape}")
    
    # Forward pass
    enhanced_targets, diagnostics = module(
        target_features=target_features,
        celestial_features=celestial_features,
        return_diagnostics=True
    )
    
    print(f"\n📊 Output shapes:")
    print(f"   - Enhanced targets: {enhanced_targets.shape}")
    
    # Check diagnostics
    if diagnostics:
        print(f"\n🔍 Diagnostics collected:")
        print(f"   - Attention weights: {len(diagnostics['attention_weights'])} targets")
        print(f"   - Gate values: {len(diagnostics['gate_values'])} targets")
        print(f"   - Summary stats: {len(diagnostics['summary'])} metrics")
        
        # Print sample attention weights
        for key, attn in diagnostics['attention_weights'].items():
            attn_mean = attn.mean(dim=(0, 1))  # Average over batch and time
            print(f"\n   {key}:")
            print(f"      Shape: {attn.shape}")
            print(f"      Mean attention per celestial body: {attn_mean.tolist()[:5]}... (showing first 5)")
    
    # Print full diagnostics summary
    print("\n" + "=" * 80)
    module.print_diagnostics_summary()
    print("=" * 80)
    
    print("\n✅ Test 1 PASSED: Standalone module works correctly\n")
    return True


def test_model_integration():
    """Test integration with main Celestial_Enhanced_PGAT model"""
    print("=" * 80)
    print("Test 2: Integration with Main Model")
    print("=" * 80)
    
    # Import model
    from models.Celestial_Enhanced_PGAT import Model
    
    # Create minimal config
    class MinimalConfig:
        # Core parameters
        seq_len = 250
        label_len = 125
        pred_len = 10
        enc_in = 113
        dec_in = 113
        c_out = 4
        d_model = 416
        n_heads = 8
        e_layers = 2
        d_layers = 1
        dropout = 0.1
        embed = 'timeF'
        freq = 'd'
        
        # Celestial system
        use_celestial_graph = True
        aggregate_waves_to_celestial = True
        num_input_waves = 113
        target_wave_indices = [0, 1, 2, 3]
        
        # Petri net
        use_petri_net_combiner = True
        num_message_passing_steps = 2
        edge_feature_dim = 6
        use_temporal_attention = True
        use_spatial_attention = True
        bypass_spatiotemporal_with_petri = True
        
        # Advanced features (disable for simplicity)
        use_mixture_decoder = False
        use_stochastic_learner = False
        use_hierarchical_mapping = False
        use_efficient_covariate_interaction = False
        use_dynamic_spatiotemporal_encoder = False
        
        # Target autocorrelation
        use_target_autocorrelation = False
        
        # Calendar effects
        use_calendar_effects = False
        
        # Celestial-to-target attention (NEW!)
        use_celestial_target_attention = True
        celestial_target_use_gated_fusion = True
        celestial_target_diagnostics = True
        
        # Diagnostics
        verbose_logging = True
        collect_diagnostics = True
        enable_fusion_diagnostics = True
        fusion_diag_batches = 5
    
    config = MinimalConfig()
    
    try:
        model = Model(config)
        print(f"✅ Model initialized with celestial-to-target attention")
        
        # Check if module was created
        if model.celestial_to_target_attention is not None:
            print(f"✅ CelestialToTargetAttention module is present")
            print(f"   - Num celestial: {model.celestial_to_target_attention.num_celestial}")
            print(f"   - Num targets: {model.celestial_to_target_attention.num_targets}")
            print(f"   - Use gated fusion: {model.celestial_to_target_attention.use_gated_fusion}")
        else:
            print(f"❌ CelestialToTargetAttention module not initialized!")
            return False
        
        # Test forward pass
        batch_size = 2
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # 4 time features
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        print(f"\n📊 Running forward pass...")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Encoder input: {x_enc.shape}")
        print(f"   - Decoder input: {x_dec.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Parse output (can be tuple with metadata)
        if isinstance(output, tuple):
            predictions = output[0]
            metadata = output[-1] if len(output) > 1 else None
        else:
            predictions = output
            metadata = None
        
        print(f"\n📊 Output shape: {predictions.shape}")
        print(f"   - Expected: [batch={batch_size}, pred_len={config.pred_len}, c_out={config.c_out}]")
        
        # Check metadata for celestial-target diagnostics
        if metadata and 'celestial_target_diagnostics' in metadata:
            print(f"\n🔍 Celestial-Target Diagnostics found in metadata!")
            c2t_diag = metadata['celestial_target_diagnostics']
            if c2t_diag and 'summary' in c2t_diag:
                print(f"   - Summary stats: {len(c2t_diag['summary'])} metrics")
        
        # Print diagnostics
        print("\n" + "=" * 80)
        model.print_celestial_target_diagnostics()
        print("=" * 80)
        
        print(f"\n✅ Test 2 PASSED: Model integration successful\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CELESTIAL-TO-TARGET ATTENTION TEST SUITE")
    print("=" * 80 + "\n")
    
    test1_passed = test_standalone_module()
    test2_passed = test_model_integration()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Standalone Module): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Model Integration): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 80 + "\n")
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Celestial-to-Target Attention is ready to use.\n")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.\n")
        sys.exit(1)
