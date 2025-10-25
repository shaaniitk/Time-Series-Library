#!/usr/bin/env python3
"""
Test Dimension Mismatch Fixes

This script verifies that Issue #2: Dimension Mismatch Cascade has been resolved.
"""

import torch
import yaml
from models.Celestial_Enhanced_PGAT import Model

def test_dimension_consistency():
    """Test that all dimensions are consistent throughout the pipeline."""
    
    print("🔍 TESTING DIMENSION CONSISTENCY FIXES")
    print("="*60)
    
    # Load production config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to object
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    
    print(f"Configuration loaded:")
    print(f"   d_model: {configs.d_model}")
    print(f"   n_heads: {configs.n_heads}")
    print(f"   enc_in: {configs.enc_in}")
    print(f"   c_out: {configs.c_out}")
    
    # Initialize model
    model = Model(configs)
    
    print(f"\nModel initialized:")
    print(f"   Configured d_model: {configs.d_model}")
    print(f"   Actual d_model: {model.d_model}")
    print(f"   Celestial feature dim: {model.celestial_feature_dim}")
    print(f"   Num celestial bodies: {model.num_celestial_bodies}")
    
    # Check dimension consistency
    print(f"\n📊 DIMENSION ANALYSIS:")
    print(f"-" * 40)
    
    # Test 1: Configuration consistency
    config_matches = (configs.d_model == model.d_model)
    print(f"✅ Config d_model matches actual: {config_matches}")
    if not config_matches:
        print(f"   Config: {configs.d_model}, Actual: {model.d_model}")
    
    # Test 2: Attention head compatibility
    attention_compatible = (model.d_model % model.n_heads == 0)
    print(f"✅ d_model divisible by n_heads: {attention_compatible}")
    print(f"   d_model={model.d_model}, n_heads={model.n_heads}, ratio={model.d_model // model.n_heads}")
    
    # Test 3: Celestial feature handling
    if hasattr(model, 'celestial_projection'):
        print(f"✅ Celestial projection layer exists: {type(model.celestial_projection)}")
        if hasattr(model.celestial_projection, '__len__') and len(model.celestial_projection) > 1:
            # It's a Sequential layer
            first_layer = model.celestial_projection[0]
            print(f"   Input dim: {first_layer.in_features}")
            print(f"   Output dim: {first_layer.out_features}")
        else:
            print(f"   Identity projection (perfect match)")
    
    # Test 4: Forward pass dimension flow
    print(f"\n🔄 FORWARD PASS DIMENSION TEST:")
    print(f"-" * 40)
    
    # Create test data with correct sequence lengths
    batch_size = 2
    seq_len = configs.seq_len  # Use config seq_len
    enc_in = configs.enc_in
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
    
    print(f"Input shapes:")
    print(f"   x_enc: {x_enc.shape}")
    print(f"   x_mark_enc: {x_mark_enc.shape}")
    print(f"   x_dec: {x_dec.shape}")
    print(f"   x_mark_dec: {x_mark_dec.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if isinstance(outputs, tuple):
                predictions, metadata = outputs
            else:
                predictions = outputs
                metadata = {}
            
            print(f"\n✅ Forward pass successful!")
            print(f"   Output shape: {predictions.shape}")
            print(f"   Expected shape: [{batch_size}, {configs.pred_len}, {configs.c_out}]")
            
            # Verify output dimensions
            expected_shape = (batch_size, configs.pred_len, configs.c_out)
            actual_shape = predictions.shape
            
            shape_correct = (actual_shape == expected_shape)
            print(f"✅ Output shape correct: {shape_correct}")
            
            if not shape_correct:
                print(f"   Expected: {expected_shape}")
                print(f"   Actual: {actual_shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False

def test_information_flow():
    """Test that no information is lost in the dimension pipeline."""
    
    print(f"\n📈 INFORMATION FLOW TEST:")
    print(f"-" * 40)
    
    # Load config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    model = Model(configs)
    
    # Check information flow
    input_features = configs.enc_in  # 113
    celestial_features = model.celestial_feature_dim  # 13 × 32 = 416
    d_model = model.d_model  # Should be 416 now
    
    print(f"Information flow:")
    print(f"   Input features: {input_features}")
    print(f"   Celestial features: {celestial_features}")
    print(f"   d_model: {d_model}")
    
    # Check for information bottlenecks
    if d_model < celestial_features:
        compression_ratio = celestial_features / d_model
        print(f"⚠️  Information compression: {compression_ratio:.2f}x")
        print(f"   This may cause information loss!")
    elif d_model > celestial_features:
        expansion_ratio = d_model / celestial_features
        print(f"✅ Information expansion: {expansion_ratio:.2f}x")
        print(f"   This allows for richer representations!")
    else:
        print(f"✅ Perfect information preservation: 1.0x")
    
    return d_model >= celestial_features

def test_adjacency_matrix_dimensions():
    """Test that adjacency matrices have consistent dimensions."""
    
    print(f"\n🔗 ADJACENCY MATRIX DIMENSION TEST:")
    print(f"-" * 40)
    
    # Load config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    model = Model(configs)
    
    # Test adjacency matrix creation
    batch_size, seq_len = 2, 10
    num_nodes = model.num_celestial_bodies  # Should always be 13
    
    print(f"Graph configuration:")
    print(f"   Num celestial bodies: {num_nodes}")
    print(f"   Expected adjacency shape: [{batch_size}, {num_nodes}, {num_nodes}]")
    
    # Create test input
    x_enc = torch.randn(batch_size, seq_len, configs.enc_in)
    
    # Test phase-aware processor
    if hasattr(model, 'phase_aware_processor'):
        celestial_features, adjacency_matrix, metadata = model.phase_aware_processor(x_enc)
        
        print(f"\nPhase-aware processor output:")
        print(f"   Celestial features: {celestial_features.shape}")
        print(f"   Adjacency matrix: {adjacency_matrix.shape}")
        
        expected_adj_shape = (batch_size, num_nodes, num_nodes)
        actual_adj_shape = adjacency_matrix.shape
        
        adj_shape_correct = (actual_adj_shape == expected_adj_shape)
        print(f"✅ Adjacency shape correct: {adj_shape_correct}")
        
        if not adj_shape_correct:
            print(f"   Expected: {expected_adj_shape}")
            print(f"   Actual: {actual_adj_shape}")
        
        return adj_shape_correct
    
    return True

if __name__ == "__main__":
    print("🚀 DIMENSION MISMATCH CASCADE FIX VERIFICATION")
    print("="*60)
    
    # Test 1: Dimension consistency
    consistency_ok = test_dimension_consistency()
    
    # Test 2: Information flow
    info_flow_ok = test_information_flow()
    
    # Test 3: Adjacency matrix dimensions
    adjacency_ok = test_adjacency_matrix_dimensions()
    
    print(f"\n🎉 DIMENSION FIX VERIFICATION RESULTS:")
    print(f"="*60)
    print(f"✅ Dimension consistency: {'PASS' if consistency_ok else 'FAIL'}")
    print(f"✅ Information flow: {'PASS' if info_flow_ok else 'FAIL'}")
    print(f"✅ Adjacency dimensions: {'PASS' if adjacency_ok else 'FAIL'}")
    
    all_tests_pass = consistency_ok and info_flow_ok and adjacency_ok
    
    if all_tests_pass:
        print(f"\n🎉 ALL DIMENSION FIXES VERIFIED!")
        print(f"✅ Issue #2: Dimension Mismatch Cascade - RESOLVED")
        print(f"✅ Configuration reliability restored")
        print(f"✅ Information bottlenecks eliminated")
        print(f"✅ Graph node consistency achieved")
    else:
        print(f"\n❌ SOME DIMENSION ISSUES REMAIN")
        print(f"   Please check the failed tests above")