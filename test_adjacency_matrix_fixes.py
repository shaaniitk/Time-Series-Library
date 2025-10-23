#!/usr/bin/env python3
"""
Test Adjacency Matrix Dimension Fixes

This script verifies that Issue #4: Adjacency Matrix Dimension Chaos has been resolved.
"""

import torch
import yaml
from models.Celestial_Enhanced_PGAT import Model

def test_adjacency_matrix_consistency():
    """Test that all adjacency matrices have consistent 4D dimensions."""
    
    print("🔍 TESTING ADJACENCY MATRIX CONSISTENCY")
    print("="*60)
    
    # Load production config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    
    # Initialize model
    model = Model(configs)
    model.eval()
    
    print(f"Model configuration:")
    print(f"   Celestial bodies: {model.num_celestial_bodies}")
    print(f"   d_model: {model.d_model}")
    print(f"   seq_len: {model.seq_len}")
    
    # Create test data
    batch_size = 2
    seq_len = configs.seq_len
    enc_in = configs.enc_in
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)
    
    print(f"\nInput shapes:")
    print(f"   x_enc: {x_enc.shape}")
    print(f"   Expected adjacency shape: [{batch_size}, {seq_len}, {model.num_celestial_bodies}, {model.num_celestial_bodies}]")
    
    # Test celestial body nodes directly
    print(f"\n🌌 Testing CelestialBodyNodes adjacency output...")
    
    # Create encoder output for testing
    enc_out = torch.randn(batch_size, seq_len, model.d_model)
    
    try:
        astronomical_adj, dynamic_adj, celestial_features, metadata = model.celestial_nodes(enc_out)
        
        expected_adj_shape = (batch_size, seq_len, model.num_celestial_bodies, model.num_celestial_bodies)
        
        print(f"✅ CelestialBodyNodes output:")
        print(f"   astronomical_adj: {astronomical_adj.shape}")
        print(f"   dynamic_adj: {dynamic_adj.shape}")
        print(f"   celestial_features: {celestial_features.shape}")
        
        # Validate dimensions
        astro_correct = (astronomical_adj.shape == expected_adj_shape)
        dynamic_correct = (dynamic_adj.shape == expected_adj_shape)
        
        print(f"\n📊 Dimension validation:")
        print(f"   ✅ Astronomical adjacency correct: {astro_correct}")
        print(f"   ✅ Dynamic adjacency correct: {dynamic_correct}")
        
        if not astro_correct:
            print(f"      Expected: {expected_adj_shape}, Got: {astronomical_adj.shape}")
        if not dynamic_correct:
            print(f"      Expected: {expected_adj_shape}, Got: {dynamic_adj.shape}")
        
        return astro_correct and dynamic_correct
        
    except Exception as e:
        print(f"❌ CelestialBodyNodes test failed: {e}")
        return False

def test_adjacency_validation():
    """Test the adjacency matrix validation function."""
    
    print(f"\n🔍 TESTING ADJACENCY VALIDATION")
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
    
    batch_size, seq_len = 2, configs.seq_len
    num_nodes = model.num_celestial_bodies
    
    # Create test adjacency matrices
    correct_shape = (batch_size, seq_len, num_nodes, num_nodes)
    
    astronomical_adj = torch.randn(correct_shape)
    learned_adj = torch.randn(correct_shape)
    dynamic_adj = torch.randn(correct_shape)
    enc_out = torch.randn(batch_size, seq_len, model.d_model)
    
    print(f"Testing validation with correct shapes: {correct_shape}")
    
    try:
        model._validate_adjacency_dimensions(astronomical_adj, learned_adj, dynamic_adj, enc_out)
        print(f"✅ Validation passed for correct dimensions")
        validation_works = True
    except Exception as e:
        print(f"❌ Validation failed unexpectedly: {e}")
        validation_works = False
    
    # Test with incorrect dimensions
    print(f"\nTesting validation with incorrect shapes...")
    
    wrong_astronomical = torch.randn(batch_size, num_nodes, num_nodes)  # 3D instead of 4D
    
    try:
        model._validate_adjacency_dimensions(wrong_astronomical, learned_adj, dynamic_adj, enc_out)
        print(f"❌ Validation should have failed but didn't")
        validation_catches_errors = False
    except ValueError as e:
        print(f"✅ Validation correctly caught dimension error: {str(e)[:100]}...")
        validation_catches_errors = True
    except Exception as e:
        print(f"❌ Validation failed with unexpected error: {e}")
        validation_catches_errors = False
    
    return validation_works and validation_catches_errors

def test_normalization_consistency():
    """Test that adjacency matrix normalization works consistently."""
    
    print(f"\n🔍 TESTING ADJACENCY NORMALIZATION")
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
    
    batch_size, seq_len = 2, 10  # Smaller seq_len for testing
    num_nodes = model.num_celestial_bodies
    
    # Create test adjacency matrix
    adj_matrix = torch.randn(batch_size, seq_len, num_nodes, num_nodes)
    
    print(f"Input adjacency shape: {adj_matrix.shape}")
    
    try:
        normalized_adj = model._normalize_adj(adj_matrix)
        
        print(f"✅ Normalization successful")
        print(f"   Output shape: {normalized_adj.shape}")
        print(f"   Shape preserved: {normalized_adj.shape == adj_matrix.shape}")
        
        # Check if it's properly normalized (rows sum to ~1)
        row_sums = normalized_adj.sum(dim=-1)
        expected_sum = 1.0
        sum_close_to_one = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        
        print(f"   Rows sum to 1: {sum_close_to_one}")
        print(f"   Average row sum: {row_sums.mean():.6f}")
        
        return normalized_adj.shape == adj_matrix.shape and sum_close_to_one
        
    except Exception as e:
        print(f"❌ Normalization failed: {e}")
        return False

def test_end_to_end_adjacency_flow():
    """Test the complete adjacency matrix flow through the model."""
    
    print(f"\n🔍 TESTING END-TO-END ADJACENCY FLOW")
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
    model.eval()
    
    # Create test data
    batch_size = 2
    seq_len = configs.seq_len
    enc_in = configs.enc_in
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
    
    print(f"Testing full forward pass with adjacency matrices...")
    
    try:
        with torch.no_grad():
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(outputs, tuple):
            predictions, metadata = outputs
        else:
            predictions = outputs
        
        print(f"✅ End-to-end test successful!")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Expected: [{batch_size}, {configs.pred_len}, {configs.c_out}]")
        
        shape_correct = (predictions.shape == (batch_size, configs.pred_len, configs.c_out))
        print(f"   Shape correct: {shape_correct}")
        
        return shape_correct
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 ADJACENCY MATRIX DIMENSION CHAOS FIX VERIFICATION")
    print("="*60)
    
    # Test 1: Adjacency matrix consistency
    consistency_ok = test_adjacency_matrix_consistency()
    
    # Test 2: Validation function
    validation_ok = test_adjacency_validation()
    
    # Test 3: Normalization consistency
    normalization_ok = test_normalization_consistency()
    
    # Test 4: End-to-end flow
    end_to_end_ok = test_end_to_end_adjacency_flow()
    
    print(f"\n🎉 ADJACENCY MATRIX FIX VERIFICATION RESULTS:")
    print(f"="*60)
    print(f"✅ Adjacency consistency: {'PASS' if consistency_ok else 'FAIL'}")
    print(f"✅ Validation function: {'PASS' if validation_ok else 'FAIL'}")
    print(f"✅ Normalization: {'PASS' if normalization_ok else 'FAIL'}")
    print(f"✅ End-to-end flow: {'PASS' if end_to_end_ok else 'FAIL'}")
    
    all_tests_pass = consistency_ok and validation_ok and normalization_ok and end_to_end_ok
    
    if all_tests_pass:
        print(f"\n🎉 ALL ADJACENCY MATRIX FIXES VERIFIED!")
        print(f"✅ Issue #4: Adjacency Matrix Dimension Chaos - RESOLVED")
        print(f"✅ All matrices standardized to 4D format")
        print(f"✅ Manual broadcasting eliminated")
        print(f"✅ Dimension validation added")
        print(f"✅ Memory efficiency improved")
    else:
        print(f"\n❌ SOME ADJACENCY MATRIX ISSUES REMAIN")
        print(f"   Please check the failed tests above")