#!/usr/bin/env python3
"""
Test script to verify MDN decoder dimension fixes
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.modular.decoder.mdn_decoder import MDNDecoder, mdn_nll_loss

def test_mdn_decoder_dimension_adaptation():
    """Test MDN decoder with different input dimensions"""
    print("Testing MDN Decoder Dimension Adaptation...")
    
    # Test configuration
    d_model = 128
    n_targets = 4
    n_components = 5
    batch_size = 8
    seq_len = 20
    
    # Create MDN decoder
    mdn_decoder = MDNDecoder(
        d_input=d_model,
        n_targets=n_targets,
        n_components=n_components,
        adaptive_input=True
    )
    
    print(f"Created MDN decoder: d_input={d_model}, n_targets={n_targets}, n_components={n_components}")
    
    # Test 1: Exact dimension match
    print("\n1. Testing exact dimension match...")
    input_exact = torch.randn(batch_size, seq_len, d_model)
    try:
        pi, mu, sigma = mdn_decoder(input_exact)
        predictions = mdn_decoder.mean_prediction(pi, mu)
        print(f"✅ Exact match: input {input_exact.shape} -> predictions {predictions.shape}")
        assert predictions.shape == (batch_size, seq_len, n_targets)
    except Exception as e:
        print(f"❌ Exact match failed: {e}")
        return False
    
    # Test 2: Different dimension (smaller)
    print("\n2. Testing smaller input dimension...")
    d_smaller = 64
    input_smaller = torch.randn(batch_size, seq_len, d_smaller)
    try:
        pi, mu, sigma = mdn_decoder(input_smaller)
        predictions = mdn_decoder.mean_prediction(pi, mu)
        print(f"✅ Smaller dim: input {input_smaller.shape} -> predictions {predictions.shape}")
        assert predictions.shape == (batch_size, seq_len, n_targets)
    except Exception as e:
        print(f"❌ Smaller dim failed: {e}")
        return False
    
    # Test 3: Different dimension (larger)
    print("\n3. Testing larger input dimension...")
    d_larger = 256
    input_larger = torch.randn(batch_size, seq_len, d_larger)
    try:
        pi, mu, sigma = mdn_decoder(input_larger)
        predictions = mdn_decoder.mean_prediction(pi, mu)
        print(f"✅ Larger dim: input {input_larger.shape} -> predictions {predictions.shape}")
        assert predictions.shape == (batch_size, seq_len, n_targets)
    except Exception as e:
        print(f"❌ Larger dim failed: {e}")
        return False
    
    # Test 4: Loss computation
    print("\n4. Testing loss computation...")
    targets = torch.randn(batch_size, seq_len, n_targets)
    try:
        loss = mdn_nll_loss(pi, mu, sigma, targets)
        print(f"✅ Loss computation: {loss.item():.6f}")
        assert torch.isfinite(loss)
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    print("\n🎉 All MDN decoder tests passed!")
    return True

def test_mdn_decoder_edge_cases():
    """Test MDN decoder edge cases"""
    print("\nTesting MDN Decoder Edge Cases...")
    
    # Test configuration
    d_model = 128
    n_targets = 4
    n_components = 3
    batch_size = 2
    seq_len = 5
    
    mdn_decoder = MDNDecoder(
        d_input=d_model,
        n_targets=n_targets,
        n_components=n_components,
        adaptive_input=True,
        sigma_min=1e-3
    )
    
    # Test 1: Very small values
    print("\n1. Testing with very small input values...")
    input_small = torch.randn(batch_size, seq_len, d_model) * 1e-6
    try:
        pi, mu, sigma = mdn_decoder(input_small)
        predictions = mdn_decoder.mean_prediction(pi, mu)
        targets = torch.randn(batch_size, seq_len, n_targets) * 1e-6
        loss = mdn_nll_loss(pi, mu, sigma, targets)
        print(f"✅ Small values: loss = {loss.item():.6f}")
        assert torch.isfinite(loss)
    except Exception as e:
        print(f"❌ Small values failed: {e}")
        return False
    
    # Test 2: Large values
    print("\n2. Testing with large input values...")
    input_large = torch.randn(batch_size, seq_len, d_model) * 10
    try:
        pi, mu, sigma = mdn_decoder(input_large)
        predictions = mdn_decoder.mean_prediction(pi, mu)
        targets = torch.randn(batch_size, seq_len, n_targets) * 10
        loss = mdn_nll_loss(pi, mu, sigma, targets)
        print(f"✅ Large values: loss = {loss.item():.6f}")
        assert torch.isfinite(loss)
    except Exception as e:
        print(f"❌ Large values failed: {e}")
        return False
    
    # Test 3: Zero values
    print("\n3. Testing with zero input values...")
    input_zero = torch.zeros(batch_size, seq_len, d_model)
    try:
        pi, mu, sigma = mdn_decoder(input_zero)
        predictions = mdn_decoder.mean_prediction(pi, mu)
        targets = torch.zeros(batch_size, seq_len, n_targets)
        loss = mdn_nll_loss(pi, mu, sigma, targets)
        print(f"✅ Zero values: loss = {loss.item():.6f}")
        assert torch.isfinite(loss)
    except Exception as e:
        print(f"❌ Zero values failed: {e}")
        return False
    
    print("\n🎉 All edge case tests passed!")
    return True

def test_mdn_decoder_non_adaptive():
    """Test MDN decoder without adaptive input"""
    print("\nTesting MDN Decoder Non-Adaptive Mode...")
    
    d_model = 128
    n_targets = 4
    n_components = 3
    batch_size = 4
    seq_len = 10
    
    # Create non-adaptive MDN decoder
    mdn_decoder = MDNDecoder(
        d_input=d_model,
        n_targets=n_targets,
        n_components=n_components,
        adaptive_input=False  # Disable adaptation
    )
    
    # Test 1: Exact dimension should work
    print("\n1. Testing exact dimension (should work)...")
    input_exact = torch.randn(batch_size, seq_len, d_model)
    try:
        pi, mu, sigma = mdn_decoder(input_exact)
        predictions = mdn_decoder.mean_prediction(pi, mu)
        print(f"✅ Non-adaptive exact: input {input_exact.shape} -> predictions {predictions.shape}")
    except Exception as e:
        print(f"❌ Non-adaptive exact failed: {e}")
        return False
    
    # Test 2: Wrong dimension should fail
    print("\n2. Testing wrong dimension (should fail gracefully)...")
    input_wrong = torch.randn(batch_size, seq_len, d_model + 10)
    try:
        pi, mu, sigma = mdn_decoder(input_wrong)
        print(f"❌ Non-adaptive wrong dimension should have failed!")
        return False
    except ValueError as e:
        print(f"✅ Non-adaptive correctly rejected wrong dimension: {e}")
    except Exception as e:
        print(f"❌ Non-adaptive failed with unexpected error: {e}")
        return False
    
    print("\n🎉 Non-adaptive tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("MDN DECODER DIMENSION FIX VALIDATION")
    print("=" * 60)
    
    success = True
    
    # Run all tests
    success &= test_mdn_decoder_dimension_adaptation()
    success &= test_mdn_decoder_edge_cases()
    success &= test_mdn_decoder_non_adaptive()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! MDN Decoder fixes are working correctly.")
        print("✅ Dimension adaptation works")
        print("✅ Edge cases handled properly")
        print("✅ Error handling is robust")
        print("✅ Loss computation is numerically stable")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60)