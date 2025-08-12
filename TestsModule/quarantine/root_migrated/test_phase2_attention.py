#!/usr/bin/env python3
"""
Quick Phase 2 Attention Components Validation Test

This script validates that the Phase 2 attention components work correctly.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_phase2_components():
    """Quick test of Phase 2 attention components"""
    print("🚀 Testing Phase 2 Attention Components")
    print("=" * 60)
    
    # Test parameters
    batch_size, seq_len, d_model, n_heads = 2, 32, 64, 4
    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Fourier Attention
    try:
        from layers.modular.attention.fourier_attention import FourierAttention
        fourier_attn = FourierAttention(d_model, n_heads)
        output, _ = fourier_attn(queries, keys, values)
        assert output.shape == queries.shape
        print("✅ FourierAttention: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FourierAttention: FAIL - {e}")
    total_tests += 1
    
    # Test 2: Wavelet Attention
    try:
        from layers.modular.attention.wavelet_attention import WaveletAttention
        wavelet_attn = WaveletAttention(d_model, n_heads, n_levels=3)
        output, _ = wavelet_attn(queries, keys, values)
        assert output.shape == queries.shape
        print("✅ WaveletAttention: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ WaveletAttention: FAIL - {e}")
    total_tests += 1
    
    # Test 3: Enhanced AutoCorrelation
    try:
        from layers.modular.attention.enhanced_autocorrelation import EnhancedAutoCorrelation
        enhanced_autocorr = EnhancedAutoCorrelation(d_model, n_heads)
        output, _ = enhanced_autocorr(queries, keys, values)
        assert output.shape == queries.shape
        print("✅ EnhancedAutoCorrelation: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ EnhancedAutoCorrelation: FAIL - {e}")
    total_tests += 1
    
    # Test 4: Bayesian Attention
    try:
        from layers.modular.attention.bayesian_attention import BayesianAttention
        bayesian_attn = BayesianAttention(d_model, n_heads)
        output, _ = bayesian_attn(queries, keys, values)
        assert output.shape == queries.shape
        print("✅ BayesianAttention: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ BayesianAttention: FAIL - {e}")
    total_tests += 1
    
    # Test 5: Meta Learning Adapter
    try:
        from layers.modular.attention.adaptive_components import MetaLearningAdapter
        meta_adapter = MetaLearningAdapter(d_model, n_heads, adaptation_steps=2)
        output, _ = meta_adapter(queries, keys, values)
        assert output.shape == queries.shape
        print("✅ MetaLearningAdapter: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ MetaLearningAdapter: FAIL - {e}")
    total_tests += 1
    
    # Test 6: Causal Convolution
    try:
        from layers.modular.attention.temporal_conv_attention import CausalConvolution
        causal_conv = CausalConvolution(d_model, n_heads, kernel_sizes=[3, 5])
        output, _ = causal_conv(queries, keys, values)
        assert output.shape == queries.shape
        print("✅ CausalConvolution: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ CausalConvolution: FAIL - {e}")
    total_tests += 1
    
    # Test Registry Integration
    try:
        from layers.modular.attention.registry import AttentionRegistry
        components = AttentionRegistry.list_components()
        phase2_components = [c for c in components if any(x in c for x in 
                           ['fourier', 'wavelet', 'enhanced', 'bayesian', 'meta', 'adaptive', 'causal', 'temporal', 'conv'])]
        assert len(phase2_components) >= 15, f"Expected at least 15 Phase 2 components, got {len(phase2_components)}"
        print(f"✅ Registry Integration: PASS ({len(phase2_components)} Phase 2 components)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Registry Integration: FAIL - {e}")
    total_tests += 1
    
    print("=" * 60)
    print(f"📊 Phase 2 Test Results: {tests_passed}/{total_tests} passed ({tests_passed/total_tests*100:.1f}%)")
    
    if tests_passed == total_tests:
        print("🎉 All Phase 2 attention components working correctly!")
        return True
    else:
        print("⚠️ Some Phase 2 components need attention")
        return False

if __name__ == "__main__":
    success = test_phase2_components()
    sys.exit(0 if success else 1)
