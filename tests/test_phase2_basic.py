#!/usr/bin/env python3
"""
Test Phase 2 Attention Components - Basic Functionality Check
"""

import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_attention_functionality():
    """Test that Phase 2 components can be instantiated and run basic forward pass"""
    print("🔍 Testing Basic Phase 2 Attention Functionality")
    print("=" * 60)
    
    # Test parameters
    batch_size, seq_len, d_model, n_heads = 2, 32, 64, 4
    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shapes: Q={queries.shape}, K={keys.shape}, V={values.shape}")
    print()
    
    results = {}
    
    # Test 1: WaveletAttention (this one was working)
    try:
        from layers.modular.attention.wavelet_attention import WaveletAttention
        component = WaveletAttention(d_model, n_heads, levels=3)
        output, _ = component(queries, keys, values)
        assert output.shape == queries.shape
        results['WaveletAttention'] = "✅ PASS"
    except Exception as e:
        results['WaveletAttention'] = f"❌ FAIL: {e}"
    
    # Test 2: BayesianAttention (this one was working)
    try:
        from layers.modular.attention.bayesian_attention import BayesianAttention
        component = BayesianAttention(d_model, n_heads)
        output, _ = component(queries, keys, values)
        assert output.shape == queries.shape
        results['BayesianAttention'] = "✅ PASS"
    except Exception as e:
        results['BayesianAttention'] = f"❌ FAIL: {e}"
    
    # Test 3: CausalConvolution (this one was working)
    try:
        from layers.modular.attention.temporal_conv_attention import CausalConvolution
        component = CausalConvolution(d_model, n_heads)
        output, _ = component(queries, keys, values)
        assert output.shape == queries.shape
        results['CausalConvolution'] = "✅ PASS"
    except Exception as e:
        results['CausalConvolution'] = f"❌ FAIL: {e}"
    
    # Test 4: FourierAttention (needs fixing)
    try:
        from layers.modular.attention.fourier_attention import FourierAttention
        component = FourierAttention(d_model, n_heads, seq_len=seq_len)
        output, _ = component(queries, keys, values)
        assert output.shape == queries.shape
        results['FourierAttention'] = "✅ PASS"
    except Exception as e:
        results['FourierAttention'] = f"❌ FAIL: {str(e)[:100]}..."
    
    # Test 5: Enhanced AutoCorrelation (restored)
    try:
        from layers.modular.attention.enhanced_autocorrelation import EnhancedAutoCorrelation
        component = EnhancedAutoCorrelation(d_model, n_heads)
        output, _ = component(queries, keys, values)
        assert output.shape == queries.shape
        results['EnhancedAutoCorrelation'] = "✅ PASS"
    except Exception as e:
        results['EnhancedAutoCorrelation'] = f"❌ FAIL: {str(e)[:100]}..."
    
    # Test 6: MetaLearningAdapter (needs proper MAML restoration)
    try:
        from layers.modular.attention.adaptive_components import MetaLearningAdapter
        component = MetaLearningAdapter(d_model, n_heads)
        output, _ = component(queries, keys, values)
        assert output.shape == queries.shape
        results['MetaLearningAdapter'] = "✅ PASS"
    except Exception as e:
        results['MetaLearningAdapter'] = f"❌ FAIL: {str(e)[:100]}..."
    
    print("📊 TEST RESULTS:")
    print("=" * 60)
    passed = 0
    total = len(results)
    
    for component, result in results.items():
        print(f"{component:25} {result}")
        if "✅ PASS" in result:
            passed += 1
    
    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} components passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All Phase 2 attention components working!")
        print("📝 Ready to proceed to Phase 3")
    else:
        print("⚠️  Some components need algorithmic restoration")
        print("📋 Refer to MODULAR_IMPLEMENTATION_GUIDE.md for detailed fix instructions")
    
    return passed == total

if __name__ == "__main__":
    success = test_basic_attention_functionality()
    sys.exit(0 if success else 1)
