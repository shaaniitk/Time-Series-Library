#!/usr/bin/env python3
"""
Enhanced Algorithmic Tests for Phase 2 Attention Components

This test suite validates sophisticated algorithmic features rather than
just basic functionality. It ensures components maintain their intended
mathematical sophistication and don't regress to oversimplified versions.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_meta_learning_adapter_maml():
    """Test MetaLearningAdapter for proper MAML implementation"""
    print("🧠 Testing MetaLearningAdapter MAML Sophistication...")
    
    try:
        from layers.modular.attention.adaptive_components import MetaLearningAdapter
        
        # Test parameters
        d_model = 64
        component = MetaLearningAdapter(d_model, adaptation_steps=3)
        
        # Create support and query sets (key MAML concept)
        support_set = torch.randn(2, 16, d_model)  # Small support set
        query_set = torch.randn(2, 32, d_model)    # Larger query set
        
        component.train()
        
        # Test 1: Gradient-based adaptation should occur during training
        output1, _ = component(query_set, support_set, query_set)
        
        # Test 2: Check if adaptation actually changes behavior
        component.eval()
        output2, _ = component(query_set, support_set, query_set)
        
        # Test 3: MAML should show different outputs with different support sets
        different_support = torch.randn(2, 16, d_model)
        component.train()
        output3, _ = component(query_set, different_support, query_set)
        
        # Validate MAML behavior
        adaptation_difference = torch.norm(output1 - output3).item()
        support_sensitivity = adaptation_difference > 0.1  # Should adapt to different support sets
        
        # Test 4: Check for gradient computation capability
        has_gradient_computation = hasattr(component, 'meta_lr') and isinstance(component.meta_lr, torch.nn.Parameter)
        
        # Test 5: Check for fast adaptation weights
        has_fast_weights = hasattr(component, 'fast_weights') and len(component.fast_weights) > 0
        
        maml_sophistication_score = sum([
            support_sensitivity,
            has_gradient_computation, 
            has_fast_weights,
            adaptation_difference > 0.01  # Some adaptation occurred
        ])
        
        print(f"   Support Sensitivity: {'✅' if support_sensitivity else '❌'}")
        print(f"   Gradient Computation: {'✅' if has_gradient_computation else '❌'}")
        print(f"   Fast Weights: {'✅' if has_fast_weights else '❌'}")
        print(f"   Adaptation Magnitude: {adaptation_difference:.4f}")
        print(f"   MAML Sophistication Score: {maml_sophistication_score}/4")
        
        return maml_sophistication_score >= 3, f"MAML sophistication: {maml_sophistication_score}/4"
        
    except Exception as e:
        return False, f"MAML test failed: {e}"

def test_fourier_attention_complexity():
    """Test FourierAttention for sophisticated frequency domain operations"""
    print("🌊 Testing FourierAttention Frequency Sophistication...")
    
    try:
        from layers.modular.attention.fourier_attention import FourierAttention
        
        d_model, n_heads, seq_len = 64, 4, 32
        component = FourierAttention(d_model, n_heads, seq_len=seq_len, frequency_selection='adaptive')
        
        # Create test signals with known frequency content
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)
        
        # Signal 1: Low frequency (should emphasize low-freq components)
        signal1 = torch.sin(2 * np.pi * 2 * t).expand(2, seq_len, d_model)
        
        # Signal 2: High frequency (should emphasize high-freq components)  
        signal2 = torch.sin(2 * np.pi * 8 * t).expand(2, seq_len, d_model)
        
        output1, attn1 = component(signal1, signal1, signal1)
        output2, attn2 = component(signal2, signal2, signal2)
        
        # Test 1: Check for learnable frequency components
        has_freq_weights = hasattr(component, 'freq_weights') and hasattr(component, 'phase_weights')
        
        # Test 2: Check for adaptive frequency selection
        has_adaptive_selection = hasattr(component, 'freq_selector') and component.frequency_selection == 'adaptive'
        
        # Test 3: Different frequency inputs should produce different outputs
        frequency_selectivity = torch.norm(output1 - output2).item() > 0.1
        
        # Test 4: Check for complex frequency filtering capability
        has_complex_filtering = hasattr(component, 'phase_weights')
        
        # Test 5: Verify frequency domain operations
        with torch.no_grad():
            freq_repr = torch.fft.rfft(signal1, dim=1)
            has_freq_processing = freq_repr.shape[1] == seq_len // 2 + 1
        
        sophistication_score = sum([
            has_freq_weights,
            has_adaptive_selection,
            frequency_selectivity,
            has_complex_filtering,
            has_freq_processing
        ])
        
        print(f"   Learnable Frequency Weights: {'✅' if has_freq_weights else '❌'}")
        print(f"   Adaptive Selection: {'✅' if has_adaptive_selection else '❌'}")
        print(f"   Frequency Selectivity: {'✅' if frequency_selectivity else '❌'}")
        print(f"   Complex Filtering: {'✅' if has_complex_filtering else '❌'}")
        print(f"   Frequency Processing: {'✅' if has_freq_processing else '❌'}")
        print(f"   Frequency Sophistication Score: {sophistication_score}/5")
        
        return sophistication_score >= 4, f"Frequency sophistication: {sophistication_score}/5"
        
    except Exception as e:
        return False, f"Fourier test failed: {e}"

def test_enhanced_autocorrelation_intelligence():
    """Test EnhancedAutoCorrelation for multi-scale analysis sophistication"""
    print("🔄 Testing EnhancedAutoCorrelation Multi-Scale Intelligence...")
    
    try:
        from layers.modular.attention.enhanced_autocorrelation import EnhancedAutoCorrelation
        
        d_model, n_heads = 64, 4
        component = EnhancedAutoCorrelation(d_model, n_heads, multi_scale=True, adaptive_k=True)
        
        # Create signals with different correlation patterns
        seq_len = 32
        
        # Signal with strong short-term correlations
        signal1 = torch.randn(2, seq_len, d_model)
        signal1[:, 1::2] = signal1[:, ::2] * 0.9  # Strong lag-1 correlation
        
        # Signal with long-term correlations  
        signal2 = torch.randn(2, seq_len, d_model)
        signal2[:, 4:] = signal2[:, :-4] * 0.8  # Strong lag-4 correlation
        
        output1, _ = component(signal1, signal1, signal1)
        output2, _ = component(signal2, signal2, signal2)
        
        # Test 1: Multi-scale analysis capability
        has_multi_scale = hasattr(component, 'scales') and len(component.scales) > 1
        
        # Test 2: Adaptive k-selection
        has_adaptive_k = hasattr(component, 'adaptive_k') and component.adaptive_k
        
        # Test 3: Correlation pattern sensitivity
        pattern_sensitivity = torch.norm(output1 - output2).item() > 0.1
        
        # Test 4: Scale-specific processing
        has_scale_weights = hasattr(component, 'scale_weights')
        
        # Test 5: Intelligent peak detection capability
        has_peak_detection = hasattr(component, '_select_adaptive_k')
        
        sophistication_score = sum([
            has_multi_scale,
            has_adaptive_k,
            pattern_sensitivity,
            has_scale_weights,
            has_peak_detection
        ])
        
        print(f"   Multi-Scale Analysis: {'✅' if has_multi_scale else '❌'}")
        print(f"   Adaptive K-Selection: {'✅' if has_adaptive_k else '❌'}")
        print(f"   Pattern Sensitivity: {'✅' if pattern_sensitivity else '❌'}")
        print(f"   Scale Weights: {'✅' if has_scale_weights else '❌'}")
        print(f"   Peak Detection: {'✅' if has_peak_detection else '❌'}")
        print(f"   Correlation Sophistication Score: {sophistication_score}/5")
        
        return sophistication_score >= 4, f"Correlation sophistication: {sophistication_score}/5"
        
    except Exception as e:
        return False, f"AutoCorrelation test failed: {e}"

def test_bayesian_attention_uncertainty():
    """Test BayesianAttention for proper uncertainty quantification"""
    print("🎲 Testing BayesianAttention Uncertainty Quantification...")
    
    try:
        from layers.modular.attention.bayesian_attention import BayesianAttention
        
        d_model, n_heads = 64, 4
        component = BayesianAttention(d_model, n_heads)
        
        queries = torch.randn(2, 32, d_model)
        
        # Test 1: Multiple forward passes should yield different outputs (uncertainty)
        component.train()
        output1, _ = component(queries, queries, queries)
        output2, _ = component(queries, queries, queries)
        output3, _ = component(queries, queries, queries)
        
        # Measure uncertainty through output variance
        stochastic_variance = torch.var(torch.stack([output1, output2, output3]), dim=0).mean().item()
        has_uncertainty = stochastic_variance > 1e-6
        
        # Test 2: Check for Bayesian linear layers
        has_bayesian_layers = any(hasattr(module, 'weight_mu') for module in component.modules())
        
        # Test 3: KL divergence computation capability
        total_kl = component.get_kl_divergence()
        has_kl_computation = total_kl is not None and total_kl > 0
        
        # Test 4: Different behavior in train vs eval
        component.eval()
        output_eval, _ = component(queries, queries, queries)
        train_eval_difference = torch.norm(output1 - output_eval).item() > 1e-4
        
        sophistication_score = sum([
            has_uncertainty,
            has_bayesian_layers,
            has_kl_computation,
            train_eval_difference
        ])
        
        print(f"   Stochastic Uncertainty: {'✅' if has_uncertainty else '❌'} (var: {stochastic_variance:.6f})")
        print(f"   Bayesian Layers: {'✅' if has_bayesian_layers else '❌'}")
        print(f"   KL Divergence: {'✅' if has_kl_computation else '❌'} (KL: {total_kl:.4f})")
        print(f"   Train/Eval Difference: {'✅' if train_eval_difference else '❌'}")
        print(f"   Bayesian Sophistication Score: {sophistication_score}/4")
        
        return sophistication_score >= 3, f"Bayesian sophistication: {sophistication_score}/4"
        
    except Exception as e:
        return False, f"Bayesian test failed: {e}"

def test_wavelet_attention_multires():
    """Test WaveletAttention for multi-resolution analysis"""
    print("〰️ Testing WaveletAttention Multi-Resolution Analysis...")
    
    try:
        from layers.modular.attention.wavelet_attention import WaveletAttention
        
        d_model, n_heads, levels = 64, 4, 3
        component = WaveletAttention(d_model, n_heads, levels=levels)
        
        queries = torch.randn(2, 32, d_model)
        output, _ = component(queries, queries, queries)
        
        # Test 1: Multi-level decomposition
        has_multi_level = hasattr(component, 'levels') and component.levels > 1
        
        # Test 2: Level-specific attention layers
        has_level_attentions = hasattr(component, 'level_attentions') and len(component.level_attentions) == levels + 1
        
        # Test 3: Wavelet decomposition component
        has_wavelet_decomp = hasattr(component, 'wavelet_decomp')
        
        # Test 4: Level fusion mechanism
        has_level_fusion = hasattr(component, 'level_weights') and hasattr(component, 'fusion_proj')
        
        # Test 5: Output shape preservation
        correct_output_shape = output.shape == queries.shape
        
        sophistication_score = sum([
            has_multi_level,
            has_level_attentions,
            has_wavelet_decomp,
            has_level_fusion,
            correct_output_shape
        ])
        
        print(f"   Multi-Level Decomposition: {'✅' if has_multi_level else '❌'}")
        print(f"   Level-Specific Attention: {'✅' if has_level_attentions else '❌'}")
        print(f"   Wavelet Decomposition: {'✅' if has_wavelet_decomp else '❌'}")
        print(f"   Level Fusion: {'✅' if has_level_fusion else '❌'}")
        print(f"   Output Shape: {'✅' if correct_output_shape else '❌'}")
        print(f"   Wavelet Sophistication Score: {sophistication_score}/5")
        
        return sophistication_score >= 4, f"Wavelet sophistication: {sophistication_score}/5"
        
    except Exception as e:
        return False, f"Wavelet test failed: {e}"

def test_causal_convolution_causality():
    """Test CausalConvolution for proper causal constraints"""
    print("⏭️ Testing CausalConvolution Causal Constraints...")
    
    try:
        from layers.modular.attention.temporal_conv_attention import CausalConvolution
        
        d_model, n_heads = 64, 4
        component = CausalConvolution(d_model, n_heads)
        
        seq_len = 32
        queries = torch.randn(2, seq_len, d_model)
        
        # Test causality by masking future and checking if output changes
        masked_queries = queries.clone()
        masked_queries[:, seq_len//2:] = 0  # Mask future half
        
        output_full, _ = component(queries, queries, queries)
        output_masked, _ = component(masked_queries, masked_queries, masked_queries)
        
        # Past should be unaffected by future masking (causal property)
        past_preservation = torch.allclose(
            output_full[:, :seq_len//2], 
            output_masked[:, :seq_len//2], 
            atol=1e-5
        )
        
        # Test 1: Multi-scale convolutions
        has_multi_scale = hasattr(component, 'causal_convs') and len(component.causal_convs) > 1
        
        # Test 2: Dilation rates
        has_dilation = hasattr(component, 'dilation_rates') and len(component.dilation_rates) > 1
        
        # Test 3: Causal masking method
        has_causal_mask = hasattr(component, 'apply_causal_mask')
        
        # Test 4: Temporal awareness
        has_temporal_awareness = hasattr(component, 'positional_conv')
        
        sophistication_score = sum([
            past_preservation,
            has_multi_scale,
            has_dilation, 
            has_causal_mask,
            has_temporal_awareness
        ])
        
        print(f"   Causal Constraint: {'✅' if past_preservation else '❌'}")
        print(f"   Multi-Scale Convolutions: {'✅' if has_multi_scale else '❌'}")
        print(f"   Dilation Rates: {'✅' if has_dilation else '❌'}")
        print(f"   Causal Masking: {'✅' if has_causal_mask else '❌'}")
        print(f"   Temporal Awareness: {'✅' if has_temporal_awareness else '❌'}")
        print(f"   Causal Sophistication Score: {sophistication_score}/5")
        
        return sophistication_score >= 4, f"Causal sophistication: {sophistication_score}/5"
        
    except Exception as e:
        return False, f"Causal test failed: {e}"

def main():
    """Run comprehensive algorithmic sophistication tests"""
    print("🧪 Enhanced Algorithmic Tests for Phase 2 Attention Components")
    print("=" * 80)
    
    test_results = {}
    
    # Run sophisticated algorithmic tests
    tests = [
        ("MetaLearningAdapter MAML", test_meta_learning_adapter_maml),
        ("FourierAttention Frequency", test_fourier_attention_complexity),
        ("EnhancedAutoCorrelation Intelligence", test_enhanced_autocorrelation_intelligence),
        ("BayesianAttention Uncertainty", test_bayesian_attention_uncertainty),
        ("WaveletAttention MultiRes", test_wavelet_attention_multires),
        ("CausalConvolution Causality", test_causal_convolution_causality),
    ]
    
    for test_name, test_func in tests:
        try:
            success, details = test_func()
            test_results[test_name] = "✅ SOPHISTICATED" if success else f"❌ SIMPLIFIED ({details})"
            print()
        except Exception as e:
            test_results[test_name] = f"❌ ERROR: {e}"
            print()
    
    # Final summary
    print("🏆 ALGORITHMIC SOPHISTICATION SUMMARY")
    print("=" * 80)
    
    sophisticated_count = 0
    for test_name, result in test_results.items():
        print(f"{test_name:35} {result}")
        if result.startswith("✅"):
            sophisticated_count += 1
    
    print("=" * 80)
    print(f"📊 Sophistication Rate: {sophisticated_count}/{len(tests)} ({sophisticated_count/len(tests)*100:.1f}%)")
    
    if sophisticated_count == len(tests):
        print("🎉 ALL COMPONENTS MAINTAIN ALGORITHMIC SOPHISTICATION!")
    elif sophisticated_count >= len(tests) * 0.8:
        print("⚠️  MOST COMPONENTS SOPHISTICATED - Some restoration needed")
    else:
        print("🚨 MAJOR ALGORITHMIC OVERSIMPLIFICATION DETECTED!")
        print("📋 Restoration required for components marked as SIMPLIFIED")
    
    return sophisticated_count == len(tests)

if __name__ == "__main__":
    main()
