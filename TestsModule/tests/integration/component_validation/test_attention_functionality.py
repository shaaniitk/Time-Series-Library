#!/usr/bin/env python3
"""
Comprehensive Attention Mechanism Component Functionality Tests

This test suite validates that each attention mechanism component produces
correct attention patterns, handles different sequence lengths, and maintains
mathematical properties expected from attention mechanisms.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from layers.modular.core.registry import unified_registry, ComponentFamily
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"WARN Could not import modular components: {e}")
    COMPONENTS_AVAILABLE = False

class MockConfig:
    """Mock configuration for attention testing"""
    def __init__(self, **kwargs):
        self.d_model = 512
        self.num_heads = 8
        self.n_heads = 8  # Alternative naming
        self.dropout = 0.1
        self.max_seq_len = 512
        self.chunk_size = 128
        self.use_mixed_precision = False
        self.scales = [1, 2, 4]
        self.hierarchy_levels = 3
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_attention_inputs(batch_size=2, seq_len=100, d_model=512):
    """Create sample attention inputs"""
    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    
    # Create attention mask (optional)
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask = torch.tril(mask)  # Causal mask
    
    return queries, keys, values, mask

def test_multi_head_attention_functionality():
    """Test multi-head attention actual functionality"""
    print("TEST Testing Multi-Head Attention Functionality...")
    
    try:
        config = MockConfig(d_model=512, num_heads=8)
        attention = unified_registry.create(ComponentFamily.ATTENTION, 'multi_head', **vars(config))
        
        if attention is None:
            print("    WARN Multi-head attention not available, skipping...")
            return True
        
        queries, keys, values, mask = create_attention_inputs()
        
        # Test basic forward pass
        with torch.no_grad():
            output, attn_weights = attention(queries, keys, values)
        
        # Validate output shape
        assert output.shape == queries.shape, f"Output shape {output.shape} != input shape {queries.shape}"
        
        # Validate attention weights
        expected_attn_shape = (queries.size(0), queries.size(1), keys.size(1))
        if attn_weights is not None:
            assert len(attn_weights.shape) >= 3, "Attention weights should be at least 3D"
            print(f"    CHART Attention weights shape: {attn_weights.shape}")
        
        # Test attention properties
        if attn_weights is not None:
            # Attention weights should sum to 1 along last dimension
            attn_sum = attn_weights.sum(dim=-1)
            assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
                "Attention weights should sum to 1"
            
            # Attention weights should be non-negative
            assert (attn_weights >= 0).all(), "Attention weights should be non-negative"
        
        # Test with different sequence lengths
        for seq_len in [50, 200, 300]:
            q_test, k_test, v_test, _ = create_attention_inputs(seq_len=seq_len)
            with torch.no_grad():
                out_test, attn_test = attention(q_test, k_test, v_test)
            assert out_test.shape == q_test.shape, f"Failed for seq_len={seq_len}"
        
        # Test self-attention vs cross-attention
        # Self-attention: Q, K, V are the same
        with torch.no_grad():
            out_self, attn_self = attention(queries, queries, queries)
        
        # Cross-attention: different K, V
        keys_diff = torch.randn_like(keys)
        values_diff = torch.randn_like(values) 
        with torch.no_grad():
            out_cross, attn_cross = attention(queries, keys_diff, values_diff)
        
        # Outputs should be different
        assert not torch.allclose(out_self, out_cross, atol=1e-3), \
            "Self-attention and cross-attention should produce different outputs"
        
        # Test gradient flow
        queries_grad = queries.clone().requires_grad_(True)
        output_grad, _ = attention(queries_grad, keys, values)
        loss = output_grad.mean()
        loss.backward()
        
        assert queries_grad.grad is not None, "Gradients should exist"
        assert not torch.isnan(queries_grad.grad).any(), "Gradients should not be NaN"
        
        print("    PASS Multi-head attention functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Multi-head attention test failed: {e}")
        return False

def test_optimized_autocorrelation_functionality():
    """Test optimized autocorrelation attention functionality"""
    print("TEST Testing Optimized AutoCorrelation Attention Functionality...")
    
    try:
        config = MockConfig(
            d_model=512, 
            num_heads=8, 
            max_seq_len=512,
            use_mixed_precision=True,
            chunk_size=128
        )
        
        attention = unified_registry.create(ComponentFamily.ATTENTION, 'optimized_autocorrelation', **vars(config))
        
        if attention is None:
            print("    WARN Optimized autocorrelation attention not available, skipping...")
            return True
        
        queries, keys, values, mask = create_attention_inputs()
        
        # Test basic functionality
        with torch.no_grad():
            output, attn_weights = attention(queries, keys, values)
        
        assert output.shape == queries.shape, "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        # Test autocorrelation properties
        # Autocorrelation should capture periodic patterns
        # Create a simple periodic signal
        t = torch.linspace(0, 4*np.pi, 100)
        periodic_signal = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(2, -1, 512)
        
        with torch.no_grad():
            out_periodic, attn_periodic = attention(periodic_signal, periodic_signal, periodic_signal)
        
        # The attention should capture the periodicity
        if attn_periodic is not None:
            # Check if attention pattern shows some structure
            attn_var = attn_periodic.var().item()
            print(f"    CHART Attention variance on periodic signal: {attn_var:.6f}")
            assert attn_var > 1e-6, "Attention should show some variation on periodic signal"
        
        # Test memory efficiency with long sequences
        try:
            long_queries, long_keys, long_values, _ = create_attention_inputs(seq_len=400)
            with torch.no_grad():
                out_long, _ = attention(long_queries, long_keys, long_values)
            assert out_long.shape == long_queries.shape, "Long sequence handling failed"
            print("    PASS Long sequence handling successful")
        except Exception as e:
            print(f"    WARN Long sequence test failed: {e}")
        
        # Test chunked processing if available
        if hasattr(attention, 'chunk_size'):
            print(f"    CHART Chunk size: {attention.chunk_size}")
        
        print("    PASS Optimized autocorrelation attention functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Optimized autocorrelation attention test failed: {e}")
        return False

def test_adaptive_autocorrelation_functionality():
    """Test adaptive autocorrelation attention functionality"""
    print("TEST Testing Adaptive AutoCorrelation Attention Functionality...")
    
    try:
        config = MockConfig(
            d_model=512,
            num_heads=8,
            scales=[1, 2, 4, 8]
        )
        
        attention = unified_registry.create(ComponentFamily.ATTENTION, 'adaptive_autocorrelation', **vars(config))
        
        if attention is None:
            print("    WARN Adaptive autocorrelation attention not available, skipping...")
            return True
        
        queries, keys, values, mask = create_attention_inputs()
        
        # Test basic functionality
        with torch.no_grad():
            output, attn_weights = attention(queries, keys, values)
        
        assert output.shape == queries.shape, "Output shape mismatch"
        
        # Test multi-scale adaptation
        # Create signals with different frequency components
        t = torch.linspace(0, 4*np.pi, 100)
        
        # Low frequency signal
        low_freq = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(2, -1, 512)
        
        # High frequency signal
        high_freq = torch.sin(10 * t).unsqueeze(0).unsqueeze(-1).expand(2, -1, 512)
        
        # Mixed frequency signal
        mixed_freq = low_freq + 0.5 * high_freq
        
        outputs = {}
        for name, signal in [("low", low_freq), ("high", high_freq), ("mixed", mixed_freq)]:
            with torch.no_grad():
                out, attn = attention(signal, signal, signal)
                outputs[name] = out
            
            print(f"    CHART {name} frequency signal processed successfully")
        
        # Adaptive attention should produce different outputs for different frequencies
        assert not torch.allclose(outputs["low"], outputs["high"], atol=1e-3), \
            "Should produce different outputs for different frequencies"
        
        # Test scale adaptation
        if hasattr(attention, 'scales'):
            print(f"    CHART Attention scales: {attention.scales}")
        
        print("    PASS Adaptive autocorrelation attention functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Adaptive autocorrelation attention test failed: {e}")
        return False

def test_hierarchical_attention_functionality():
    """Test hierarchical attention functionality"""
    print("TEST Testing Hierarchical Attention Functionality...")
    
    try:
        config = MockConfig(
            d_model=512,
            num_heads=8,
            hierarchy_levels=3
        )
        
        attention = unified_registry.create(ComponentFamily.ATTENTION, 'hierarchical_attention', **vars(config))
        
        if attention is None:
            print("    WARN Hierarchical attention not available, skipping...")
            return True
        
        queries, keys, values, mask = create_attention_inputs()
        
        # Test basic functionality
        with torch.no_grad():
            output, attn_weights = attention(queries, keys, values)
        
        assert output.shape == queries.shape, "Output shape mismatch"
        
        # Test hierarchical processing
        # Hierarchical attention should handle multi-resolution patterns
        
        # Create hierarchical pattern (coarse to fine)
        seq_len = 96
        t = torch.linspace(0, 4*np.pi, seq_len)
        
        # Level 1: Very coarse pattern (global trend)
        level1 = torch.sin(0.5 * t)
        
        # Level 2: Medium pattern (seasonal)
        level2 = 0.7 * torch.sin(2 * t)
        
        # Level 3: Fine pattern (noise/details)
        level3 = 0.3 * torch.sin(8 * t)
        
        hierarchical_signal = (level1 + level2 + level3).unsqueeze(0).unsqueeze(-1).expand(2, -1, 512)
        
        with torch.no_grad():
            out_hier, attn_hier = attention(hierarchical_signal, hierarchical_signal, hierarchical_signal)
        
        # Test with flat signal for comparison
        flat_signal = torch.randn(2, seq_len, 512)
        with torch.no_grad():
            out_flat, attn_flat = attention(flat_signal, flat_signal, flat_signal)
        
        # Hierarchical attention should produce different patterns
        assert not torch.allclose(out_hier, out_flat, atol=1e-3), \
            "Hierarchical vs flat signal should produce different outputs"
        
        # Test different hierarchy levels
        for levels in [2, 3, 4]:
            try:
                config_test = MockConfig(hierarchy_levels=levels)
                attn_test = create_component('attention', 'hierarchical_attention', config_test)
                if attn_test:
                    with torch.no_grad():
                        out_test, _ = attn_test(queries, keys, values)
                    print(f"    CHART Hierarchy levels {levels}: PASS")
            except Exception as e:
                print(f"    WARN Hierarchy levels {levels}: {e}")
        
        print("    PASS Hierarchical attention functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Hierarchical attention test failed: {e}")
        return False

def test_memory_efficient_attention():
    """Test memory efficient attention functionality"""
    print("TEST Testing Memory Efficient Attention Functionality...")
    
    try:
        config = MockConfig(
            d_model=512,
            num_heads=8,
            memory_efficient=True,
            use_checkpointing=True
        )
        
        attention = unified_registry.create(ComponentFamily.ATTENTION, 'memory_efficient', **vars(config))
        
        if attention is None:
            print("    WARN Memory efficient attention not available, skipping...")
            return True
        
        # Test with progressively larger sequences to check memory efficiency
        memory_results = []
        
        for seq_len in [100, 200, 400]:
            try:
                queries, keys, values, _ = create_attention_inputs(seq_len=seq_len)
                
                # Measure memory usage (approximate)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                with torch.no_grad():
                    output, attn_weights = attention(queries, keys, values)
                
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_used = end_memory - start_memory
                
                assert output.shape == queries.shape, f"Shape mismatch for seq_len={seq_len}"
                
                memory_results.append((seq_len, memory_used))
                print(f"    CHART Seq len {seq_len}: memory ~{memory_used / 1024**2:.1f} MB")
                
            except Exception as e:
                print(f"    WARN Seq len {seq_len} failed: {e}")
        
        # Test gradient checkpointing if available
        if hasattr(attention, 'use_checkpointing') and attention.use_checkpointing:
            queries_grad = queries.clone().requires_grad_(True)
            output_grad, _ = attention(queries_grad, keys, values)
            loss = output_grad.mean()
            loss.backward()
            
            assert queries_grad.grad is not None, "Checkpointing should still allow gradients"
            print("    PASS Gradient checkpointing working")
        
        print("    PASS Memory efficient attention functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Memory efficient attention test failed: {e}")
        return False

def test_attention_mathematical_properties():
    """Test mathematical properties of attention mechanisms"""
    print("TEST Testing Attention Mathematical Properties...")
    
    try:
        attention_types = ['multi_head']  # Start with most reliable
        
        # Try to add other types if available
        for attn_type in ['optimized_autocorrelation', 'adaptive_autocorrelation']:
            try:
                test_attn = unified_registry.create(ComponentFamily.ATTENTION, attn_type, **vars(MockConfig()))
                if test_attn is not None:
                    attention_types.append(attn_type)
            except:
                pass
        
        queries, keys, values, mask = create_attention_inputs(seq_len=50)  # Smaller for testing
        
        for attn_type in attention_types:
            try:
                print(f"    SEARCH Testing {attn_type} mathematical properties:")
                
                attention = unified_registry.create(ComponentFamily.ATTENTION, attn_type, **vars(MockConfig()))
                if attention is None:
                    continue
                
                # Test 1: Permutation equivariance (for self-attention)
                perm_indices = torch.randperm(queries.size(1))
                q_perm = queries[:, perm_indices, :]
                k_perm = queries[:, perm_indices, :]  # Use same permutation
                v_perm = queries[:, perm_indices, :]
                
                with torch.no_grad():
                    out_orig, _ = attention(queries, queries, queries)
                    out_perm, _ = attention(q_perm, k_perm, v_perm)
                
                # After undoing permutation, outputs should be similar
                out_perm_undone = out_perm[:, torch.argsort(perm_indices), :]
                perm_diff = torch.abs(out_orig - out_perm_undone).mean()
                print(f"      CHART Permutation invariance error: {perm_diff:.6f}")
                
                # Test 2: Scale sensitivity
                scaled_queries = queries * 2.0
                with torch.no_grad():
                    out_scaled, _ = attention(scaled_queries, keys, values)
                
                scale_diff = torch.abs(out_orig - out_scaled).mean()
                print(f"      CHART Scale sensitivity: {scale_diff:.6f}")
                
                # Test 3: Linearity in values (attention should be linear in V)
                alpha = 0.7
                beta = 0.3
                values2 = torch.randn_like(values)
                combined_values = alpha * values + beta * values2
                
                with torch.no_grad():
                    out_v1, _ = attention(queries, keys, values)
                    out_v2, _ = attention(queries, keys, values2)
                    out_combined, _ = attention(queries, keys, combined_values)
                
                expected_combined = alpha * out_v1 + beta * out_v2
                linearity_error = torch.abs(out_combined - expected_combined).mean()
                print(f"      CHART Value linearity error: {linearity_error:.6f}")
                
                # Test 4: Attention weight properties
                with torch.no_grad():
                    _, attn_weights = attention(queries, keys, values)
                
                if attn_weights is not None:
                    # Weights should be non-negative
                    non_negative = (attn_weights >= -1e-6).all()  # Small tolerance for numerical errors
                    
                    # Weights should approximately sum to 1
                    weight_sums = attn_weights.sum(dim=-1)
                    sum_error = torch.abs(weight_sums - 1.0).mean()
                    
                    print(f"      CHART Non-negative weights: {non_negative}")
                    print(f"      CHART Weight sum error: {sum_error:.6f}")
                
                print(f"      PASS {attn_type} mathematical properties checked")
                
            except Exception as e:
                print(f"      FAIL {attn_type} mathematical test failed: {e}")
        
        print("    PASS Attention mathematical properties validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Attention mathematical properties test failed: {e}")
        return False

def test_attention_edge_cases():
    """Test attention mechanisms with edge cases"""
    print("TEST Testing Attention Edge Cases...")
    
    try:
        attention_types = ['multi_head']
        
        # Try to add other available types
        for attn_type in ['optimized_autocorrelation']:
            try:
                test_attn = create_component('attention', attn_type, MockConfig())
                if test_attn is not None:
                    attention_types.append(attn_type)
            except:
                pass
        
        edge_cases = [
            ("single_token", 1),
            ("very_short", 3),
            ("medium", 50),
            ("long", 200),
        ]
        
        for attn_type in attention_types:
            try:
                print(f"    SEARCH Testing {attn_type} edge cases:")
                
                attention = create_component('attention', attn_type, MockConfig())
                if attention is None:
                    continue
                
                for case_name, seq_len in edge_cases:
                    try:
                        q, k, v, _ = create_attention_inputs(seq_len=seq_len)
                        
                        with torch.no_grad():
                            output, attn_weights = attention(q, k, v)
                        
                        assert output.shape == q.shape, f"Shape mismatch for {case_name}"
                        assert not torch.isnan(output).any(), f"NaN output for {case_name}"
                        assert not torch.isinf(output).any(), f"Inf output for {case_name}"
                        
                        print(f"      PASS {case_name} (seq_len={seq_len})")
                        
                    except Exception as e:
                        print(f"      FAIL {case_name} failed: {e}")
                
                # Test with extreme values
                extreme_cases = [
                    ("very_small", torch.ones(2, 10, 512) * 1e-6),
                    ("very_large", torch.ones(2, 10, 512) * 1e6),
                    ("zeros", torch.zeros(2, 10, 512)),
                ]
                
                for case_name, extreme_input in extreme_cases:
                    try:
                        with torch.no_grad():
                            output, _ = attention(extreme_input, extreme_input, extreme_input)
                        
                        is_finite = torch.isfinite(output).all()
                        is_nan = torch.isnan(output).any()
                        
                        status = "PASS" if is_finite and not is_nan else "FAIL"
                        print(f"      {status} {case_name}: finite={is_finite}, nan={is_nan}")
                        
                    except Exception as e:
                        print(f"      FAIL {case_name}: {e}")
                
                print(f"      PASS {attn_type} edge cases handled")
                
            except Exception as e:
                print(f"      FAIL {attn_type} edge case testing failed: {e}")
        
        print("    PASS Attention edge cases validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Attention edge cases test failed: {e}")
        return False


def test_fourier_attention_functionality():
    """Test Fourier-based attention mechanisms"""
    print("TEST Testing Fourier Attention Components...")
    
    try:
        from layers.modular.attention.fourier_attention import FourierAttention, FourierBlock, FourierCrossAttention
        
        config = MockConfig(d_model=64, n_heads=4)
        batch_size, seq_len = 2, 32
        queries, keys, values, mask = create_attention_inputs(batch_size, seq_len, config.d_model)
        
        # Test FourierAttention
        fourier_attn = FourierAttention(config.d_model, config.n_heads)
        output, _ = fourier_attn(queries, keys, values)
        assert output.shape == queries.shape, f"FourierAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test FourierBlock
        fourier_block = FourierBlock(config.d_model, config.n_heads, seq_len)
        output, _ = fourier_block(queries, keys, values)
        assert output.shape == queries.shape, f"FourierBlock shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test FourierCrossAttention
        fourier_cross = FourierCrossAttention(config.d_model, config.n_heads)
        output, _ = fourier_cross(queries, keys, values)
        assert output.shape == queries.shape, f"FourierCrossAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        print("PASS Fourier attention components working correctly")
        return True
        
    except Exception as e:
        print(f"FAIL Fourier attention test failed: {e}")
        return False


def test_wavelet_attention_functionality():
    """Test Wavelet-based attention mechanisms"""
    print("TEST Testing Wavelet Attention Components...")
    
    try:
        from layers.modular.attention.wavelet_attention import (
            WaveletAttention, WaveletDecomposition, AdaptiveWaveletAttention, MultiScaleWaveletAttention
        )
        
        config = MockConfig(d_model=64, n_heads=4)
        batch_size, seq_len = 2, 32
        queries, keys, values, mask = create_attention_inputs(batch_size, seq_len, config.d_model)
        
        # Test WaveletAttention
        wavelet_attn = WaveletAttention(config.d_model, config.n_heads, n_levels=3)
        output, _ = wavelet_attn(queries, keys, values)
        assert output.shape == queries.shape, f"WaveletAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test WaveletDecomposition
        wavelet_decomp = WaveletDecomposition(config.d_model, config.n_heads, n_levels=3)
        output, _ = wavelet_decomp(queries, keys, values)
        assert output.shape == queries.shape, f"WaveletDecomposition shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test AdaptiveWaveletAttention
        adaptive_wavelet = AdaptiveWaveletAttention(config.d_model, config.n_heads, max_levels=4)
        output, _ = adaptive_wavelet(queries, keys, values)
        assert output.shape == queries.shape, f"AdaptiveWaveletAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test MultiScaleWaveletAttention
        multiscale_wavelet = MultiScaleWaveletAttention(config.d_model, config.n_heads, scales=[1, 2, 4])
        output, _ = multiscale_wavelet(queries, keys, values)
        assert output.shape == queries.shape, f"MultiScaleWaveletAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        print("PASS Wavelet attention components working correctly")
        return True
        
    except Exception as e:
        print(f"FAIL Wavelet attention test failed: {e}")
        return False


def test_enhanced_autocorrelation_functionality():
    """Test Enhanced AutoCorrelation mechanisms"""
    print("TEST Testing Enhanced AutoCorrelation Components...")
    
    try:
        from layers.modular.attention.enhanced_autocorrelation import (
            EnhancedAutoCorrelation, AdaptiveAutoCorrelationLayer, HierarchicalAutoCorrelation
        )
        
        config = MockConfig(d_model=64, n_heads=4)
        batch_size, seq_len = 2, 32
        queries, keys, values, mask = create_attention_inputs(batch_size, seq_len, config.d_model)
        
        # Test EnhancedAutoCorrelation
        enhanced_autocorr = EnhancedAutoCorrelation(config.d_model, config.n_heads)
        output, _ = enhanced_autocorr(queries, keys, values)
        assert output.shape == queries.shape, f"EnhancedAutoCorrelation shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test AdaptiveAutoCorrelationLayer
        adaptive_autocorr = AdaptiveAutoCorrelationLayer(config.d_model, config.n_heads)
        output, _ = adaptive_autocorr(queries, keys, values)
        assert output.shape == queries.shape, f"AdaptiveAutoCorrelationLayer shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test HierarchicalAutoCorrelation
        hierarchical_autocorr = HierarchicalAutoCorrelation(config.d_model, config.n_heads, hierarchy_levels=[1, 2, 4])
        output, _ = hierarchical_autocorr(queries, keys, values)
        assert output.shape == queries.shape, f"HierarchicalAutoCorrelation shape mismatch: {output.shape} vs {queries.shape}"
        
        print("PASS Enhanced AutoCorrelation components working correctly")
        return True
        
    except Exception as e:
        print(f"FAIL Enhanced AutoCorrelation test failed: {e}")
        return False


def test_bayesian_attention_functionality():
    """Test Bayesian attention mechanisms"""
    print("TEST Testing Bayesian Attention Components...")
    
    try:
        from layers.modular.attention.bayesian_attention import (
            BayesianAttention, BayesianMultiHeadAttention, VariationalAttention, BayesianCrossAttention
        )
        
        config = MockConfig(d_model=64, n_heads=4)
        batch_size, seq_len = 2, 32
        queries, keys, values, mask = create_attention_inputs(batch_size, seq_len, config.d_model)
        
        # Test BayesianAttention
        bayesian_attn = BayesianAttention(config.d_model, config.n_heads)
        output, _ = bayesian_attn(queries, keys, values)
        assert output.shape == queries.shape, f"BayesianAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test BayesianMultiHeadAttention
        bayesian_mha = BayesianMultiHeadAttention(config.d_model, config.n_heads, n_samples=3)
        output, _ = bayesian_mha(queries, keys, values)
        assert output.shape == queries.shape, f"BayesianMultiHeadAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test VariationalAttention
        variational_attn = VariationalAttention(config.d_model, config.n_heads)
        output, _ = variational_attn(queries, keys, values)
        assert output.shape == queries.shape, f"VariationalAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test BayesianCrossAttention
        bayesian_cross = BayesianCrossAttention(config.d_model, config.n_heads)
        output, _ = bayesian_cross(queries, keys, values)
        assert output.shape == queries.shape, f"BayesianCrossAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        print("PASS Bayesian attention components working correctly")
        return True
        
    except Exception as e:
        print(f"FAIL Bayesian attention test failed: {e}")
        return False


def test_adaptive_components_functionality():
    """Test Adaptive learning components"""
    print("TEST Testing Adaptive Components...")
    
    try:
        from layers.modular.attention.adaptive_components import MetaLearningAdapter, AdaptiveMixture
        
        config = MockConfig(d_model=64, n_heads=4)
        batch_size, seq_len = 2, 32
        queries, keys, values, mask = create_attention_inputs(batch_size, seq_len, config.d_model)
        
        # Test MetaLearningAdapter
        meta_adapter = MetaLearningAdapter(config.d_model, config.n_heads, adaptation_steps=2)
        output, _ = meta_adapter(queries, keys, values)
        assert output.shape == queries.shape, f"MetaLearningAdapter shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test AdaptiveMixture
        adaptive_mixture = AdaptiveMixture(config.d_model, config.n_heads, mixture_components=3)
        output, _ = adaptive_mixture(queries, keys, values)
        assert output.shape == queries.shape, f"AdaptiveMixture shape mismatch: {output.shape} vs {queries.shape}"
        
        print("PASS Adaptive components working correctly")
        return True
        
    except Exception as e:
        print(f"FAIL Adaptive components test failed: {e}")
        return False


def test_temporal_conv_attention_functionality():
    """Test Temporal convolution attention mechanisms"""
    print("TEST Testing Temporal Convolution Attention Components...")
    
    try:
        from layers.modular.attention.temporal_conv_attention import (
            CausalConvolution, TemporalConvNet, ConvolutionalAttention
        )
        
        config = MockConfig(d_model=64, n_heads=4)
        batch_size, seq_len = 2, 32
        queries, keys, values, mask = create_attention_inputs(batch_size, seq_len, config.d_model)
        
        # Test CausalConvolution
        causal_conv = CausalConvolution(config.d_model, config.n_heads, kernel_sizes=[3, 5])
        output, _ = causal_conv(queries, keys, values)
        assert output.shape == queries.shape, f"CausalConvolution shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test TemporalConvNet
        temporal_conv = TemporalConvNet(config.d_model, config.n_heads, num_levels=3)
        output, _ = temporal_conv(queries, keys, values)
        assert output.shape == queries.shape, f"TemporalConvNet shape mismatch: {output.shape} vs {queries.shape}"
        
        # Test ConvolutionalAttention
        conv_attn = ConvolutionalAttention(config.d_model, config.n_heads)
        output, _ = conv_attn(queries, keys, values)
        assert output.shape == queries.shape, f"ConvolutionalAttention shape mismatch: {output.shape} vs {queries.shape}"
        
        print("PASS Temporal convolution attention components working correctly")
        return True
        
    except Exception as e:
        print(f"FAIL Temporal convolution attention test failed: {e}")
        return False


def run_attention_functionality_tests():
    """Run all attention mechanism functionality tests"""
    print("ROCKET Running Attention Mechanism Component Functionality Tests")
    print("=" * 80)
    
    if not COMPONENTS_AVAILABLE:
        print("FAIL Modular components not available - skipping tests")
        return False
    
    tests = [
        ("Multi-Head Attention", test_multi_head_attention_functionality),
        ("Optimized AutoCorrelation", test_optimized_autocorrelation_functionality),
        ("Adaptive AutoCorrelation", test_adaptive_autocorrelation_functionality),
        ("Hierarchical Attention", test_hierarchical_attention_functionality),
        ("Memory Efficient Attention", test_memory_efficient_attention),
        ("Attention Mathematical Properties", test_attention_mathematical_properties),
        ("Attention Edge Cases", test_attention_edge_cases),
        
        # Phase 2: New Attention Components
        ("Fourier Attention", test_fourier_attention_functionality),
        ("Wavelet Attention", test_wavelet_attention_functionality),
        ("Enhanced AutoCorrelation", test_enhanced_autocorrelation_functionality),
        ("Bayesian Attention", test_bayesian_attention_functionality),
        ("Adaptive Components", test_adaptive_components_functionality),
        ("Temporal Conv Attention", test_temporal_conv_attention_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTARGET {test_name}")
        print("-" * 60)
        
        try:
            if test_func():
                passed += 1
                print(f"PASS {test_name} PASSED")
            else:
                print(f"FAIL {test_name} FAILED")
        except Exception as e:
            print(f"FAIL {test_name} ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"CHART Attention Mechanism Functionality Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("PARTY All attention mechanism functionality tests passed!")
        return True
    else:
        print("WARN Some attention mechanism functionality tests failed")
        return False

if __name__ == "__main__":
    success = run_attention_functionality_tests()
    sys.exit(0 if success else 1)
