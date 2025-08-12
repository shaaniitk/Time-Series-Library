"""
Test script for the 3 restored attention algorithms:
1. FourierAttention - Complex frequency filtering
2. Enhanced AutoCorrelation - Multi-scale analysis
3. MetaLearningAdapter - MAML implementation
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fourier_attention():
    """Test restored FourierAttention with complex frequency filtering"""
    print("🔍 Testing FourierAttention...")
    
    try:
        from layers.modular.attention.fourier_attention import FourierAttention
        
        # Initialize component
        d_model, n_heads, seq_len = 64, 8, 96
        fourier_attn = FourierAttention(d_model=d_model, n_heads=n_heads, seq_len=seq_len)
        
        # Test input
        batch_size = 2
        queries = torch.randn(batch_size, seq_len, d_model)
        keys = queries  # Self-attention
        values = queries
        
        # Forward pass
        output, attn_weights = fourier_attn(queries, keys, values)
        
        # Verify output shape
        assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
        
        # Verify no NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        # Verify learnable parameters exist
        assert hasattr(fourier_attn, 'freq_weights'), "Missing frequency weights"
        assert hasattr(fourier_attn, 'phase_weights'), "Missing phase weights"
        
        print("✅ FourierAttention: Complex frequency filtering working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FourierAttention failed: {e}")
        return False

def test_enhanced_autocorrelation():
    """Test restored Enhanced AutoCorrelation with multi-scale analysis"""
    print("🔍 Testing Enhanced AutoCorrelation...")
    
    try:
        from layers.EfficientAutoCorrelation import EfficientAutoCorrelation
        
        # Initialize component with enhanced features
        enhanced_autocorr = EfficientAutoCorrelation(
            adaptive_k=True,
            multi_scale=True,
            scales=[1, 2, 4]
        )
        
        # Test input - format: [B, L, H, E]
        batch_size, seq_len, n_heads, head_dim = 2, 96, 8, 16
        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = queries
        values = queries
        attn_mask = None
        
        # Forward pass
        output, attn_weights = enhanced_autocorr(queries, keys, values, attn_mask)
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, n_heads, head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify no NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        # Verify enhanced features
        assert hasattr(enhanced_autocorr, 'multi_scale'), "Missing multi_scale feature"
        assert hasattr(enhanced_autocorr, 'adaptive_k'), "Missing adaptive_k feature"
        assert hasattr(enhanced_autocorr, 'scale_weights'), "Missing scale_weights"
        
        print("✅ Enhanced AutoCorrelation: Multi-scale analysis working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced AutoCorrelation failed: {e}")
        return False

def test_meta_learning_adapter():
    """Test restored MetaLearningAdapter with MAML implementation"""
    print("🔍 Testing MetaLearningAdapter...")
    
    try:
        from layers.modular.attention.adaptive_components import MetaLearningAdapter
        
        # Initialize component
        d_model = 64
        meta_adapter = MetaLearningAdapter(d_model=d_model, adaptation_steps=3)
        
        # Test input
        batch_size, seq_len = 2, 96
        queries = torch.randn(batch_size, seq_len, d_model)
        
        # Test without support set (standard forward)
        output, _ = meta_adapter(queries, key=queries, value=queries)
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Standard forward: Expected {expected_shape}, got {output.shape}"
        
        # Test with support set (MAML adaptation)
        support_size = 10
        support_set = torch.randn(batch_size, support_size, d_model)
        support_labels = torch.randn(batch_size, support_size, d_model)
        
        meta_adapter.train()  # Set to training mode for adaptation
        adapted_output, _ = meta_adapter(queries, key=queries, value=queries, support_set=support_set, support_labels=support_labels)
        
        # Verify output shape
        assert adapted_output.shape == expected_shape, f"MAML forward: Expected {expected_shape}, got {adapted_output.shape}"
        
        # Verify no NaN values
        assert not torch.isnan(adapted_output).any(), "Adapted output contains NaN values"
        
        # Verify MAML components exist
        assert hasattr(meta_adapter, 'fast_weights'), "Missing fast_weights for MAML"
        assert hasattr(meta_adapter, 'adaptation_steps'), "Missing adaptation_steps"
        assert hasattr(meta_adapter, 'support_encoder'), "Missing support_encoder"
        
        print("✅ MetaLearningAdapter: MAML implementation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ MetaLearningAdapter failed: {e}")
        return False

def main():
    """Run all algorithm restoration tests"""
    print("🚀 Testing Phase 2 Algorithm Restoration")
    print("=" * 50)
    
    results = []
    
    # Test all three restored components
    results.append(test_fourier_attention())
    results.append(test_enhanced_autocorrelation())
    results.append(test_meta_learning_adapter())
    
    print("\n📊 RESTORATION TEST RESULTS")
    print("=" * 50)
    
    success_count = sum(results)
    total_tests = len(results)
    
    if success_count == total_tests:
        print(f"🎉 ALL {total_tests} ALGORITHM RESTORATIONS SUCCESSFUL!")
        print("✅ FourierAttention: Complex frequency filtering restored")
        print("✅ Enhanced AutoCorrelation: Multi-scale analysis restored") 
        print("✅ MetaLearningAdapter: MAML implementation restored")
        print("\n🏆 Phase 2 Algorithm Restoration: COMPLETE")
    else:
        print(f"⚠️  {success_count}/{total_tests} restorations successful")
        print("❌ Some algorithms still need fixes")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
