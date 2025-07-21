#!/usr/bin/env python3
"""
COMPREHENSIVE AUTOCORRELATION TEST
Tests both AutoCorrelationAttention and AutoCorrelationLayer
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_autocorrelation_components():
    """Test both AutoCorrelation components"""
    
    print("🧪 TESTING AUTOCORRELATION COMPONENTS")
    print("=" * 50)
    
    # Test parameters
    d_model = 512
    n_heads = 8
    batch_size = 32
    seq_len = 96
    
    try:
        # Test AutoCorrelationAttention
        print("\n📡 Testing AutoCorrelationAttention...")
        from attention_clean import AutoCorrelationAttention, get_attention_component, ATTENTION_REGISTRY
        
        # Create AutoCorrelationAttention
        autocorr_attention = AutoCorrelationAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            factor=1, 
            dropout=0.1
        )
        
        print("   ✅ Successfully created AutoCorrelationAttention instance")
        
        # Test input shapes - AutoCorrelation expects 4D tensors
        queries = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads)
        keys = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads)
        values = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads)
        
        print(f"   - Input shape: {queries.shape}")
        
        # Forward pass
        with torch.no_grad():
            output, corr = autocorr_attention(queries, keys, values)
        
        print(f"   - Output shape: {output.shape}")
        print(f"   - Correlation shape: {corr.shape}")
        print("   ✅ AutoCorrelationAttention forward pass works!")
        
        # Test registry access
        autocorr_registry = get_attention_component('autocorrelation', d_model=512, n_heads=8)
        print("   ✅ AutoCorrelationAttention via registry works!")
        
    except Exception as e:
        print(f"   ❌ AutoCorrelationAttention failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test AutoCorrelationLayer
        print("\n🏗️  Testing AutoCorrelationLayer...")
        from layers import AutoCorrelationLayer
        
        # Create AutoCorrelationLayer
        autocorr_layer = AutoCorrelationLayer(
            d_model=d_model,
            n_heads=n_heads,
            factor=1,
            dropout=0.1
        )
        
        print("   ✅ Successfully created AutoCorrelationLayer instance")
        
        # Test input for layer - expects 3D tensors
        x = torch.randn(batch_size, seq_len, d_model)
        
        print(f"   - Input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = autocorr_layer(x)
        
        print(f"   - Output shape: {output.shape}")
        print("   ✅ AutoCorrelationLayer forward pass works!")
        
    except Exception as e:
        print(f"   ❌ AutoCorrelationLayer failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test registry integration
        print("\n📝 Testing Registry Integration...")
        from attention_clean import ATTENTION_REGISTRY, get_attention_component
        
        print(f"   - Available components: {list(ATTENTION_REGISTRY.keys())}")
        
        # Test getting AutoCorrelationLayer via registry (if available)
        if 'autocorrelation_layer' in ATTENTION_REGISTRY:
            layer_from_registry = get_attention_component('autocorrelation_layer', d_model=512, n_heads=8)
            print("   ✅ AutoCorrelationLayer via registry works!")
        else:
            print("   ⚠️  AutoCorrelationLayer not available in registry")
        
    except Exception as e:
        print(f"   ❌ Registry integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 AutoCorrelation components test complete!")

if __name__ == "__main__":
    test_autocorrelation_components()
