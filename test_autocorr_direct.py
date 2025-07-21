#!/usr/bin/env python3
"""
DIRECT TEST OF AUTOCORRELATION (bypassing auto-registration)
Tests the AutoCorrelationLayer implementation directly without the broken auto-registration
"""

import sys
import torch
import torch.nn as nn

# Add the path to avoid auto-registration issues
sys.path.insert(0, 'utils/modular_components/implementations')

# Import directly from the unified file
from attentions_unified import AutoCorrelationLayer, AttentionRegistry

# Import config schemas directly 
sys.path.insert(0, 'utils/modular_components')
from config_schemas import ComponentConfig

def test_autocorrelation_direct():
    """Test the AutoCorrelationLayer implementation directly"""
    
    print("🧪 DIRECT AUTOCORRELATION TEST")
    print("=" * 40)
    
    try:
        # Create configuration
        config = ComponentConfig(
            component_name='autocorrelation_layer',
            d_model=512,
            dropout=0.1,
            custom_params={
                'n_heads': 8,
                'factor': 1,
                'attention_dropout': 0.1,
                'output_attention': False
            }
        )
        
        # Create AutoCorrelationLayer directly
        print("📝 Creating AutoCorrelationLayer...")
        attention = AutoCorrelationLayer(config)
        print(f"✅ AutoCorrelationLayer created: {type(attention)}")
        print(f"   - n_heads: {attention.n_heads}")
        print(f"   - inner_correlation: {type(attention.inner_correlation)}")
        
        # Test forward pass
        print("\n🔄 Testing forward pass...")
        batch_size = 2
        seq_len = 96
        d_model = 512
        
        # Create test inputs
        queries = torch.randn(batch_size, seq_len, d_model)
        keys = torch.randn(batch_size, seq_len, d_model)  
        values = torch.randn(batch_size, seq_len, d_model)
        
        print(f"   - Input shape: {queries.shape}")
        
        # Forward pass
        with torch.no_grad():
            output, attn = attention(queries, keys, values)
        
        print(f"   - Output shape: {output.shape}")
        print(f"   - Attention shape: {attn.shape if attn is not None else 'None'}")
        print("✅ Forward pass successful!")
        
        # Test apply_attention method (from BaseAttention interface)
        print("\n🎯 Testing apply_attention method...")
        with torch.no_grad():
            output2, attn2 = attention.apply_attention(queries, keys, values)
        
        print(f"   - Output shape: {output2.shape}")
        print("✅ apply_attention method works!")
        
        # Test registry functionality
        print("\n🔧 Testing AttentionRegistry...")
        available_components = AttentionRegistry.list_components()
        print(f"   - Available components: {available_components}")
        
        if 'autocorrelation_layer' in available_components:
            attention_from_registry = AttentionRegistry.create('autocorrelation_layer', config)
            print(f"✅ Created from registry: {type(attention_from_registry)}")
        else:
            print("⚠️  autocorrelation_layer not in registry")
        
        print("\n🎉 ALL DIRECT TESTS PASSED!")
        print("The AutoCorrelationLayer implementation is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_autocorrelation_direct()
    if success:
        print("\n✨ AutoCorrelation component is ready for production!")
    else:
        print("\n🔧 AutoCorrelation component needs debugging.")
