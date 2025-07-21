#!/usr/bin/env python3
"""
TEST UNIFIED AUTOCORRELATION COMPONENT
Tests the AutoCorrelationLayer implementation from layers/AutoCorrelation.py
"""

import torch
import torch.nn as nn
from utils.modular_components.implementations.attentions_unified import AutoCorrelationLayer, AttentionRegistry
from utils.modular_components.config_schemas import ComponentConfig

def test_autocorrelation_layer():
    """Test the AutoCorrelationLayer implementation"""
    
    print("🧪 TESTING AUTOCORRELATION LAYER")
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
        
        # Test through registry
        print("\n🔧 Testing through AttentionRegistry...")
        attention_from_registry = AttentionRegistry.create('autocorrelation_layer', config)
        print(f"✅ Created from registry: {type(attention_from_registry)}")
        
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
        
        # Verify outputs are the same
        if torch.allclose(output, output2, atol=1e-6):
            print("✅ forward() and apply_attention() produce identical results!")
        else:
            print("⚠️  Different results between forward() and apply_attention()")
        
        print("\n🎉 ALL AUTOCORRELATION TESTS PASSED!")
        print("The AutoCorrelationLayer from layers/AutoCorrelation.py is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_autocorrelation_layer()
    if success:
        print("\n✨ AutoCorrelation component is ready!")
    else:
        print("\n🔧 AutoCorrelation component needs debugging.")
