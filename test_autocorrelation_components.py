#!/usr/bin/env python3
"""
TEST AUTOCORRELATION COMPONENTS
Test both AutoCorrelationAttention and AutoCorrelationLayer implementations
"""

import torch
import torch.nn as nn
from argparse import Namespace
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_autocorrelation_components():
    """Test AutoCorrelation attention and layer components"""
    
    print("🧪 TESTING AUTOCORRELATION COMPONENTS")
    print("=" * 50)
    
    try:
        # Test AutoCorrelationAttention
        print("\n📡 Testing AutoCorrelationAttention...")
        from utils.modular_components.implementations.attention_clean import AutoCorrelationAttention
        
        # Create attention component
        autocorr_attention = AutoCorrelationAttention(
            d_model=512, 
            n_heads=8, 
            factor=1, 
            dropout=0.1
        )
        
        print(f"   ✅ AutoCorrelationAttention created")
        print(f"   - d_model: {autocorr_attention.d_model}")
        print(f"   - n_heads: {autocorr_attention.n_heads}")
        print(f"   - factor: {autocorr_attention.factor}")
        
        # Test forward pass
        batch_size, seq_len, d_model = 2, 96, 512
        
        # Create test inputs with the expected shape for AutoCorrelation
        queries = torch.randn(batch_size, seq_len, autocorr_attention.n_heads, d_model // autocorr_attention.n_heads)
        keys = torch.randn(batch_size, seq_len, autocorr_attention.n_heads, d_model // autocorr_attention.n_heads)
        values = torch.randn(batch_size, seq_len, autocorr_attention.n_heads, d_model // autocorr_attention.n_heads)
        
        print(f"   - Input shape: {queries.shape}")
        
        with torch.no_grad():
            output, corr = autocorr_attention(queries, keys, values)
        
        print(f"   - Output shape: {output.shape}")
        print(f"   - Correlation shape: {corr.shape}")
        print("   ✅ AutoCorrelationAttention forward pass successful!")
        
        # Test AutoCorrelationLayer
        print("\n🏗️  Testing AutoCorrelationLayer...")
        from utils.modular_components.implementations.layers import AutoCorrelationLayer
        
        # Create layer component
        autocorr_layer = AutoCorrelationLayer(
            d_model=512,
            n_heads=8,
            factor=1,
            d_ff=2048,
            dropout=0.1,
            activation='relu'
        )
        
        print(f"   ✅ AutoCorrelationLayer created")
        print(f"   - d_model: {autocorr_layer.d_model}")
        print(f"   - d_ff: {autocorr_layer.d_ff}")
        print(f"   - factor: {autocorr_layer.factor}")
        
        # Test forward pass with standard transformer input shape
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"   - Input shape: {x.shape}")
        
        with torch.no_grad():
            output = autocorr_layer(x)
        
        print(f"   - Output shape: {output.shape}")
        print("   ✅ AutoCorrelationLayer forward pass successful!")
        
        # Test capabilities
        print("\n🔍 Testing component capabilities...")
        
        attention_caps = autocorr_layer.autocorr_attention.__class__.__name__
        layer_caps = autocorr_layer.get_capabilities()
        
        print(f"   - Attention type: {attention_caps}")
        print(f"   - Layer capabilities: {layer_caps}")
        
        # Test registry access
        print("\n📝 Testing registry access...")
        from utils.modular_components.implementations.attention_clean import ATTENTION_REGISTRY
        from utils.modular_components.implementations.layers import LAYER_REGISTRY
        
        print(f"   - Available attention components: {list(ATTENTION_REGISTRY.keys())}")
        print(f"   - Available layer components: {list(LAYER_REGISTRY.keys())}")
        
        if 'autocorrelation' in ATTENTION_REGISTRY:
            print("   ✅ AutoCorrelationAttention available in registry")
        
        if 'autocorrelation_layer' in LAYER_REGISTRY:
            print("   ✅ AutoCorrelationLayer available in registry")
        
        if 'autocorrelation_layer' in ATTENTION_REGISTRY:
            print("   ✅ AutoCorrelationLayer also available in attention registry (backward compatibility)")
        
        print("\n🎉 ALL AUTOCORRELATION TESTS PASSED!")
        print("Both AutoCorrelationAttention and AutoCorrelationLayer are working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_autocorrelation_components()
    if success:
        print("\n✨ AutoCorrelation components are ready for use!")
        print("📍 AutoCorrelationAttention: Pure attention mechanism in attentions module")
        print("📍 AutoCorrelationLayer: Complete layer with norm+residual+FFN in layers module")
    else:
        print("\n🔧 Further debugging needed.")
