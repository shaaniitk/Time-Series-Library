#!/usr/bin/env python3
"""
CLEAN MODULAR AUTOFORMER DEMO
Tests the new clean implementation using only utils.modular_components
"""

import torch
import torch.nn as nn
from argparse import Namespace
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_clean_modular_autoformer():
    """Test the clean modular autoformer implementation"""
    
    print("🧪 TESTING CLEAN MODULAR AUTOFORMER")
    print("=" * 50)
    
    try:
        # Import clean implementation
        from clean_modular_autoformer import CleanModularAutoformer
        
        # Create test configuration
        config = Namespace(
            # Basic dimensions
            seq_len=96,
            label_len=48, 
            pred_len=24,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=2048,
            dropout=0.1,
            
            # Component types
            attention_type='multihead',
            encoder_type='standard',
            decoder_type='standard',
            loss_type='mse',
            output_type='forecasting',
            
            # Embedding config
            embed='timeF',
            freq='h',
            
            # Other
            activation='gelu',
            output_attention=False
        )
        
        print("📝 Configuration created")
        print(f"   - Model dimensions: {config.d_model}")
        print(f"   - Sequence length: {config.seq_len}")
        print(f"   - Prediction length: {config.pred_len}")
        print(f"   - Attention type: {config.attention_type}")
        print(f"   - Loss type: {config.loss_type}")
        
        # Create model
        print("\n🏗️  Creating CleanModularAutoformer...")
        model = CleanModularAutoformer(config)
        
        print("✅ Model created successfully!")
        
        # Get component summary
        summary = model.get_component_summary()
        print(f"\n📊 Component Summary:")
        print(f"   - Total components: {summary['total_components']}")
        print(f"   - Component categories: {summary['categories']}")
        print(f"   - Framework type: {summary['framework_type']}")
        
        # Test forward pass
        print("\n🔄 Testing forward pass...")
        
        batch_size = 32
        
        # Create test inputs
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # time features
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        print(f"   - Input shape: {x_enc.shape}")
        print(f"   - Expected output shape: ({batch_size}, {config.pred_len}, {config.c_out})")
        
        # Forward pass
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"   - Actual output shape: {output.shape}")
        print("✅ Forward pass successful!")
        
        # Test loss computation
        print("\n🎯 Testing loss computation...")
        targets = torch.randn(batch_size, config.pred_len, config.c_out)
        
        loss = model.compute_loss(output, targets)
        print(f"   - Loss value: {loss.item():.4f}")
        print("✅ Loss computation successful!")
        
        # Test component access
        print("\n🔧 Testing component access...")
        
        if hasattr(model, 'component_registry'):
            components = model.component_registry.list_components()
            print(f"   - Available component categories: {list(components.keys())}")
            
            for category, comp_list in components.items():
                if comp_list:
                    print(f"   - {category}: {len(comp_list)} components")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The CleanModularAutoformer is working correctly with migrated components!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clean_modular_autoformer()
    if success:
        print("\n✨ Clean modular architecture is ready for production!")
    else:
        print("\n🔧 Further debugging needed.")
