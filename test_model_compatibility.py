#!/usr/bin/env python3
"""
Quick test to verify Enhanced_SOTA_PGAT compatibility with production script
"""
import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_model_compatibility():
    """Test if Enhanced_SOTA_PGAT works with production script interface"""
    print("🔍 Testing Enhanced_SOTA_PGAT compatibility...")
    
    try:
        # Import the model
        from Enhanced_SOTA_PGAT_Refactored import Enhanced_SOTA_PGAT
        
        # Create a simple config
        class SimpleConfig:
            def __init__(self):
                self.seq_len = 250
                self.pred_len = 10
                self.enc_in = 118
                self.c_out = 4
                self.d_model = 128  # Use a power of 2 for better divisibility
                self.n_heads = 8  # 128 is divisible by 8
                self.dropout = 0.1
                
                # Component flags
                self.use_multi_scale_patching = True
                self.use_hierarchical_mapper = True
                self.use_stochastic_learner = True
                self.use_gated_graph_combiner = True
                self.use_mixture_decoder = True
        
        config = SimpleConfig()
        
        # Initialize model
        print("✅ Creating model...")
        model = Enhanced_SOTA_PGAT(config)
        
        # Test forward pass with production script signature
        batch_size = 2
        seq_len = 250
        pred_len = 10
        features = 118
        
        x_enc = torch.randn(batch_size, seq_len, features)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Temporal marks
        x_dec = torch.randn(batch_size, pred_len, features)
        x_mark_dec = torch.randn(batch_size, pred_len, 4)
        
        print("✅ Testing forward pass...")
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(output, tuple):
            print(f"✅ Forward pass successful! Output is tuple with {len(output)} elements")
            for i, elem in enumerate(output):
                if hasattr(elem, 'shape'):
                    print(f"   Element {i}: {elem.shape}")
        else:
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        # Test required attributes
        print("✅ Testing required attributes...")
        assert hasattr(model, 'use_mixture_decoder'), "Missing use_mixture_decoder"
        assert hasattr(model, 'use_stochastic_learner'), "Missing use_stochastic_learner"
        assert hasattr(model, 'get_regularization_loss'), "Missing get_regularization_loss method"
        
        reg_loss = model.get_regularization_loss()
        print(f"✅ Regularization loss: {reg_loss}")
        
        print("🎉 All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_compatibility()
    sys.exit(0 if success else 1)