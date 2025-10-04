#!/usr/bin/env python3
"""
Debug script to identify and fix the graph attention error
"""

import sys
import os
import torch
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_graph_attention():
    """Debug the graph attention error specifically."""
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        
        class TestConfig:
            def __init__(self):
                self.seq_len = 96
                self.pred_len = 24
                self.enc_in = 7
                self.c_out = 3
                self.d_model = 512
                self.n_heads = 8
                self.dropout = 0.1
                self.use_mixture_density = True
                self.autocorr_factor = 1
                self.max_eigenvectors = 16
                self.enable_graph_attention = True  # Ensure it's enabled
        
        config = TestConfig()
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        
        # Create test inputs
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
        
        print("🔍 Debugging graph attention error...")
        print(f"Graph attention enabled: {getattr(config, 'enable_graph_attention', True)}")
        print(f"Graph attention component: {type(model.graph_attention)}")
        
        # Monkey patch to catch the exact graph attention error
        original_graph_attention_call = None
        
        def debug_forward(self, wave_window, target_window, graph=None):
            # Just run the normal forward pass but catch the graph attention error
            try:
                return original_forward(wave_window, target_window, graph)
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                traceback.print_exc()
                return f"FAILED: {e}"
        
        # Store original forward
        original_forward = model.forward
        
        # Apply monkey patch
        model.forward = debug_forward.__get__(model, SOTA_Temporal_PGAT)
        
        # Test
        result = model(wave_window, target_window, graph=None)
        
        if result == "SUCCESS":
            print(f"\n🎉 Graph attention is working correctly!")
            return True
        else:
            print(f"\n💡 Graph attention needs to be fixed: {result}")
            return False
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_graph_attention()