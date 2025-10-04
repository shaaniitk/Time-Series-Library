#!/usr/bin/env python3
"""
Debug script to trace tensor shapes through the SOTA_Temporal_PGAT forward pass
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_forward_pass():
    """Debug the forward pass step by step to identify where tensor shapes go wrong."""
    
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
        
        config = TestConfig()
        
        print("🔍 Creating model...")
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        
        print(f"✅ Model created successfully")
        print(f"   - enc_in (input features): {config.enc_in}")
        print(f"   - c_out (output features): {config.c_out}")
        print(f"   - seq_len (input sequence): {config.seq_len}")
        print(f"   - pred_len (prediction sequence): {config.pred_len}")
        
        # Create test inputs
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
        
        print(f"\n📊 Input tensor shapes:")
        print(f"   - wave_window: {wave_window.shape}")
        print(f"   - target_window: {target_window.shape}")
        
        # Add debug prints to the model by monkey-patching
        original_forward = model.forward
        
        def debug_forward(self, wave_window, target_window, graph=None):
            print(f"\n🚀 Starting forward pass...")
            
            # Initial concatenation
            combined_input = torch.cat([wave_window, target_window], dim=1)
            print(f"   - combined_input: {combined_input.shape}")
            
            # After embedding
            batch_size, seq_len, features = combined_input.shape
            if features == getattr(self, 'd_model', 512):
                embedded = combined_input
            else:
                embedded = self.embedding(combined_input.view(-1, features)).view(batch_size, seq_len, -1)
            print(f"   - embedded: {embedded.shape}")
            
            # Split back
            wave_len = wave_window.shape[1]
            target_len = target_window.shape[1]
            wave_embedded = embedded[:, :wave_len, :]
            target_embedded = embedded[:, wave_len:wave_len+target_len, :]
            print(f"   - wave_embedded: {wave_embedded.shape}")
            print(f"   - target_embedded: {target_embedded.shape}")
            
            # Node counts (the fix we made)
            wave_nodes = getattr(self.config, 'enc_in', 7)
            target_nodes = getattr(self.config, 'c_out', 3)
            transition_nodes = max(1, min(wave_nodes, target_nodes))
            print(f"   - wave_nodes: {wave_nodes}")
            print(f"   - target_nodes: {target_nodes}")
            print(f"   - transition_nodes: {transition_nodes}")
            
            print(f"\n❌ This is where the shape mismatch occurs:")
            print(f"   - wave_embedded has shape {wave_embedded.shape} (batch, seq_len, d_model)")
            print(f"   - But model expects wave features with shape ({batch_size}, {wave_nodes}, {self.d_model})")
            print(f"   - The model is trying to treat sequence length ({wave_len}) as number of nodes")
            print(f"   - But we fixed it to use feature count ({wave_nodes}) as number of nodes")
            print(f"   - The issue is that the tensor reshaping logic hasn't been updated")
            
            return None  # Don't continue the forward pass
        
        # Monkey patch for debugging
        model.forward = debug_forward.__get__(model, SOTA_Temporal_PGAT)
        
        # Run debug forward pass
        try:
            output = model(wave_window, target_window, graph=None)
        except Exception as e:
            print(f"\n💡 The core issue is clear:")
            print(f"   - We fixed the node count calculation to use feature dimensions")
            print(f"   - But the tensor processing pipeline still expects sequence-length-based shapes")
            print(f"   - We need to add proper tensor reshaping/aggregation logic")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        return False

if __name__ == "__main__":
    debug_forward_pass()