#!/usr/bin/env python3
"""
Test the positional encoding fixes
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_positional_encoding_fixes():
    """Test that positional encoding components work without device errors."""
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        
        class TestConfig:
            def __init__(self):
                self.seq_len = 96
                self.pred_len = 24
                self.enc_in = 7
                self.c_out = 3
                self.d_model = 256
                self.n_heads = 8
                self.dropout = 0.1
                self.use_mixture_density = True
                self.autocorr_factor = 1
                self.max_eigenvectors = 16
                self.enable_graph_attention = True
                self.enable_graph_positional_encoding = True  # Enable this
                self.enable_structural_positional_encoding = True  # Enable this
        
        config = TestConfig()
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        
        # Create test inputs
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
        
        print("🧪 Testing positional encoding fixes...")
        
        # Run forward pass and capture output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            output = model(wave_window, target_window, graph=None)
        
        output_text = f.getvalue()
        
        # Check for skip messages
        structural_skipped = "Structural positional encoding skipped" in output_text
        graph_skipped = "Graph positional encoding skipped" in output_text
        
        print(f"Results:")
        print(f"  - Structural positional encoding skipped: {structural_skipped}")
        print(f"  - Graph positional encoding skipped: {graph_skipped}")
        
        if not structural_skipped and not graph_skipped:
            print("✅ All positional encoding components working!")
            return True
        else:
            print("❌ Some positional encoding components still being skipped")
            if structural_skipped or graph_skipped:
                print("Captured output:")
                print(output_text)
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_positional_encoding_fixes()
    sys.exit(0 if success else 1)