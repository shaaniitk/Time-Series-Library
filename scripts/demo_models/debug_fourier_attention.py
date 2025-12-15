#!/usr/bin/env python3
"""
Debug FourierAttention to understand the tensor dimension issue
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_fourier_attention():
    print("🔍 Debugging FourierAttention")
    print("=" * 50)
    
    from layers.modular.attention.fourier_attention import FourierAttention
    
    # Test parameters
    batch_size, seq_len, d_model, n_heads = 2, 32, 64, 4
    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shapes: Q={queries.shape}, K={keys.shape}, V={values.shape}")
    
    try:
        fourier_attn = FourierAttention(d_model, n_heads, seq_len=seq_len)
        print("✅ Component initialization: SUCCESS")
        
        print("🔄 Starting forward pass...")
        output, _ = fourier_attn(queries, keys, values)
        print(f"✅ Forward pass: SUCCESS, output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fourier_attention()
