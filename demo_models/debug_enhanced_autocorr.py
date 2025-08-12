#!/usr/bin/env python3
"""
Debug Enhanced AutoCorrelation to understand the 4D input issue
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_enhanced_autocorr():
    print("🔍 Debugging Enhanced AutoCorrelation")
    print("=" * 50)
    
    from layers.modular.attention.enhanced_autocorrelation import EnhancedAutoCorrelation
    
    # Test parameters
    batch_size, seq_len, d_model, n_heads = 2, 32, 64, 4
    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shapes: Q={queries.shape}, K={keys.shape}, V={values.shape}")
    
    try:
        enhanced_autocorr = EnhancedAutoCorrelation(d_model, n_heads)
        print("✅ Component initialization: SUCCESS")
        
        print("🔄 Starting forward pass...")
        output, _ = enhanced_autocorr(queries, keys, values)
        print(f"✅ Forward pass: SUCCESS, output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_enhanced_autocorr()
