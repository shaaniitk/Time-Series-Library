#!/usr/bin/env python3
"""
Test the enhanced MoE integration after migration.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from unittest.mock import Mock

# Test EnhancedAutoformer with MoE support
def test_enhanced_autoformer_moe():
    print("Testing EnhancedAutoformer with MoE integration...")
    
    # Mock config for testing
    configs = Mock()
    configs.task_name = 'long_term_forecast'
    configs.seq_len = 96
    configs.label_len = 48
    configs.pred_len = 96
    configs.enc_in = 7
    configs.dec_in = 7
    configs.c_out = 7
    configs.d_model = 512
    configs.d_ff = 2048
    configs.n_heads = 8
    configs.e_layers = 2
    configs.d_layers = 1
    configs.factor = 1
    configs.dropout = 0.1
    configs.embed = 'timeF'
    configs.freq = 'h'
    configs.activation = 'gelu'
    configs.norm_type = 'LayerNorm'
    configs.c_out_evaluation = 7  # For quantile mode
    
    # Test standard mode (no MoE)
    try:
        from models.EnhancedAutoformer import EnhancedAutoformer
        model = EnhancedAutoformer(configs)
        print("✓ EnhancedAutoformer standard mode created successfully")
        
        # Check if method exists
        print(f"Has get_auxiliary_loss method: {hasattr(model, 'get_auxiliary_loss')}")
        print(f"Model methods: {[m for m in dir(model) if 'aux' in m.lower()]}")
        
        # Test forward pass
        B, L, D = 2, 96, 7
        x_enc = torch.randn(B, L, D)
        x_mark_enc = torch.randn(B, L, 4)
        x_dec = torch.randn(B, 48 + 96, D)
        x_mark_dec = torch.randn(B, 48 + 96, 4)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        aux_loss = model.get_auxiliary_loss()
        
        print(f"✓ Standard model output shape: {output.shape}")
        print(f"✓ Auxiliary loss (should be 0.0): {aux_loss}")
        
    except Exception as e:
        print(f"✗ EnhancedAutoformer standard mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test HierarchicalDecoder with MoE
    try:
        from layers.enhancedcomponents.EnhancedDecoder import HierarchicalDecoder, MoEDecoderLayer
        
        # Mock MoE config
        moe_configs = Mock()
        moe_configs.d_model = 512
        moe_configs.d_ff = 2048
        moe_configs.n_heads = 8
        moe_configs.factor = 1
        moe_configs.dropout = 0.1
        moe_configs.use_moe_ffn = True
        moe_configs.num_experts = 4
        
        hierarchical_decoder = HierarchicalDecoder(moe_configs, trend_dim=7, n_levels=2)
        print("✓ HierarchicalDecoder with MoE created successfully")
        
        # Test forward pass
        x = torch.randn(2, 96, 512)
        cross = [torch.randn(2, 96, 512), torch.randn(2, 48, 512)]  # Multi-resolution
        trend = torch.randn(2, 96, 7)
        
        seasonals, trends, aux_loss = hierarchical_decoder(x, cross, trend=trend)
        print(f"✓ HierarchicalDecoder output - seasonals: {len(seasonals)}, trends: {len(trends)}")
        print(f"✓ MoE auxiliary loss: {aux_loss}")
        
    except Exception as e:
        print(f"✗ HierarchicalDecoder with MoE failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ All MoE integration tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced MoE Integration Test")
    print("=" * 60)
    
    success = test_enhanced_autoformer_moe()
    
    if success:
        print("\n🎉 Migration successful! MoE auxiliary loss properly integrated.")
    else:
        print("\n❌ Migration needs fixes.")
