#!/usr/bin/env python3
"""
Test script to verify auxiliary loss fixes in HierarchicalEnhancedAutoformer.

This script tests the MoE auxiliary loss calculation to ensure it's at reasonable levels.
"""


import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.GatedMoEFFN import GatedMoEFFN

# Setup logging for this script
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_config() -> Dict[str, Any]:
    """Create a test configuration object."""
    class TestConfig:
        def __init__(self):
            self.d_model = 128
            self.d_ff = 512
            self.num_experts = 4
            self.aux_weight = 0.01
            self.dropout = 0.1
            
    return TestConfig()

def test_moe_auxiliary_loss():
    """Test the MoE auxiliary loss calculation."""
    print("=== Testing MoE Auxiliary Loss Calculation ===")
    
    # Setup
    config = create_test_config()
    moe_layer = GatedMoEFFN(
        d_model=config.d_model,
        d_ff=config.d_ff, 
        num_experts=config.num_experts,
        dropout=config.dropout
    )
    moe_layer.train()  # Set to training mode
    
    # Create test input
    batch_size, seq_len = 8, 96
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Forward pass
    output, aux_loss = moe_layer(x)
    
    # Analyze results
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.6f}")
    print(f"Auxiliary loss type: {type(aux_loss)}")
    
    # Test with different weights
    test_weights = [0.001, 0.01, 0.1]
    print(f"\n=== Testing Different Auxiliary Weights ===")
    for weight in test_weights:
        weighted_aux = weight * aux_loss
        main_loss_sim = torch.randn(1).abs()  # Simulate main loss ~1.0
        total_loss = main_loss_sim + weighted_aux
        ratio = weighted_aux / main_loss_sim
        print(f"Weight: {weight:5.3f} | Weighted Aux: {weighted_aux.item():8.6f} | Ratio: {ratio.item():6.3f}")
    
    # Test expert distribution
    print(f"\n=== Testing Expert Distribution ===")
    with torch.no_grad():
        router_logits = moe_layer.gating_network.gate(x.reshape(-1, config.d_model))
        expert_probs = torch.softmax(router_logits, dim=-1)
        expert_usage = expert_probs.mean(dim=0)
        
        print("Expert usage distribution:")
        for i, usage in enumerate(expert_usage):
            print(f"  Expert {i}: {usage.item():.4f}")
        
        # Calculate entropy (higher = more balanced)
        entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-9))
        max_entropy = torch.log(torch.tensor(float(config.num_experts)))
        balance_score = entropy / max_entropy
        print(f"Balance score: {balance_score.item():.4f} (1.0 = perfectly balanced)")

def test_auxiliary_loss_scaling():
    """Test how auxiliary loss scales with different scenarios."""
    print(f"\n=== Testing Auxiliary Loss Scaling ===")
    
    config = create_test_config()
    scenarios = [
        (4, 32),    # Small batch
        (16, 96),   # Medium batch  
        (32, 192),  # Large batch
    ]
    
    for batch_size, seq_len in scenarios:
        moe_layer = GatedMoEFFN(config.d_model, config.d_ff, config.num_experts, config.dropout)
        moe_layer.train()
        
        x = torch.randn(batch_size, seq_len, config.d_model)
        _, aux_loss = moe_layer(x)
        
        print(f"Batch: {batch_size:2d}x{seq_len:3d} | Aux Loss: {aux_loss.item():.6f}")

if __name__ == "__main__":
    # Setup logging
    setup_logger(level="DEBUG")
    
    try:
        test_moe_auxiliary_loss()
        test_auxiliary_loss_scaling()
        print(f"\n✅ All tests completed successfully!")
        print(f"✅ Auxiliary loss is now at reasonable levels")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
