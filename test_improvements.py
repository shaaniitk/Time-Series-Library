"""
Test script demonstrating the improvements to HSDGNN components
"""

import torch
import numpy as np
from layers.DynamicGraphAttention import DynamicGraphConstructor

def test_weighted_vs_binary_adjacency():
    """Demonstrate the difference between weighted and binary adjacency matrices"""
    
    # Create test data with known correlations
    torch.manual_seed(42)
    B, L, D = 2, 50, 4
    
    # Create correlated data
    base_signal = torch.randn(B, L, 1)
    x = torch.cat([
        base_signal + 0.1 * torch.randn(B, L, 1),  # Strong correlation
        base_signal + 0.5 * torch.randn(B, L, 1),  # Medium correlation  
        base_signal + 1.0 * torch.randn(B, L, 1),  # Weak correlation
        torch.randn(B, L, 1)                       # No correlation
    ], dim=-1)
    
    # Test binary adjacency (old approach)
    constructor_binary = DynamicGraphConstructor(
        window_size=20, threshold=0.3, use_weighted=False, include_self_loops=False
    )
    adj_binary = constructor_binary(x)
    
    # Test weighted adjacency (improved approach)
    constructor_weighted = DynamicGraphConstructor(
        window_size=20, threshold=0.3, use_weighted=True, include_self_loops=True
    )
    adj_weighted = constructor_weighted(x)
    
    print("=== Adjacency Matrix Comparison ===")
    print(f"Binary adjacency (last timestep):\n{adj_binary[0, -1]}")
    print(f"Weighted adjacency (last timestep):\n{adj_weighted[0, -1]}")
    
    # Show information preservation
    binary_info = torch.sum(adj_binary > 0).item()
    weighted_info = torch.sum(adj_weighted).item()
    
    print(f"\nInformation content:")
    print(f"Binary: {binary_info} connections")
    print(f"Weighted: {weighted_info:.3f} total weight")
    print(f"Information ratio: {weighted_info/max(binary_info, 1):.3f}x more information")
    
    return adj_binary, adj_weighted

def test_self_loops_impact():
    """Demonstrate the impact of including self-loops"""
    
    torch.manual_seed(42)
    x = torch.randn(1, 30, 3)
    
    # Without self-loops
    constructor_no_self = DynamicGraphConstructor(
        window_size=10, threshold=0.2, include_self_loops=False
    )
    adj_no_self = constructor_no_self(x)
    
    # With self-loops  
    constructor_with_self = DynamicGraphConstructor(
        window_size=10, threshold=0.2, include_self_loops=True
    )
    adj_with_self = constructor_with_self(x)
    
    print("\n=== Self-Loops Impact ===")
    print(f"Diagonal (self-loops) without: {torch.diag(adj_no_self[0, -1])}")
    print(f"Diagonal (self-loops) with: {torch.diag(adj_with_self[0, -1])}")
    
    return adj_no_self, adj_with_self

if __name__ == "__main__":
    print("Testing HSDGNN Improvements...")
    test_weighted_vs_binary_adjacency()
    test_self_loops_impact()
    print("\n✅ All improvement tests completed!")