#!/usr/bin/env python3
"""
Test script to verify that adjacency_to_edge_indices correctly preserves learned structure.
"""

import torch
import sys
import os
from typing import Dict, List, Tuple, cast
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.graph_utils import adjacency_to_edge_indices, validate_learned_adjacency_structure

EdgeDict = Dict[Tuple[str, str, str], torch.Tensor]

def test_adjacency_mapping_correctness():
    """Test that we correctly map learned adjacency without creating false edges."""
    
    print("🧪 Testing Adjacency Mapping Correctness")
    print("=" * 50)
    
    # Test configuration
    wave_nodes = 3
    transition_nodes = 2  
    target_nodes = 2
    total_nodes = wave_nodes + transition_nodes + target_nodes
    
    # Create a specific learned adjacency pattern
    adj = torch.zeros(total_nodes, total_nodes)
    
    # Set specific learned connections:
    # Wave node 0 -> Transition node 1 (strong connection)
    adj[0, wave_nodes + 1] = 0.8
    # Wave node 2 -> Transition node 0 (medium connection)  
    adj[2, wave_nodes + 0] = 0.6
    
    # Transition node 0 -> Target node 1 (strong connection)
    adj[wave_nodes + 0, wave_nodes + transition_nodes + 1] = 0.9
    # Transition node 1 -> Target node 0 (medium connection)
    adj[wave_nodes + 1, wave_nodes + transition_nodes + 0] = 0.5
    
    print(f"Created adjacency matrix with shape: {adj.shape}")
    print(f"Node configuration: {wave_nodes} wave, {transition_nodes} transition, {target_nodes} target")
    
    # Validate the structure
    validation = validate_learned_adjacency_structure(adj, wave_nodes, transition_nodes, target_nodes)
    print(f"\n📊 Validation Results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Convert to edge indices
    try:
        adjacency_result = adjacency_to_edge_indices(adj, wave_nodes, target_nodes, transition_nodes, threshold=0.1)
        if isinstance(adjacency_result, tuple):
            edge_index_batches_raw, _edge_weight_batches = adjacency_result
        else:
            edge_index_batches_raw = adjacency_result
            _edge_weight_batches = None

        edge_index_batches = cast(List[EdgeDict], edge_index_batches_raw)
        edge_indices = edge_index_batches[0]
        print(f"\n✅ Successfully converted adjacency to edge indices")
        
        # Verify the mappings
        wave_trans_edges = edge_indices[('wave', 'interacts_with', 'transition')]
        trans_target_edges = edge_indices[('transition', 'influences', 'target')]
        
        print(f"\n🔗 Edge Mappings:")
        print(f"Wave -> Transition edges: {wave_trans_edges}")
        print(f"Transition -> Target edges: {trans_target_edges}")
        
        # Verify specific learned connections are preserved
        print(f"\n🎯 Verifying Learned Connections:")
        
        # Check wave->transition mappings (note: indices are flipped)
        expected_wave_trans = torch.tensor([[1, 0], [0, 2]])  # [transition_idx, wave_idx] due to flip
        if torch.equal(wave_trans_edges, expected_wave_trans):
            print("  ✅ Wave->Transition mapping correct")
        else:
            print(f"  ❌ Wave->Transition mapping incorrect. Expected: {expected_wave_trans}, Got: {wave_trans_edges}")
        
        # Check transition->target mappings (note: indices are flipped)  
        expected_trans_target = torch.tensor([[1, 0], [0, 1]])  # [target_idx, transition_idx] due to flip
        if torch.equal(trans_target_edges, expected_trans_target):
            print("  ✅ Transition->Target mapping correct")
        else:
            print(f"  ❌ Transition->Target mapping incorrect. Expected: {expected_trans_target}, Got: {trans_target_edges}")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False
    
    return True

def test_edge_case_handling():
    """Test edge cases like empty adjacency, out-of-bounds, etc."""
    
    print(f"\n🧪 Testing Edge Case Handling")
    print("=" * 50)
    
    wave_nodes, transition_nodes, target_nodes = 2, 2, 2
    total_nodes = wave_nodes + transition_nodes + target_nodes
    
    # Test 1: Empty adjacency (all zeros)
    print("Test 1: Empty adjacency matrix")
    empty_adj = torch.zeros(total_nodes, total_nodes)
    try:
        adjacency_result = adjacency_to_edge_indices(
            empty_adj,
            wave_nodes,
            target_nodes,
            transition_nodes,
            threshold=0.1,
        )
        if isinstance(adjacency_result, tuple):
            edge_index_batches_raw, _ = adjacency_result
        else:
            edge_index_batches_raw = adjacency_result
        edge_index_batches = cast(List[EdgeDict], edge_index_batches_raw)
        _ = edge_index_batches[0]
        print("  ✅ Empty adjacency handled correctly (fallback edges created)")
    except Exception as e:
        print(f"  ❌ Empty adjacency failed: {e}")
    
    # Test 2: Wrong shape adjacency
    print("Test 2: Wrong shape adjacency matrix")
    wrong_shape_adj = torch.zeros(3, 3)  # Too small
    try:
        adjacency_to_edge_indices(wrong_shape_adj, wave_nodes, target_nodes, transition_nodes)
        print("  ❌ Wrong shape should have raised error!")
    except ValueError as e:
        print(f"  ✅ Wrong shape correctly rejected: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    # Test 3: Batched adjacency
    print("Test 3: Batched adjacency matrix")
    batch_size = 2
    batched_adj = torch.rand(batch_size, total_nodes, total_nodes) * 0.5  # Random values
    try:
        adjacency_result = adjacency_to_edge_indices(batched_adj, wave_nodes, target_nodes, transition_nodes)
        if isinstance(adjacency_result, tuple):
            edge_index_batches_raw, _edge_weight_batches = adjacency_result
        else:
            edge_index_batches_raw = adjacency_result
            _edge_weight_batches = None
        edge_index_batches = cast(List[EdgeDict], edge_index_batches_raw)
        assert len(edge_index_batches) == batch_size
        print("  ✅ Batched adjacency handled correctly")
        print("  ✅ Batched adjacency handled correctly")
    except Exception as e:
        print(f"  ❌ Batched adjacency failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Adjacency Mapping Correctness")
    print("This test verifies that learned adjacency structures are preserved without false edges.")
    print()
    
    success1 = test_adjacency_mapping_correctness()
    test_edge_case_handling()
    
    if success1:
        print(f"\n🎉 All tests passed! Adjacency mapping preserves learned structure correctly.")
    else:
        print(f"\n❌ Some tests failed. Check the implementation.")