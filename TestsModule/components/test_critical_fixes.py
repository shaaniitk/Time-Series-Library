#!/usr/bin/env python3
"""
Test critical fixes for graph utilities: per-sample adjacency, edge weights, and type consistency.
"""
import pytest
import torch
import sys
import os
from typing import Dict, List, Tuple, cast

EdgeDict = Dict[Tuple[str, str, str], torch.Tensor]
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils.graph_utils import (
    adjacency_to_edge_indices,
    ensure_tensor_graph_format,
    ensure_tensor_graph_format_with_weights,
    prepare_graph_proposal
)
from torch_geometric.data import HeteroData

@pytest.mark.smoke
def test_per_sample_adjacency_preservation():
    """Test that per-sample adjacency structures are preserved, not collapsed to first sample."""
    
    print("🧪 Testing Per-Sample Adjacency Preservation")
    
    # Create batched adjacency with different patterns per sample
    batch_size = 3
    total_nodes = 7
    wave_nodes, transition_nodes, target_nodes = 3, 2, 2
    
    # Create different adjacency patterns for each batch sample
    batched_adj = torch.zeros(batch_size, total_nodes, total_nodes)
    
    # Sample 0: Strong wave[0] -> transition[0] connection
    batched_adj[0, 0, 3] = 0.9  # wave[0] -> transition[0]
    batched_adj[0, 3, 5] = 0.8  # transition[0] -> target[0]
    
    # Sample 1: Strong wave[1] -> transition[1] connection  
    batched_adj[1, 1, 4] = 0.7  # wave[1] -> transition[1]
    batched_adj[1, 4, 6] = 0.6  # transition[1] -> target[1]
    
    # Sample 2: Strong wave[2] -> transition[0] connection
    batched_adj[2, 2, 3] = 0.5  # wave[2] -> transition[0]
    batched_adj[2, 3, 5] = 0.4  # transition[0] -> target[0]
    
    # Test batch averaging (should preserve information from all samples)
    adjacency_result = adjacency_to_edge_indices(
        batched_adj,
        wave_nodes,
        target_nodes,
        transition_nodes,
        threshold=0.1,
    )

    if isinstance(adjacency_result, tuple):
        edge_index_batches_raw, _edge_weight_batches = adjacency_result
    else:
        edge_index_batches_raw = adjacency_result
        _edge_weight_batches = None

    edge_index_batches = cast(List[EdgeDict], edge_index_batches_raw)

    assert len(edge_index_batches) == batch_size

    expected_wave_edges = [
        torch.tensor([[0], [0]]),
        torch.tensor([[1], [1]]),
        torch.tensor([[0], [2]]),
    ]
    expected_trans_edges = [
        torch.tensor([[0], [0]]),
        torch.tensor([[1], [1]]),
        torch.tensor([[0], [0]]),
    ]

    for idx, edge_dict in enumerate(edge_index_batches):
        wave_trans_edges = edge_dict[('wave', 'interacts_with', 'transition')]
        trans_target_edges = edge_dict[('transition', 'influences', 'target')]
        print(f"  Sample {idx} wave->transition edges: {wave_trans_edges}")
        print(f"  Sample {idx} transition->target edges: {trans_target_edges}")
        assert torch.equal(wave_trans_edges, expected_wave_edges[idx])
        assert torch.equal(trans_target_edges, expected_trans_edges[idx])
    
    # Test specific batch sample selection
    edge_indices_sample0_raw = adjacency_to_edge_indices(
        batched_adj, wave_nodes, target_nodes, transition_nodes, threshold=0.1, batch_idx=0
    )
    
    edge_indices_sample1_raw = adjacency_to_edge_indices(
        batched_adj, wave_nodes, target_nodes, transition_nodes, threshold=0.1, batch_idx=1
    )
    
    edge_indices_sample0 = cast(EdgeDict, edge_indices_sample0_raw)
    edge_indices_sample1 = cast(EdgeDict, edge_indices_sample1_raw)
    
    # Different samples should produce different edge patterns
    wave_trans_0 = edge_indices_sample0[('wave', 'interacts_with', 'transition')]
    wave_trans_1 = edge_indices_sample1[('wave', 'interacts_with', 'transition')]
    
    print(f"  Sample 0 wave->transition: {wave_trans_0}")
    print(f"  Sample 1 wave->transition: {wave_trans_1}")
    
    # Should be different (unless by coincidence)
    different_patterns = not torch.equal(wave_trans_0, wave_trans_1)
    assert different_patterns, "Different batch samples should produce different edge patterns"
    
    print("✅ Per-sample adjacency preservation working correctly")

@pytest.mark.extended
def test_edge_weight_preservation():
    """Test that edge weights are properly extracted and preserved."""
    
    print("🧪 Testing Edge Weight Preservation")
    
    # Create adjacency and corresponding weights
    total_nodes = 7
    wave_nodes, transition_nodes, target_nodes = 3, 2, 2
    
    adj = torch.zeros(total_nodes, total_nodes)
    weights = torch.zeros(total_nodes, total_nodes)
    
    # Set up specific connections with weights
    adj[0, 3] = 0.9    # wave[0] -> transition[0]
    weights[0, 3] = 0.85
    
    adj[1, 4] = 0.7    # wave[1] -> transition[1]  
    weights[1, 4] = 0.75
    
    adj[3, 5] = 0.8    # transition[0] -> target[0]
    weights[3, 5] = 0.82
    
    adj[4, 6] = 0.6    # transition[1] -> target[1]
    weights[4, 6] = 0.65
    
    # Test with edge weights
    result = adjacency_to_edge_indices(
        adj, wave_nodes, target_nodes, transition_nodes, 
        edge_weights=weights, threshold=0.1
    )
    
    # Should return tuple (edge_indices_list, edge_weights_list)
    assert isinstance(result, tuple), f"Expected tuple when weights provided, got {type(result)}"
    edge_index_batches_raw, edge_weight_batches_raw = result
    edge_index_batches = cast(List[EdgeDict], edge_index_batches_raw)
    edge_weight_batches = cast(List[EdgeDict], edge_weight_batches_raw)

    edge_indices = edge_index_batches[0]
    edge_weights = edge_weight_batches[0]

    print(f"  Edge indices keys: {list(edge_indices.keys())}")
    print(f"  Edge weights keys: {list(edge_weights.keys())}")

    # Check wave->transition weights
    wave_trans_weights = edge_weights[('wave', 'interacts_with', 'transition')]
    print(f"  Wave->transition weights: {wave_trans_weights}")
    
    # Should have preserved the actual weight values
    assert wave_trans_weights is not None, "Wave->transition weights should not be None"
    assert len(wave_trans_weights) > 0, "Should have wave->transition weights"
    
    # Check that weights are not just 1.0 (which would indicate they weren't preserved)
    unique_weights = torch.unique(wave_trans_weights)
    print(f"  Unique wave->transition weights: {unique_weights}")
    assert len(unique_weights) > 1 or unique_weights[0] != 1.0, "Weights should be preserved, not defaulted to 1.0"
    
    # Test without edge weights (should return just edge indices)
    result_no_weights_raw = adjacency_to_edge_indices(
        adj,
        wave_nodes,
        target_nodes,
        transition_nodes,
        threshold=0.1,
    )

    assert isinstance(result_no_weights_raw, list), (
        f"Expected list when no weights, got {type(result_no_weights_raw)}"
    )
    no_weight_batches = cast(List[EdgeDict], result_no_weights_raw)
    assert isinstance(no_weight_batches[0], dict)
    
    print("✅ Edge weight preservation working correctly")

@pytest.mark.smoke  
def test_tensor_format_type_consistency():
    """Test that ensure_tensor_graph_format returns consistent types."""
    
    print("🧪 Testing Tensor Format Type Consistency")
    
    # Create test HeteroData
    data = HeteroData()
    data['wave'].num_nodes = 3
    data['transition'].num_nodes = 2
    data['target'].num_nodes = 2
    
    # Add edges with weights
    data['wave', 'interacts_with', 'transition'].edge_index = torch.tensor([[0, 1], [0, 1]])
    data['wave', 'interacts_with', 'transition'].edge_attr = torch.tensor([0.8, 0.6])
    
    data['transition', 'influences', 'target'].edge_index = torch.tensor([[0, 1], [0, 1]])
    data['transition', 'influences', 'target'].edge_attr = torch.tensor([0.9, 0.7])
    
    expected_nodes = 7
    
    # Test ensure_tensor_graph_format (should return tensor only)
    result_tensor_only = ensure_tensor_graph_format(data, expected_nodes)
    assert isinstance(result_tensor_only, torch.Tensor), f"Expected tensor, got {type(result_tensor_only)}"
    assert result_tensor_only.shape == (expected_nodes, expected_nodes), f"Expected shape ({expected_nodes}, {expected_nodes}), got {result_tensor_only.shape}"
    
    print(f"  ensure_tensor_graph_format returned: {type(result_tensor_only)} with shape {result_tensor_only.shape}")
    
    # Test ensure_tensor_graph_format_with_weights with preserve_weights=False
    result_no_weights = ensure_tensor_graph_format_with_weights(data, expected_nodes, preserve_weights=False)
    assert isinstance(result_no_weights, torch.Tensor), f"Expected tensor when preserve_weights=False, got {type(result_no_weights)}"
    
    # Test ensure_tensor_graph_format_with_weights with preserve_weights=True
    result_with_weights = ensure_tensor_graph_format_with_weights(data, expected_nodes, preserve_weights=True)
    assert isinstance(result_with_weights, tuple), f"Expected tuple when preserve_weights=True, got {type(result_with_weights)}"
    
    adj, weights = result_with_weights
    assert isinstance(adj, torch.Tensor), f"Expected tensor for adjacency, got {type(adj)}"
    assert weights is None or isinstance(weights, torch.Tensor), f"Expected tensor or None for weights, got {type(weights)}"
    
    print(f"  ensure_tensor_graph_format_with_weights(preserve_weights=True) returned: ({type(adj)}, {type(weights)})")
    
    # Test with regular tensor input
    tensor_input = torch.rand(expected_nodes, expected_nodes)
    result_tensor_input = ensure_tensor_graph_format(tensor_input, expected_nodes)
    assert isinstance(result_tensor_input, torch.Tensor), f"Expected tensor for tensor input, got {type(result_tensor_input)}"
    assert torch.equal(result_tensor_input, tensor_input), "Tensor input should be returned unchanged when shape matches"
    
    print("✅ Tensor format type consistency working correctly")

@pytest.mark.integration
def test_prepare_graph_proposal_integration():
    """Test that prepare_graph_proposal works with the fixed utilities."""
    
    print("🧪 Testing prepare_graph_proposal Integration")
    
    # Create test HeteroData with weights
    data = HeteroData()
    data['wave'].num_nodes = 3
    data['transition'].num_nodes = 2
    data['target'].num_nodes = 2
    
    data['wave', 'interacts_with', 'transition'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 0]])
    data['wave', 'interacts_with', 'transition'].edge_attr = torch.tensor([0.8, 0.6, 0.4])
    
    data['transition', 'influences', 'target'].edge_index = torch.tensor([[0, 1], [0, 1]])
    data['transition', 'influences', 'target'].edge_attr = torch.tensor([0.9, 0.7])
    
    batch_size = 2
    total_nodes = 7
    
    # Test with preserve_weights=True
    adj_batched, weights_batched = prepare_graph_proposal(
        data, None, batch_size, total_nodes, preserve_weights=True
    )
    
    assert isinstance(adj_batched, torch.Tensor), f"Expected tensor for adjacency, got {type(adj_batched)}"
    assert adj_batched.shape == (batch_size, total_nodes, total_nodes), f"Expected shape ({batch_size}, {total_nodes}, {total_nodes}), got {adj_batched.shape}"
    
    if weights_batched is not None:
        assert isinstance(weights_batched, torch.Tensor), f"Expected tensor for weights, got {type(weights_batched)}"
        assert weights_batched.shape == (batch_size, total_nodes, total_nodes), f"Expected weights shape ({batch_size}, {total_nodes}, {total_nodes}), got {weights_batched.shape}"
        
        # Check that weights are preserved
        unique_weights = torch.unique(weights_batched)
        print(f"  Unique weights in prepared proposal: {unique_weights}")
        assert len(unique_weights) > 2, f"Expected multiple weight values, got {unique_weights}"  # More than just 0 and 1
    
    print("✅ prepare_graph_proposal integration working correctly")

if __name__ == "__main__":
    print("🚀 Testing Critical Graph Utility Fixes")
    print("This test verifies that per-sample adjacency, edge weights, and type consistency are all working.")
    print()
    
    test_per_sample_adjacency_preservation()
    test_edge_weight_preservation()
    test_tensor_format_type_consistency()
    test_prepare_graph_proposal_integration()
    
    print(f"\n🎉 All critical fixes working correctly! Graph utilities are production-ready.")