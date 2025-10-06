#!/usr/bin/env python3
"""
Simple test to verify adjacency_to_edge_indices import works.
"""
import pytest
import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

@pytest.mark.smoke
def test_graph_utils_imports():
    """Test that all graph utility imports work correctly."""
    try:
        from utils.graph_utils import adjacency_to_edge_indices
        assert adjacency_to_edge_indices is not None
        assert callable(adjacency_to_edge_indices)
        
        # Test other imports
        from utils.graph_utils import convert_hetero_to_dense_adj, prepare_graph_proposal, validate_graph_proposals
        assert convert_hetero_to_dense_adj is not None
        assert prepare_graph_proposal is not None
        assert validate_graph_proposals is not None
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")