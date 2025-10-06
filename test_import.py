#!/usr/bin/env python3
"""
Simple test to verify adjacency_to_edge_indices import works.
"""

try:
    from utils.graph_utils import adjacency_to_edge_indices
    print("✅ Import successful")
    print(f"Function: {adjacency_to_edge_indices}")
    print(f"Function type: {type(adjacency_to_edge_indices)}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

# Also test other imports
try:
    from utils.graph_utils import convert_hetero_to_dense_adj, prepare_graph_proposal, validate_graph_proposals
    print("✅ Other imports successful")
except ImportError as e:
    print(f"❌ Other imports failed: {e}")