#!/usr/bin/env python3
"""
Test to verify Enhanced_SOTA_PGAT imports work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Test the specific import that's causing issues
    from utils.graph_utils import adjacency_to_edge_indices
    print("✅ adjacency_to_edge_indices import successful")
    
    # Test the Enhanced_SOTA_PGAT import
    from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
    print("✅ Enhanced_SOTA_PGAT import successful")
    
    # Test that the function is accessible within the model
    import torch
    from types import SimpleNamespace
    
    config = SimpleNamespace(
        d_model=64, n_heads=4, seq_len=24, pred_len=6,
        enc_in=7, c_out=3, dropout=0.1,
        use_multi_scale_patching=True,
        use_hierarchical_mapper=True,
        use_stochastic_learner=True,
        use_gated_graph_combiner=True,
        patch_len=8, stride=4,
        enable_dynamic_graph=True,
        enable_graph_attention=True,
        use_autocorr_attention=False,
        use_mixture_density=True,
        mixture_multivariate_mode='independent',
        mdn_components=3,
        enable_memory_optimization=True
    )
    
    model = Enhanced_SOTA_PGAT(config)
    print("✅ Enhanced_SOTA_PGAT model creation successful")
    
    # Test that adjacency_to_edge_indices is accessible
    test_adj = torch.rand(7, 7)
    edge_indices = adjacency_to_edge_indices(test_adj, 3, 2, 2)
    print("✅ adjacency_to_edge_indices function call successful")
    print(f"  Returned edge types: {list(edge_indices.keys())}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()