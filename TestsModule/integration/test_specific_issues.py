#!/usr/bin/env python3
"""
Test specific issues mentioned in the analysis.
"""

import torch
import sys
import os
from types import SimpleNamespace
import pytest

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@pytest.mark.integration
def test_prepare_graph_proposal_preserve_weights():
    """Test that prepare_graph_proposal accepts preserve_weights parameter."""
    
    print("🧪 Testing prepare_graph_proposal with preserve_weights")
    print("=" * 50)
    
    try:
        from utils.graph_utils import prepare_graph_proposal
        
        # Create test data
        adj = torch.rand(7, 7)
        weights = torch.rand(7, 7)
        batch_size = 2
        total_nodes = 7
        
        # Test with preserve_weights=True
        result = prepare_graph_proposal(adj, weights, batch_size, total_nodes, preserve_weights=True)
        print("✅ prepare_graph_proposal accepts preserve_weights=True")
        print(f"  Result shapes: adj={result[0].shape}, weights={result[1].shape if result[1] is not None else None}")
        
        # Test with preserve_weights=False
        result2 = prepare_graph_proposal(adj, weights, batch_size, total_nodes, preserve_weights=False)
        print("✅ prepare_graph_proposal accepts preserve_weights=False")
        
        return True
        
    except TypeError as e:
        if "preserve_weights" in str(e):
            print(f"❌ prepare_graph_proposal does NOT accept preserve_weights: {e}")
            return False
        else:
            print(f"❌ Other TypeError: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

@pytest.mark.smoke
def test_adjacency_to_edge_indices_import():
    """Test that adjacency_to_edge_indices can be imported and used."""
    
    print(f"\n🧪 Testing adjacency_to_edge_indices import and usage")
    print("=" * 50)
    
    try:
        from utils.graph_utils import adjacency_to_edge_indices
        print("✅ adjacency_to_edge_indices import successful")
        
        # Test function call
        adj = torch.rand(7, 7)
        result = adjacency_to_edge_indices(adj, 3, 2, 2)
        print("✅ adjacency_to_edge_indices function call successful")
        if isinstance(result, list):
            print(f"  Result batches: {len(result)}")
            if result:
                print(f"  Result keys (sample 0): {list(result[0].keys())}")
        elif isinstance(result, tuple):
            print("  Result returned edge indices with weights")
        
        return True
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

@pytest.mark.extended
def test_batch_preservation():
    """Test that batch dimensions are preserved through the pipeline."""
    
    print(f"\n🧪 Testing batch dimension preservation")
    print("=" * 50)
    
    try:
        from utils.graph_utils import ensure_tensor_graph_format, prepare_graph_proposal
        
        # Create batched adjacency
        batch_size = 3
        total_nodes = 7
        batched_adj = torch.rand(batch_size, total_nodes, total_nodes)
        
        print(f"Input batched adjacency: {batched_adj.shape}")
        
        # Test ensure_tensor_graph_format
        result1 = ensure_tensor_graph_format(batched_adj, total_nodes)
        print(f"ensure_tensor_graph_format result: {result1.shape}")
        
        if result1.shape[0] != batch_size:
            print(f"❌ Batch dimension collapsed! Expected {batch_size}, got {result1.shape[0]}")
            return False
        else:
            print("✅ Batch dimension preserved in ensure_tensor_graph_format")
        
        # Test prepare_graph_proposal
        result2 = prepare_graph_proposal(batched_adj, None, batch_size, total_nodes)
        print(f"prepare_graph_proposal result: {result2[0].shape}")
        
        if result2[0].shape[0] != batch_size:
            print(f"❌ Batch dimension collapsed in prepare_graph_proposal! Expected {batch_size}, got {result2[0].shape[0]}")
            return False
        else:
            print("✅ Batch dimension preserved in prepare_graph_proposal")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.integration
def test_enhanced_pgat_calls():
    """Test that Enhanced_SOTA_PGAT can make the problematic calls."""
    
    print(f"\n🧪 Testing Enhanced_SOTA_PGAT problematic calls")
    print("=" * 50)
    
    try:
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
        from utils.graph_utils import prepare_graph_proposal
        
        # Create minimal config
        config = SimpleNamespace(
            d_model=32, n_heads=2, seq_len=12, pred_len=3,
            enc_in=4, c_out=2, dropout=0.1,
            use_multi_scale_patching=True,
            use_hierarchical_mapper=True,
            use_stochastic_learner=True,
            use_gated_graph_combiner=True,
            patch_len=4, stride=2,
            enable_dynamic_graph=True,
            enable_graph_attention=True,
            use_autocorr_attention=False,
            use_mixture_density=True,
            mixture_multivariate_mode='independent',
            mdn_components=2,
            enable_memory_optimization=True
        )
        
        model = Enhanced_SOTA_PGAT(config)
        print("✅ Enhanced_SOTA_PGAT creation successful")
        
        # Test the specific calls that were mentioned as problematic
        batch_size = 2
        total_nodes = 7
        
        # Simulate the calls made in the forward method
        dyn_hetero = torch.rand(total_nodes, total_nodes)  # Simulate dynamic graph output
        dyn_weights = None
        
        # This is the call that should fail according to the analysis
        try:
            dyn_proposal = prepare_graph_proposal(dyn_hetero, dyn_weights, batch_size, total_nodes, preserve_weights=True)
            print("✅ dyn_proposal call successful")
        except Exception as e:
            print(f"❌ dyn_proposal call failed: {e}")
            return False
        
        # Test forward pass
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.c_out)
        
        with torch.no_grad():
            output = model(wave_window, target_window)
            print("✅ Forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Specific Issues")
    print("This test addresses the specific problems mentioned in the analysis.")
    print()
    
    success1 = test_prepare_graph_proposal_preserve_weights()
    success2 = test_adjacency_to_edge_indices_import()
    success3 = test_batch_preservation()
    success4 = test_enhanced_pgat_calls()
    
    if all([success1, success2, success3, success4]):
        print(f"\n🎉 All specific issues resolved!")
    else:
        print(f"\n❌ Some issues remain. Check the implementation.")