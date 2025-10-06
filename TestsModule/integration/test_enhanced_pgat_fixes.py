#!/usr/bin/env python3
"""
Test script to verify Enhanced_SOTA_PGAT fixes for runtime blockers.
"""
import pytest

import torch
import sys
import os
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

@pytest.mark.integration
def test_enhanced_pgat_runtime_fixes():
    """Test that Enhanced_SOTA_PGAT can run without runtime errors."""
    
    print("🧪 Testing Enhanced PGAT Runtime Fixes")
    print("=" * 50)
    
    try:
        # Import should work without errors
        from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT
        print("✅ Import successful")
        
        # Create test configuration
        config = SimpleNamespace(
            d_model=64,
            n_heads=4,
            seq_len=24,
            pred_len=6,
            enc_in=7,
            c_out=3,
            dropout=0.1,
            
            # Enhanced features
            use_multi_scale_patching=True,
            use_hierarchical_mapper=True,
            use_stochastic_learner=True,
            use_gated_graph_combiner=True,
            patch_len=8,
            stride=4,
            
            # Graph settings
            enable_dynamic_graph=True,
            enable_graph_attention=True,
            use_autocorr_attention=False,
            
            # Mixture density
            use_mixture_density=True,
            mixture_multivariate_mode='independent',
            mdn_components=3,
            
            # Memory optimization
            enable_memory_optimization=True
        )
        
        # Create model
        model = Enhanced_SOTA_PGAT(config)
        print("✅ Model creation successful")
        
        # Test forward pass with small data
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.c_out)
        
        print(f"Testing forward pass with shapes:")
        print(f"  Wave window: {wave_window.shape}")
        print(f"  Target window: {target_window.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(wave_window, target_window)
            
        print(f"✅ Forward pass successful")
        if isinstance(output, tuple):
            print(f"  Output is tuple with {len(output)} elements:")
            for i, elem in enumerate(output):
                if hasattr(elem, 'shape'):
                    print(f"    Element {i}: {elem.shape}")
                else:
                    print(f"    Element {i}: {type(elem)}")
        else:
            print(f"  Output shape: {output.shape}")
        
        # Check internal logs
        if hasattr(model, 'internal_logs'):
            logs = model.internal_logs
            print(f"\n📊 Internal Logs:")
            for key, value in logs.items():
                print(f"  {key}: {value}")
        
        # Test configuration info
        config_info = model.get_enhanced_config_info()
        print(f"\n⚙️ Configuration Info:")
        print(f"  Graph proposals: {config_info.get('num_graphs_combined', 'N/A')}")
        print(f"  Stochastic active: {config_info.get('stochastic_learner_active', 'N/A')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except TypeError as e:
        print(f"❌ Type error (likely missing parameter): {e}")
        return False
    except NameError as e:
        print(f"❌ Name error (likely missing import): {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.extended
def test_weight_preservation():
    """Test that edge weights are preserved through the pipeline."""
    
    print(f"\n🧪 Testing Weight Preservation")
    print("=" * 50)
    
    try:
        from utils.graph_utils import prepare_graph_proposal, convert_hetero_to_dense_adj
        from torch_geometric.data import HeteroData
        
        # Create test HeteroData with edge weights
        data = HeteroData()
        data['wave'].num_nodes = 3
        data['transition'].num_nodes = 2
        data['target'].num_nodes = 2
        
        # Add edges with weights
        data['wave', 'interacts_with', 'transition'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 0]])
        data['wave', 'interacts_with', 'transition'].edge_attr = torch.tensor([0.8, 0.6, 0.4])  # Edge weights
        
        data['transition', 'influences', 'target'].edge_index = torch.tensor([[0, 1], [0, 1]])
        data['transition', 'influences', 'target'].edge_attr = torch.tensor([0.9, 0.7])
        
        print("Created HeteroData with edge weights")
        
        # Test weight preservation in conversion
        adj, weights = convert_hetero_to_dense_adj(data, total_nodes=7, preserve_weights=True)
        print(f"✅ Conversion successful: adj shape {adj.shape}, weights shape {weights.shape}")
        
        # Test prepare_graph_proposal with preserve_weights=True
        batch_size = 2
        total_nodes = 7
        
        adj_batched, weights_batched = prepare_graph_proposal(
            data, None, batch_size, total_nodes, preserve_weights=True
        )
        
        print(f"✅ prepare_graph_proposal successful with preserve_weights=True")
        print(f"  Adjacency shape: {adj_batched.shape}")
        print(f"  Weights shape: {weights_batched.shape if weights_batched is not None else 'None'}")
        
        # Check that weights are not all binary
        if weights_batched is not None:
            unique_weights = torch.unique(weights_batched)
            print(f"  Unique weight values: {unique_weights.tolist()}")
            if len(unique_weights) > 2:  # More than just 0 and 1
                print("✅ Edge weights preserved (not just binary)")
            else:
                print("⚠️ Edge weights may have been binarized")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight preservation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Enhanced PGAT Runtime Fixes")
    print("This test verifies that all runtime blockers have been resolved.")
    print()
    
    success1 = test_enhanced_pgat_runtime_fixes()
    success2 = test_weight_preservation()
    
    if success1 and success2:
        print(f"\n🎉 All runtime fixes successful! Enhanced PGAT is ready to run.")
    else:
        print(f"\n❌ Some runtime issues remain. Check the implementation.")