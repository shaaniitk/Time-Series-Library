#!/usr/bin/env python3
"""
Debug script to trace the edge indexing issue in enhanced_pgat_layer.py
"""

import sys
import os
import torch
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_edge_indexing():
    """Debug the edge indexing issue step by step."""
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        
        class TestConfig:
            def __init__(self):
                self.seq_len = 96
                self.pred_len = 24
                self.enc_in = 7
                self.c_out = 3
                self.d_model = 512
                self.n_heads = 8
                self.dropout = 0.1
                self.use_mixture_density = True
                self.autocorr_factor = 1
                self.max_eigenvectors = 16
        
        config = TestConfig()
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        
        # Create test inputs
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
        
        print("🔍 Debugging edge indexing issue...")
        print(f"Expected node counts:")
        print(f"  - wave_nodes: {config.enc_in}")
        print(f"  - target_nodes: {config.c_out}")
        print(f"  - transition_nodes: {min(config.enc_in, config.c_out)}")
        
        # Monkey patch the enhanced_pgat_layer to add debug info
        try:
            from layers.modular.graph.enhanced_pgat_layer import EnhancedPGAT_CrossAttn_Layer
            
            # Store original forward method
            original_forward = EnhancedPGAT_CrossAttn_Layer.forward
            
            def debug_forward(self, x_dict, t_dict, edge_index_dict):
                print(f"\n🔍 EnhancedPGAT_CrossAttn_Layer.forward called:")
                print(f"  x_dict keys: {list(x_dict.keys())}")
                for key, tensor in x_dict.items():
                    print(f"    {key}: {tensor.shape}")
                
                print(f"  edge_index_dict keys: {list(edge_index_dict.keys())}")
                for key, edge_index in edge_index_dict.items():
                    print(f"    {key}: {edge_index.shape}")
                    print(f"      edge_index[0] (target): {edge_index[0]}")
                    print(f"      edge_index[1] (source): {edge_index[1]}")
                    print(f"      max edge_index[0]: {edge_index[0].max().item()}")
                    print(f"      max edge_index[1]: {edge_index[1].max().item()}")
                    
                    # Check for out-of-bounds indices
                    source_type, relation, target_type = key
                    if source_type in x_dict and target_type in x_dict:
                        source_size = x_dict[source_type].shape[0]
                        target_size = x_dict[target_type].shape[0]
                        
                        invalid_source = (edge_index[1] >= source_size).any()
                        invalid_target = (edge_index[0] >= target_size).any()
                        
                        if invalid_source:
                            print(f"      ❌ INVALID SOURCE INDICES: max={edge_index[1].max().item()}, size={source_size}")
                        if invalid_target:
                            print(f"      ❌ INVALID TARGET INDICES: max={edge_index[0].max().item()}, size={target_size}")
                            print(f"         edge_index[0] should be < {target_size}, but contains {edge_index[0].max().item()}")
                        
                        if not invalid_source and not invalid_target:
                            print(f"      ✅ All indices valid")
                
                # Call original method
                return original_forward(self, x_dict, t_dict, edge_index_dict)
            
            # Apply monkey patch
            EnhancedPGAT_CrossAttn_Layer.forward = debug_forward
            
        except ImportError:
            print("Could not import EnhancedPGAT_CrossAttn_Layer for debugging")
        
        try:
            output = model(wave_window, target_window, graph=None)
            print("✅ Forward pass completed successfully!")
        except Exception as e:
            print(f"\n❌ Error caught: {e}")
            print(f"\n📍 Analysis:")
            
            error_str = str(e)
            if "index" in error_str and "out of bounds" in error_str:
                # Extract the problematic index and tensor size
                import re
                match = re.search(r'index (\d+) is out of bounds for dimension \d+ with size (\d+)', error_str)
                if match:
                    bad_index = int(match.group(1))
                    tensor_size = int(match.group(2))
                    print(f"  - Trying to access index {bad_index} in tensor of size {tensor_size}")
                    print(f"  - Valid indices should be 0 to {tensor_size-1}")
                    print(f"  - This suggests edge indices are still using old node counts")
                    
                    if tensor_size == 3:
                        print(f"  - Tensor size 3 matches target_nodes (c_out=3) ✅")
                        print(f"  - But edge index {bad_index} suggests it expects more nodes")
                        print(f"  - Likely cause: edge_index created with old node counts")
            
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_edge_indexing()