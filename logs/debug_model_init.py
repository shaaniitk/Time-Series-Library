"""
Debug model initialization
"""

import torch
from Enhanced_SOTA_PGAT_Refactored import Enhanced_SOTA_PGAT

class SimpleConfig:
    def __init__(self):
        self.seq_len = 24
        self.pred_len = 6
        self.enc_in = 118
        self.c_out = 4
        self.d_model = 64
        self.n_heads = 4
        self.dropout = 0.1
        
        # All components disabled
        self.use_multi_scale_patching = False
        self.use_hierarchical_mapper = False
        self.use_stochastic_learner = False
        self.use_gated_graph_combiner = False
        self.use_mixture_decoder = False
        
        self.num_wave_features = 114

try:
    print("Creating config...")
    config = SimpleConfig()
    
    print("Creating model...")
    model = Enhanced_SOTA_PGAT(config)
    
    print("Model created successfully!")
    print(f"Model type: {type(model)}")
    print(f"Has forward method: {hasattr(model, 'forward')}")
    
    # Test forward pass
    print("Creating test data...")
    wave_window = torch.randn(2, 24, 118)
    target_window = torch.randn(2, 6, 118)
    
    print("Running forward pass...")
    with torch.no_grad():
        output = model(wave_window, target_window)
    
    print(f"Success! Output shape: {output.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()