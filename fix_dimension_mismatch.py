"""
Quick fix for dimension mismatch in systematic training
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
        
        # Test with minimal components first
        self.use_multi_scale_patching = False
        self.use_hierarchical_mapper = False
        self.use_stochastic_learner = False
        self.use_gated_graph_combiner = False
        self.use_mixture_decoder = False
        
        self.num_wave_features = 114

def debug_model_dimensions():
    """Debug the dimension mismatch issue"""
    
    config = SimpleConfig()
    model = Enhanced_SOTA_PGAT(config)
    
    # Create test data
    batch_size = 2
    wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
    target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
    
    print(f"Input shapes:")
    print(f"  wave_window: {wave_window.shape}")
    print(f"  target_window: {target_window.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(wave_window, target_window)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected target shape for loss: {target_window[:, :, :config.c_out].shape}")
    
    # Test loss computation
    targets = target_window[:, :, :config.c_out]  # [batch, pred_len, c_out]
    
    print(f"\nLoss computation:")
    print(f"  output: {output.shape}")
    print(f"  targets: {targets.shape}")
    
    # The issue: output is [batch, c_out, c_out] but should be [batch, pred_len, c_out]
    
    if output.shape[1] != targets.shape[1]:
        print(f"❌ DIMENSION MISMATCH: output seq_len={output.shape[1]}, target seq_len={targets.shape[1]}")
        print(f"   Model is outputting {output.shape[1]} timesteps but should output {targets.shape[1]} timesteps")
    else:
        print(f"✅ Dimensions match!")

if __name__ == "__main__":
    debug_model_dimensions()