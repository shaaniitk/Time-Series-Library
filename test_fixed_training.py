"""
Test training with fixed dimensions
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
        
        # Test different component combinations
        self.use_multi_scale_patching = False
        self.use_hierarchical_mapper = False
        self.use_stochastic_learner = False
        self.use_gated_graph_combiner = False
        self.use_mixture_decoder = False
        
        self.num_wave_features = 114

def test_training():
    """Test that training works with fixed dimensions"""
    
    config = SimpleConfig()
    model = Enhanced_SOTA_PGAT(config)
    
    # Create synthetic data
    batch_size = 4
    wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
    target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
    targets = target_window[:, :, :config.c_out]
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Testing training loop...")
    
    for epoch in range(3):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(wave_window, target_window)
        
        # Compute loss
        loss = criterion(output, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: loss={loss.item():.6f}, output_shape={output.shape}")
    
    print("✅ Training test successful!")

def test_different_configs():
    """Test different component configurations"""
    
    configs = [
        ("Baseline", {"use_multi_scale_patching": False, "use_hierarchical_mapper": False}),
        ("With Patching", {"use_multi_scale_patching": True, "use_hierarchical_mapper": False}),
        ("With Hierarchical", {"use_multi_scale_patching": False, "use_hierarchical_mapper": True}),
        ("With Both", {"use_multi_scale_patching": True, "use_hierarchical_mapper": True}),
        ("With MDN", {"use_multi_scale_patching": False, "use_hierarchical_mapper": False, "use_mixture_decoder": True}),
    ]
    
    for name, config_updates in configs:
        print(f"\n🔧 Testing {name}")
        
        config = SimpleConfig()
        for key, value in config_updates.items():
            setattr(config, key, value)
        
        try:
            model = Enhanced_SOTA_PGAT(config)
            
            # Test forward pass
            wave_window = torch.randn(2, config.seq_len, config.enc_in)
            target_window = torch.randn(2, config.pred_len, config.enc_in)
            
            with torch.no_grad():
                output = model(wave_window, target_window)
            
            if isinstance(output, tuple):
                print(f"✅ {name}: MDN output shapes {[o.shape for o in output]}")
            else:
                print(f"✅ {name}: output shape {output.shape}")
                
        except Exception as e:
            print(f"❌ {name}: {e}")

if __name__ == "__main__":
    test_training()
    test_different_configs()