"""
Test Enhanced SOTA PGAT with realistic sequence lengths
seq_len=750, pred_len=20 (much more realistic for financial data)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from Enhanced_SOTA_PGAT_Refactored import Enhanced_SOTA_PGAT
from layers.utils.model_utils import PatchConfigGenerator
import time

class RealisticConfig:
    def __init__(self):
        # REALISTIC FINANCIAL TIME SERIES CONFIGURATION
        self.seq_len = 750      # ~3 years of daily data
        self.pred_len = 20      # ~1 month prediction
        self.enc_in = 118       # Your celestial + OHLC features
        self.c_out = 4          # OHLC targets
        self.d_model = 128      # Model dimension
        self.n_heads = 8        # Attention heads
        self.dropout = 0.1
        
        # Component toggles
        self.use_multi_scale_patching = True   # Should be much better now!
        self.use_hierarchical_mapper = True
        self.use_stochastic_learner = True
        self.use_gated_graph_combiner = True
        self.use_mixture_decoder = True
        
        # Enhanced parameters
        self.num_wave_features = 114
        self.mdn_components = 3
        self.mixture_multivariate_mode = 'independent'
        self.num_wave_patch_latents = 128    # More latents for longer sequences
        self.num_target_patch_latents = 64

def analyze_realistic_patching():
    """Analyze how multi-scale patching works with realistic sequence lengths"""
    
    print("🚀 MULTI-SCALE PATCHING WITH REALISTIC SEQUENCES")
    print("=" * 60)
    
    seq_len = 750
    pred_len = 20
    
    print(f"📊 Realistic Configuration:")
    print(f"   • Input sequence length: {seq_len} (≈3 years daily data)")
    print(f"   • Prediction length: {pred_len} (≈1 month ahead)")
    print(f"   • Ratio: {pred_len/seq_len:.3f} ({pred_len/seq_len*100:.1f}% prediction horizon)")
    
    # Generate patch configurations for realistic length
    patch_configs = PatchConfigGenerator.create_adaptive_patch_configs(seq_len)
    
    print(f"\n🔧 Multi-Scale Patch Analysis:")
    print(f"   Generated {len(patch_configs)} patch scales:")
    
    total_patches = 0
    for i, config in enumerate(patch_configs):
        patch_len = config['patch_len']
        stride = config['stride']
        num_patches = (seq_len - patch_len) // stride + 1
        total_patches += num_patches
        
        # Calculate what this represents in time
        days_per_patch = patch_len  # Assuming daily data
        overlap = patch_len - stride
        
        print(f"\n   Scale {i+1}: patch_len={patch_len}, stride={stride}")
        print(f"   → Creates {num_patches} patches")
        print(f"   → Each patch covers {days_per_patch} days")
        print(f"   → Overlap: {overlap} days ({overlap/patch_len*100:.1f}%)")
        
        if patch_len <= 7:
            scale_type = "Weekly patterns"
        elif patch_len <= 30:
            scale_type = "Monthly patterns"
        elif patch_len <= 90:
            scale_type = "Quarterly patterns"
        else:
            scale_type = "Long-term trends"
        
        print(f"   → Captures: {scale_type}")
    
    print(f"\n📈 Total patches across all scales: {total_patches}")
    print(f"   This creates a rich multi-scale representation!")
    
    return patch_configs

def test_realistic_model():
    """Test model with realistic sequence lengths"""
    
    print(f"\n🧪 TESTING MODEL WITH REALISTIC SEQUENCES")
    print("=" * 50)
    
    config = RealisticConfig()
    
    print(f"📋 Configuration:")
    print(f"   • seq_len: {config.seq_len}")
    print(f"   • pred_len: {config.pred_len}")
    print(f"   • enc_in: {config.enc_in}")
    print(f"   • c_out: {config.c_out}")
    print(f"   • d_model: {config.d_model}")
    
    try:
        print(f"\n🔧 Creating model...")
        start_time = time.time()
        model = Enhanced_SOTA_PGAT(config)
        creation_time = time.time() - start_time
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"   • Creation time: {creation_time:.2f}s")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        
        # Test forward pass
        print(f"\n🔄 Testing forward pass...")
        batch_size = 2  # Small batch for memory efficiency
        
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
        
        print(f"   • Input shapes: wave={wave_window.shape}, target={target_window.shape}")
        
        start_time = time.time()
        with torch.no_grad():
            output = model(wave_window, target_window)
        forward_time = time.time() - start_time
        
        if isinstance(output, tuple):
            print(f"✅ Forward pass successful!")
            print(f"   • Forward time: {forward_time:.2f}s")
            print(f"   • Output type: MDN (Mixture Density Network)")
            print(f"   • Output shapes: {[o.shape for o in output]}")
        else:
            print(f"✅ Forward pass successful!")
            print(f"   • Forward time: {forward_time:.2f}s")
            print(f"   • Output shape: {output.shape}")
        
        return model, True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return None, False

def test_training_step():
    """Test a few training steps with realistic data"""
    
    print(f"\n🏋️ TESTING TRAINING WITH REALISTIC SEQUENCES")
    print("=" * 50)
    
    config = RealisticConfig()
    
    try:
        model = Enhanced_SOTA_PGAT(config)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for larger model
        
        # Create more realistic synthetic data
        batch_size = 2
        
        print(f"📊 Creating synthetic financial time series...")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Sequence length: {config.seq_len}")
        print(f"   • Prediction length: {config.pred_len}")
        
        # Generate more realistic financial data patterns
        t = torch.linspace(0, config.seq_len/252, config.seq_len)  # ~3 years in trading days
        
        wave_data = []
        target_data = []
        
        for i in range(batch_size):
            # Create realistic market patterns
            trend = 0.1 * t  # Long-term trend
            seasonal = 0.05 * torch.sin(2 * torch.pi * t * 4)  # Quarterly cycles
            noise = 0.02 * torch.randn_like(t)
            
            base_price = 100 + trend + seasonal + noise
            
            # Generate OHLC + celestial features
            ohlc = torch.stack([
                base_price + 0.01 * torch.randn_like(base_price),  # Open
                base_price + 0.02 + 0.01 * torch.randn_like(base_price),  # High
                base_price - 0.02 + 0.01 * torch.randn_like(base_price),  # Low
                base_price + 0.005 * torch.randn_like(base_price),  # Close
            ], dim=-1)
            
            # Add celestial features (114 features)
            celestial = torch.randn(config.seq_len, 114) * 0.1
            
            # Combine features
            wave_sample = torch.cat([ohlc, celestial], dim=-1)
            
            # Create target (next pred_len steps)
            target_t = torch.linspace(config.seq_len/252, (config.seq_len + config.pred_len)/252, config.pred_len)
            target_trend = 0.1 * target_t
            target_seasonal = 0.05 * torch.sin(2 * torch.pi * target_t * 4)
            target_noise = 0.02 * torch.randn_like(target_t)
            target_base = 100 + target_trend + target_seasonal + target_noise
            
            target_ohlc = torch.stack([
                target_base + 0.01 * torch.randn_like(target_base),
                target_base + 0.02 + 0.01 * torch.randn_like(target_base),
                target_base - 0.02 + 0.01 * torch.randn_like(target_base),
                target_base + 0.005 * torch.randn_like(target_base),
            ], dim=-1)
            
            target_celestial = torch.randn(config.pred_len, 114) * 0.1
            target_sample = torch.cat([target_ohlc, target_celestial], dim=-1)
            
            wave_data.append(wave_sample)
            target_data.append(target_sample)
        
        wave_tensor = torch.stack(wave_data)
        target_tensor = torch.stack(target_data)
        targets = target_tensor[:, :, :config.c_out]
        
        print(f"✅ Synthetic data created")
        print(f"   • Wave tensor: {wave_tensor.shape}")
        print(f"   • Target tensor: {target_tensor.shape}")
        print(f"   • Targets for loss: {targets.shape}")
        
        # Test training steps
        print(f"\n🏃 Running training steps...")
        
        for epoch in range(3):
            start_time = time.time()
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(wave_tensor, target_tensor)
            
            # Compute loss
            if isinstance(output, tuple):
                # MDN loss
                loss = model.loss(output, targets)
            else:
                # MSE loss
                loss = nn.MSELoss()(output, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_time = time.time() - start_time
            
            print(f"   Epoch {epoch+1}: loss={loss.item():.6f}, time={epoch_time:.2f}s")
        
        print(f"✅ Training test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_configurations():
    """Compare small vs realistic sequence lengths"""
    
    print(f"\n📊 CONFIGURATION COMPARISON")
    print("=" * 40)
    
    configs = [
        ("Small (Previous Test)", 24, 6),
        ("Realistic (Your Use Case)", 750, 20)
    ]
    
    for name, seq_len, pred_len in configs:
        print(f"\n🔧 {name}:")
        print(f"   • seq_len: {seq_len}")
        print(f"   • pred_len: {pred_len}")
        print(f"   • Ratio: {pred_len/seq_len:.3f} ({pred_len/seq_len*100:.1f}%)")
        
        patch_configs = PatchConfigGenerator.create_adaptive_patch_configs(seq_len)
        total_patches = sum((seq_len - config['patch_len']) // config['stride'] + 1 
                          for config in patch_configs)
        
        print(f"   • Patch scales: {len(patch_configs)}")
        print(f"   • Total patches: {total_patches}")
        
        if seq_len >= 100:
            print(f"   • Multi-scale benefit: 🏆 HIGH (long sequences)")
        else:
            print(f"   • Multi-scale benefit: ⚠️  LOW (short sequences)")

def main():
    """Main testing function"""
    
    print("🎯 TESTING ENHANCED SOTA PGAT WITH REALISTIC SEQUENCES")
    print("=" * 70)
    
    # Analyze patching with realistic lengths
    patch_configs = analyze_realistic_patching()
    
    # Compare configurations
    compare_configurations()
    
    # Test model creation and forward pass
    model, success = test_realistic_model()
    
    if success:
        # Test training
        training_success = test_training_step()
        
        if training_success:
            print(f"\n🎉 SUCCESS! Multi-scale patching works great with realistic sequences!")
            print(f"   With seq_len=750, the model can capture:")
            print(f"   • Short-term patterns (daily/weekly)")
            print(f"   • Medium-term trends (monthly)")
            print(f"   • Long-term cycles (quarterly/seasonal)")
            print(f"   This is exactly what multi-scale patching was designed for!")
        else:
            print(f"\n⚠️  Model creation succeeded but training had issues")
    else:
        print(f"\n❌ Model creation failed with realistic sequences")

if __name__ == "__main__":
    main()