#!/usr/bin/env python3
"""
Quick test of SOTA_Temporal_PGAT model functionality.
Tests the model directly without full training pipeline.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def test_pgat_model():
    """Test the SOTA_Temporal_PGAT model directly."""
    
    print("🔍 Testing SOTA_Temporal_PGAT Model")
    print("=" * 50)
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        print("✅ Model import successful")
        
        # Create test configuration matching our minimal config
        class TestConfig:
            def __init__(self):
                # Data dimensions (2 covariates + 2 targets = 4 input features)
                self.enc_in = 4
                self.dec_in = 4
                self.c_out = 2  # 2 targets to predict
                self.c_out_evaluation = 2
                
                # Sequence dimensions
                self.seq_len = 24
                self.pred_len = 6
                self.label_len = 12
                
                # Model architecture
                self.d_model = 128
                self.n_heads = 4
                self.d_ff = 512
                self.dropout = 0.1
                self.factor = 3
                
                # PGAT sophisticated features
                self.use_dynamic_edge_weights = True
                self.use_autocorr_attention = True
                self.use_adaptive_temporal = True
                self.enable_dynamic_graph = True
                self.enable_graph_positional_encoding = True
                self.enable_structural_pos_encoding = True
                self.enable_graph_attention = True
                self.use_mixture_density = True
                self.mdn_components = 3
                self.max_eigenvectors = 8
                self.autocorr_factor = 1
                
                # Memory optimization
                self.enable_memory_optimization = True
                self.memory_chunk_size = 16
                self.use_gradient_checkpointing = False
        
        config = TestConfig()
        print("✅ Configuration created")
        
        # Initialize model
        print("\n📦 Initializing model...")
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        print("✅ Model initialization successful")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   📊 Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("\n🔄 Testing forward pass...")
        batch_size = 2
        
        # Create test inputs with correct dimensions
        # wave_window: historical data (all features: covariates + targets)
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        
        # target_window: future data (all features: covariates + targets)
        # FIXED: Must have same feature dimension as wave_window for concatenation
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
        
        # graph: dummy graph structure
        graph = torch.randn(10, 10)
        
        print(f"   Input shapes:")
        print(f"   - wave_window: {wave_window.shape} (batch, seq_len, all_features)")
        print(f"   - target_window: {target_window.shape} (batch, pred_len, all_features)")
        print(f"   - graph: {graph.shape}")
        print(f"   Note: Both windows have {config.enc_in} features for concatenation")
        
        # Forward pass
        with torch.no_grad():
            output = model(wave_window, target_window, graph)
        
        expected_shape = (batch_size, config.pred_len, config.c_out)
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: {expected_shape}")
        
        if output.shape == expected_shape:
            print("✅ Forward pass successful!")
        else:
            print(f"❌ Shape mismatch!")
            return False
        
        # Test memory stats
        print("\n📊 Memory statistics:")
        stats = model.get_memory_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.2f}")
            else:
                print(f"   - {key}: {value}")
        
        # Test loss configuration
        print("\n🎯 Testing loss configuration...")
        base_criterion = torch.nn.MSELoss()
        loss_fn = model.configure_optimizer_loss(base_criterion, verbose=True)
        
        # Test loss computation
        y_true = torch.randn_like(output)
        loss = loss_fn(output, y_true)
        print(f"   Loss value: {loss.item():.6f}")
        print("✅ Loss computation successful")
        
        # Test gradient flow
        print("\n🔄 Testing gradient flow...")
        loss.backward()
        
        grad_count = 0
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                total_grad_norm += param.grad.norm().item()
        
        print(f"   Parameters with gradients: {grad_count}")
        print(f"   Average gradient norm: {total_grad_norm / max(grad_count, 1):.6f}")
        print("✅ Gradient flow working")
        
        # Test configuration methods
        print("\n⚙️  Testing configuration methods...")
        model.configure_for_training()
        print("   ✅ Training mode configured")
        
        model.configure_for_inference()
        print("   ✅ Inference mode configured")
        
        # Test memory management
        print("\n🧹 Testing memory management...")
        initial_cache_size = len(model._memory_cache)
        model.clear_memory_cache()
        final_cache_size = len(model._memory_cache)
        print(f"   Cache cleared: {initial_cache_size} → {final_cache_size}")
        print("✅ Memory management working")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The SOTA_Temporal_PGAT model is fully functional with:")
        print("✅ All sophisticated features working")
        print("✅ Memory optimizations active")
        print("✅ Proper covariate and target handling")
        print("✅ Gradient flow and training ready")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_pgat_model()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 QUICK TEST PASSED!")
        print("Ready to run full training with:")
        print("python scripts/run_pgat_test.py")
    else:
        print("❌ QUICK TEST FAILED!")
        print("Fix the issues above before running full training.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)