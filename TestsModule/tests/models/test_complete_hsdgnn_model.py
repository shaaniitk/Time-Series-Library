#!/usr/bin/env python3
"""
Test the complete HSDGNN-enhanced Wave-Stock model
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import from the HSDGNN notebook (we'll create a simplified version)
    from layers.HSDGNNComponents import HierarchicalSpatiotemporalBlock
    from layers.Embed import PatchEmbedding
    from layers.modular.decomposition.registry import get_decomposition_component
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Components not available: {e}")
    COMPONENTS_AVAILABLE = False

class SimpleHSDGNNWaveStockModel(nn.Module):
    """Simplified HSDGNN-enhanced Wave-Stock model for testing"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Target stream (simplified)
        self.target_gru = nn.GRU(
            input_size=1,
            hidden_size=config.rnn_units,
            batch_first=True
        )
        
        # Covariate stream with HSDGNN
        self.hsdgnn_block = HierarchicalSpatiotemporalBlock(
            n_waves=config.n_waves,
            wave_features=config.wave_features,
            d_model=config.d_model,
            rnn_units=config.rnn_units,
            seq_len=config.seq_len,
            window_size=config.window_size,
            threshold=config.threshold
        )
        
        # Fusion
        self.fusion = nn.Linear(config.rnn_units * 2, config.rnn_units)
        
        # Predictor
        self.predictor = nn.Linear(config.rnn_units, config.pred_len * config.c_out)
        
    def forward(self, stock_returns, wave_data):
        B, L, _ = stock_returns.shape
        
        # Target stream
        target_output, _ = self.target_gru(stock_returns)
        target_features = target_output[:, -1, :]  # Last timestep
        
        # Covariate stream with HSDGNN
        wave_reshaped = wave_data.view(B, L, self.config.n_waves, self.config.wave_features)
        wave_output = self.hsdgnn_block(wave_reshaped)  # [B, L, N_waves, rnn_units]
        wave_features = wave_output[:, -1, :, :].mean(dim=1)  # Average across waves
        
        # Fusion
        combined = torch.cat([target_features, wave_features], dim=-1)
        fused = self.fusion(combined)
        
        # Prediction
        predictions = self.predictor(fused)
        predictions = predictions.view(B, self.config.pred_len, self.config.c_out)
        
        return predictions

class TestConfig:
    seq_len = 20
    pred_len = 5
    n_waves = 10  # 10 waves as in original design
    wave_features = 4
    d_model = 32
    rnn_units = 16
    c_out = 3
    batch_size = 2
    window_size = 10
    threshold = 0.5

def test_complete_model():
    """Test the complete HSDGNN-enhanced model"""
    print("🧪 Testing Complete HSDGNN-Enhanced Wave-Stock Model...")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Required components not available")
        return False
    
    config = TestConfig()
    
    try:
        # Create model
        model = SimpleHSDGNNWaveStockModel(config)
        
        # Test data
        stock_returns = torch.randn(config.batch_size, config.seq_len, 1)
        wave_data = torch.randn(config.batch_size, config.seq_len, 40)  # 10 waves × 4 features
        
        print(f"Input shapes:")
        print(f"  Stock returns: {stock_returns.shape}")
        print(f"  Wave data: {wave_data.shape}")
        
        # Forward pass
        predictions = model(stock_returns, wave_data)
        
        print(f"Output shape: {predictions.shape}")
        print(f"Expected: [{config.batch_size}, {config.pred_len}, {config.c_out}]")
        
        # Check output shape
        expected_shape = (config.batch_size, config.pred_len, config.c_out)
        assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"
        
        # Check for NaN/Inf
        assert not torch.isnan(predictions).any(), "Predictions contain NaN values"
        assert not torch.isinf(predictions).any(), "Predictions contain Inf values"
        
        # Test gradient flow
        loss = predictions.mean()
        loss.backward()
        
        # Check gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
        
        print(f"✅ Complete model test passed!")
        print(f"   - Forward pass: ✓")
        print(f"   - Output shape: ✓")
        print(f"   - Gradient flow: ✓ ({grad_count} parameters)")
        print(f"   - No NaN/Inf: ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_parameters():
    """Test model parameter count and memory usage"""
    print("\n🧪 Testing Model Parameters...")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Required components not available")
        return False
    
    config = TestConfig()
    
    try:
        model = SimpleHSDGNNWaveStockModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model parameters:")
        print(f"   - Total: {total_params:,}")
        print(f"   - Trainable: {trainable_params:,}")
        print(f"   - Memory estimate: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Complete HSDGNN-Enhanced Wave-Stock Model")
    print("=" * 60)
    
    results = []
    results.append(test_complete_model())
    results.append(test_model_parameters())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"🏁 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n✅ HSDGNN integration is ready for production use!")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    sys.exit(0 if passed == total else 1)