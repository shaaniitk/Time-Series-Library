#!/usr/bin/env python3
"""
Test MoE compatibility with the migrated Enhanced Autoformer.
"""
import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockMoEConfig:
    """Mock configuration for MoE testing."""
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.c_out_evaluation = 7
        self.d_model = 512
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 'h'
        self.norm_type = 'LayerNorm'
        self.factor = 1
        self.top_k = 5
        self.n_heads = 8
        self.moving_avg = 25
        self.autocorr = True

class MockMoELayer(torch.nn.Module):
    """Mock MoE layer that returns 3-tuple with auxiliary loss."""
    def __init__(self, d_model, c_out):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Simulate MoE layer output: (output, residual_trend, aux_loss)
        batch_size, seq_len, _ = x.shape
        output = torch.randn_like(x)
        residual_trend = torch.randn(batch_size, seq_len, self.c_out)
        aux_loss = torch.tensor(0.1)  # Mock auxiliary loss
        return output, residual_trend, aux_loss

def test_moe_compatibility():
    """Test MoE compatibility with the Enhanced Autoformer."""
    print("Testing MoE compatibility...")
    
    try:
        from models.EnhancedAutoformer import EnhancedDecoder, LearnableSeriesDecomp
        
        # Test EnhancedDecoder with mock MoE layers
        configs = MockMoEConfig()
        
        # Create mock MoE layers
        mock_layers = [MockMoELayer(configs.d_model, configs.c_out_evaluation) for _ in range(2)]
        
        # Create decoder with MoE support
        decoder = EnhancedDecoder(
            layers=mock_layers,
            c_out=configs.c_out_evaluation,
            use_moe_ffn=True
        )
        
        # Test input
        batch_size = 4
        seq_len = configs.label_len + configs.pred_len
        x = torch.randn(batch_size, seq_len, configs.d_model)
        cross = torch.randn(batch_size, configs.seq_len, configs.d_model)
        trend = torch.randn(batch_size, seq_len, configs.c_out_evaluation)
        
        # Test decoder forward with MoE layers
        with torch.no_grad():
            result = decoder(x, cross, trend=trend)
            
        # Should return 3-tuple: (seasonal, trend, aux_loss)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        seasonal, final_trend, total_aux_loss = result
        
        assert seasonal.shape == (batch_size, seq_len, configs.d_model), f"Seasonal shape: {seasonal.shape}"
        assert final_trend.shape == (batch_size, seq_len, configs.c_out_evaluation), f"Trend shape: {final_trend.shape}"
        assert isinstance(total_aux_loss, torch.Tensor), f"Aux loss type: {type(total_aux_loss)}"
        assert total_aux_loss.item() > 0, f"Aux loss should be positive: {total_aux_loss.item()}"
        
        print("✅ MoE auxiliary loss handling working correctly")
        print(f"   - Seasonal output shape: {seasonal.shape}")
        print(f"   - Trend output shape: {final_trend.shape}")
        print(f"   - Total auxiliary loss: {total_aux_loss.item():.4f}")
        
        # Test LearnableSeriesDecomp with different input sizes
        for input_dim in [7, 20, 118]:  # Common dimensions in the codebase
            decomp = LearnableSeriesDecomp(input_dim=input_dim)
            test_input = torch.randn(32, 96, input_dim)
            
            with torch.no_grad():
                seasonal, trend = decomp(test_input)
                
            assert seasonal.shape == test_input.shape
            assert trend.shape == test_input.shape
            print(f"✅ LearnableSeriesDecomp working for {input_dim}D input")
        
        print("🎉 All MoE compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MoE test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_moe_compatibility()
    sys.exit(0 if success else 1)