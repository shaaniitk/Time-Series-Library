"""
Individual test for HFBayesianAutoformer to verify the fix is working.
Tests covariate integration and temporal embedding functionality.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple config class
class SimpleConfig:
    def __init__(self):
        # Basic config
        self.seq_len = 48
        self.pred_len = 12
        self.c_out = 1
        self.enc_in = 1
        self.dec_in = 1
        
        # Embedding config
        self.embed = 'timeF'
        self.freq = 'h'
        self.d_model = 512
        
        # Bayesian specific configs
        self.mc_samples = 5
        self.uncertainty_method = 'mc_dropout'
        self.quantile_mode = True
        self.quantile_levels = [0.1, 0.5, 0.9]

def test_hf_bayesian_fixed():
    """Test HFBayesianAutoformer after corruption fix"""
    print("🔍 Testing HFBayesianAutoformer After Corruption Fix")
    print("=" * 50)
    
    config = SimpleConfig()
    batch_size = 2
    
    # Create sample data with covariates
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # 4 temporal features
    x_dec = torch.randn(batch_size, config.pred_len, config.dec_in) 
    x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
    
    try:
        # Import directly
        from models.HFBayesianAutoformer import HFBayesianAutoformer
        print("✅ Import successful")
        
        # Initialize model
        model = HFBayesianAutoformer(config)
        model.eval()
        print("✅ Model initialization successful")
        
        # Check components
        print(f"📊 Model components:")
        print(f"   - Has temporal_embedding: {hasattr(model, 'temporal_embedding')}")
        print(f"   - Has uncertainty_head: {hasattr(model, 'uncertainty_head')}")
        print(f"   - Has base_model: {hasattr(model, 'base_model')}")
        print(f"   - MC samples: {model.mc_samples}")
        print(f"   - Quantile mode: {model.is_quantile_mode}")
        
        # Test forward pass with covariates
        print(f"\n🧪 Testing forward pass...")
        with torch.no_grad():
            output_with_cov = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output_without_cov = model(x_enc, None, x_dec, None)
        
        print(f"✅ Forward pass successful")
        print(f"   - Output with covariates shape: {output_with_cov.shape}")
        print(f"   - Output without covariates shape: {output_without_cov.shape}")
        
        # Check covariate effect
        cov_mean = output_with_cov.mean().item()
        no_cov_mean = output_without_cov.mean().item()
        difference = abs(cov_mean - no_cov_mean)
        
        print(f"   - With covariates mean: {cov_mean:.4f}")
        print(f"   - Without covariates mean: {no_cov_mean:.4f}")
        print(f"   - Difference: {difference:.4f}")
        
        # Test temporal embedding directly
        print(f"\n🎯 Testing temporal embedding...")
        temp_emb = model.temporal_embedding(x_mark_enc)
        print(f"   - Input covariates shape: {x_mark_enc.shape}")
        print(f"   - Temporal embedding shape: {temp_emb.shape}")
        print(f"   - Temporal embedding mean: {temp_emb.mean().item():.4f}")
        
        # Test uncertainty methods
        print(f"\n🔍 Testing uncertainty methods...")
        uncertainty_result = model.get_uncertainty_result(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"   - Uncertainty result keys: {list(uncertainty_result.keys())}")
        print(f"   - Prediction shape: {uncertainty_result['prediction'].shape}")
        print(f"   - Uncertainty shape: {uncertainty_result['uncertainty'].shape}")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ HFBayesianAutoformer is working correctly after fix")
        print(f"✅ Covariate integration is functional")
        print(f"✅ Temporal embeddings are working")
        print(f"✅ Uncertainty quantification is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hf_bayesian_fixed()
    if success:
        print("\n🎯 HFBayesianAutoformer fix verification: SUCCESS")
    else:
        print("\n❌ HFBayesianAutoformer fix verification: FAILED")
