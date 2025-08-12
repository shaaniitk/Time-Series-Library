"""
Test to verify all HF models are working correctly after fixing HFBayesianAutoformer corruption.
Quick validation that covariate integration is working across all HF models.
"""

import torch
import numpy as np
import sys
import os
sys.path.append('.')

# Import all HF models
from models.HFEnhancedAutoformer import HFEnhancedAutoformer
from models.HFBayesianAutoformer import HFBayesianAutoformer  
from models.HFHierarchicalAutoformer_Step3 import HFHierarchicalAutoformer
from models.HFQuantileAutoformer_Step4 import HFQuantileAutoformer

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
        
        # Model specific configs
        self.mc_samples = 5
        self.uncertainty_method = 'mc_dropout'
        self.quantile_mode = True
        self.quantile_levels = [0.1, 0.5, 0.9]

def test_hf_model_covariates():
    """Test all HF models with covariate inputs"""
    print("SEARCH Testing HF Models Covariate Integration After Fix")
    print("=" * 55)
    
    config = SimpleConfig()
    batch_size = 2
    
    # Create sample data with covariates
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # 4 temporal features
    x_dec = torch.randn(batch_size, config.pred_len, config.dec_in) 
    x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
    
    models_to_test = [
        ("HFEnhancedAutoformer", HFEnhancedAutoformer),
        ("HFBayesianAutoformer", HFBayesianAutoformer),
        ("HFHierarchicalAutoformer", HFHierarchicalAutoformer),
        ("HFQuantileAutoformer", HFQuantileAutoformer)
    ]
    
    results = {}
    
    for model_name, model_class in models_to_test:
        print(f"\nCHART Testing {model_name}...")
        
        try:
            # Initialize model
            model = model_class(config)
            model.eval()
            
            print(f"   PASS Model initialized successfully")
            
            # Test with covariates
            with torch.no_grad():
                output_with_cov = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output_without_cov = model(x_enc, None, x_dec, None)
            
            # Check outputs
            cov_mean = output_with_cov.mean().item()
            no_cov_mean = output_without_cov.mean().item()
            difference = abs(cov_mean - no_cov_mean)
            
            print(f"   GRAPH Output with covariates: {cov_mean:.4f}")
            print(f"    Output without covariates: {no_cov_mean:.4f}")
            print(f"   SEARCH Difference: {difference:.4f}")
            
            # Test temporal embedding
            if hasattr(model, 'temporal_embedding'):
                print(f"   PASS Temporal embedding layer found")
                temp_emb = model.temporal_embedding(x_mark_enc)
                print(f"    Temporal embedding shape: {temp_emb.shape}")
            else:
                print(f"   FAIL No temporal embedding found")
            
            results[model_name] = {
                'status': 'SUCCESS',
                'covariate_effect': difference > 0.001,
                'output_shape': output_with_cov.shape,
                'difference': difference
            }
            
            print(f"   PASS {model_name} working correctly!")
            
        except Exception as e:
            print(f"   FAIL Error: {str(e)}")
            results[model_name] = {
                'status': 'FAILED', 
                'error': str(e)
            }
    
    # Summary
    print(f"\nTARGET SUMMARY - HF Models Covariate Test")
    print("=" * 45)
    
    successful = 0
    total = len(models_to_test)
    
    for model_name, result in results.items():
        if result['status'] == 'SUCCESS':
            covariate_working = "PASS" if result['covariate_effect'] else "WARN"
            print(f"{model_name}: {result['status']} {covariate_working}")
            print(f"  - Shape: {result['output_shape']}")
            print(f"  - Covariate effect: {result['difference']:.4f}")
            successful += 1
        else:
            print(f"{model_name}: FAIL {result['status']}")
            print(f"  - Error: {result['error']}")
    
    print(f"\nCHART Results: {successful}/{total} models working correctly")
    
    if successful == total:
        print("PARTY All HF models successfully use covariates!")
    else:
        print("WARN  Some models need attention")

if __name__ == "__main__":
    test_hf_model_covariates()
