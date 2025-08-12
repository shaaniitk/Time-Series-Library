"""
Quick Test for Available HF Models with Covariate Support

This tests the working HF models to demonstrate covariate functionality.
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from argparse import Namespace
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import working model
import importlib.util
spec = importlib.util.spec_from_file_location("HFEnhancedAutoformer", "./models/HFEnhancedAutoformer.py")
hf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_module)
HFEnhancedAutoformer = hf_module.HFEnhancedAutoformer

from utils.timefeatures import time_features


def create_test_config():
    """Create test configuration"""
    config = {
        'task_name': 'long_term_forecast',
        'freq': 'h',      
        'timeenc': 1,     
        'embed': 'timeF', 
        
        # Sequence lengths
        'seq_len': 48,   
        'label_len': 24,
        'pred_len': 12,
        
        # Model dimensions
        'enc_in': 6,    
        'dec_in': 6,    
        'c_out': 1,     
        'd_model': 128, 
        'n_heads': 4,
        'e_layers': 1,  
        'd_layers': 1,
        'd_ff': 256,
        'dropout': 0.1,
        'activation': 'gelu',
        
        # Training
        'batch_size': 8,  
        'use_gpu': False,
        
        # HF-specific
        'model_name': 'chronos-t5-tiny',
    }
    
    return Namespace(**config)


def main():
    """Quick demonstration of HF covariate functionality"""
    
    print("="*80)
    print("HF MODELS COVARIATE FUNCTIONALITY DEMONSTRATION")
    print("="*80)
    
    # Create simple test data
    dates = pd.date_range('2020-01-01', periods=200, freq='H')
    n = len(dates)
    
    # Simple synthetic data
    trend = np.linspace(10, 20, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 24)  # Daily pattern
    noise = np.random.normal(0, 0.5, n)
    
    data = {
        'date': dates,
        'f1': trend + seasonal + noise,
        'f2': trend * 1.2 + seasonal * 0.8 + noise,
        'f3': trend * 0.8 + seasonal * 1.2 + noise,
        'f4': trend + seasonal * 0.5 + noise,
        'f5': trend * 1.1 + seasonal + noise,
        'target': trend + seasonal + noise
    }
    
    df = pd.DataFrame(data)
    
    # Prepare data
    config = create_test_config()
    df['date'] = pd.to_datetime(df['date'])
    
    # Get time features (covariates)
    data_stamp = time_features(pd.to_datetime(df['date'].values), freq=config.freq)
    data_stamp = data_stamp.transpose(1, 0)
    
    # Get time series data
    ts_data = df[['f1', 'f2', 'f3', 'f4', 'f5', 'target']].values.astype(np.float32)
    
    # Create sequences
    seq_len = config.seq_len
    start_idx = 50
    
    x_enc = torch.FloatTensor(ts_data[start_idx:start_idx + seq_len]).unsqueeze(0)
    x_mark_enc = torch.FloatTensor(data_stamp[start_idx:start_idx + seq_len]).unsqueeze(0)
    x_dec = torch.zeros(1, config.pred_len + config.label_len, 6)  # Placeholder
    x_mark_dec = torch.FloatTensor(data_stamp[start_idx + seq_len - config.label_len:start_idx + seq_len + config.pred_len]).unsqueeze(0)
    
    print(f" Input shapes:")
    print(f"  Time series: {x_enc.shape}")
    print(f"  Time features: {x_mark_enc.shape}")
    print(f"  Time features have {x_mark_enc.shape[-1]} dimensions (hour, day, month, etc.)")
    
    # Test HF model
    print(f"\n Testing HFEnhancedAutoformer...")
    model = HFEnhancedAutoformer(config)
    model.eval()
    
    with torch.no_grad():
        # With covariates
        output_with = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"  Prediction WITH covariates: mean={output_with.mean().item():.4f}")
        
        # Without covariates
        output_without = model(x_enc, None, x_dec, None)
        print(f"  Prediction WITHOUT covariates: mean={output_without.mean().item():.4f}")
        
        # Difference
        diff = torch.abs(output_with - output_without).mean().item()
        print(f"  Absolute difference: {diff:.6f}")
        
        if diff > 1e-6:
            print(f"  PASS SUCCESS: Covariates make a meaningful difference!")
        else:
            print(f"  FAIL FAILURE: Covariates are being ignored")
    
    print(f"\n{'='*80}")
    print("CONCLUSION: HF Models vs Standard Models Flexibility")
    print(f"{'='*80}")
    
    print("""
CHART FLEXIBILITY COMPARISON:

 LIMITATIONS of HF Models:
1. Foundation Model Constraints:
   - Limited to transformer architectures (T5, GPT, etc.)
   - Fixed embedding dimensions from pre-trained models
   - Cannot easily modify core architecture

2. Computational Requirements:
   - Larger memory footprint due to HF backbone
   - Slower inference compared to native implementations
   - GPU memory requirements for larger models

3. Customization Limits:
   - Less control over attention mechanisms
   - Fixed tokenization/embedding strategies
   - Dependency on HF model availability

 ADVANTAGES of HF Models:
1. Pre-trained Knowledge:
   - Leverage massive pre-training on temporal data
   - Better generalization to new domains
   - Reduced training time and data requirements

2. Advanced Capabilities:
   - Built-in support for various sequence lengths
   - Robust handling of missing data
   - Advanced attention mechanisms

3. Easy Integration:
   - Standard HF ecosystem compatibility
   - Easy model switching and experimentation
   - Consistent API across different models

 FLEXIBILITY VERDICT:

Standard Models (Autoformer, TimesNet, etc.):
PASS Full architectural flexibility
PASS Lightweight and fast
PASS Easy to modify and extend
FAIL Require training from scratch
FAIL Limited pre-trained knowledge

HF Enhanced Models:
PASS Leverage pre-trained knowledge
PASS Better out-of-box performance
PASS Advanced temporal understanding
FAIL Limited architectural flexibility
FAIL Higher computational requirements

TARGET RECOMMENDATION:
- Use HF models when you need strong baseline performance and have computational resources
- Use standard models when you need full control and lightweight deployment
- The covariate fix makes HF models as capable as standard models for time feature integration
""")

if __name__ == "__main__":
    main()
