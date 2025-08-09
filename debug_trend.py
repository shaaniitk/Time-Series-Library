#!/usr/bin/env python3
"""
Debug the trend dimension issue
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from unittest.mock import Mock

def debug_trend_dimensions():
    print("Debugging trend dimension issue...")
    
    # Mock config for testing
    configs = Mock()
    configs.task_name = 'long_term_forecast'
    configs.seq_len = 96
    configs.label_len = 48
    configs.pred_len = 96
    configs.enc_in = 7
    configs.dec_in = 7
    configs.c_out = 7
    configs.d_model = 512
    configs.d_ff = 2048
    configs.n_heads = 8
    configs.e_layers = 2
    configs.d_layers = 1
    configs.factor = 1
    configs.dropout = 0.1
    configs.embed = 'timeF'
    configs.freq = 'h'
    configs.activation = 'gelu'
    configs.norm_type = 'LayerNorm'
    configs.c_out_evaluation = 7
    
    try:
        # Force reload of the module
        import importlib
        if 'models.EnhancedAutoformer' in sys.modules:
            importlib.reload(sys.modules['models.EnhancedAutoformer'])
        
        from models.EnhancedAutoformer import EnhancedAutoformer
        print("DEBUG: Successfully imported EnhancedAutoformer")
        
        print("DEBUG: About to create model...")
        model = EnhancedAutoformer(configs)
        print("DEBUG: Model created successfully")
        print(f"Decoder type: {type(model.decoder)}")
        print(f"Decoder class name: {model.decoder.__class__.__name__}")
        print(f"Decoder module: {model.decoder.__class__.__module__}")
        
        # Check model configuration
        print(f"Model d_model: {model.d_model}")
        print(f"Model c_out: {model.c_out}")
        print(f"Model num_target_variables: {model.num_target_variables}")
        
        # Check if trend_projection exists
        print(f"Has trend_projection: {hasattr(model.decoder, 'trend_projection')}")
        if hasattr(model.decoder, 'trend_projection'):
            print(f"Trend projection: {model.decoder.trend_projection}")
        
        # Print decoder layer info
        print(f"Number of decoder layers: {len(model.decoder.layers)}")
        if len(model.decoder.layers) > 0:
            first_layer = model.decoder.layers[0]
            print(f"First layer decomp1 input_dim: {first_layer.decomp1.input_dim}")
            
        if hasattr(model.decoder, 'c_out'):
            print(f"Decoder c_out: {model.decoder.c_out}")
        else:
            print("Decoder doesn't have c_out attribute!")
            
        if hasattr(model.decoder, 'd_model'):
            print(f"Decoder d_model: {model.decoder.d_model}")
        else:
            print("Decoder doesn't have d_model attribute!")
            
        # Let's manually test the dimension issue
        B, L = 2, 144  # label_len + pred_len
        trend_arg = torch.randn(B, L, 7)  # c_out dimensions
        
        # Create a fake residual_trend from decoder layer
        residual_trend = torch.randn(B, L, 512)  # d_model dimensions
        
        print(f"Input trend shape: {trend_arg.shape}")
        print(f"Residual trend shape: {residual_trend.shape}")
        
        # Try to project manually
        if hasattr(model.decoder, 'trend_projection'):
            projected_trend = model.decoder.trend_projection(residual_trend)
            print(f"Projected trend shape: {projected_trend.shape}")
            
            # Test addition
            result = trend_arg + projected_trend
            print(f"Addition result shape: {result.shape}")
        else:
            print("No trend_projection found!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_trend_dimensions()
