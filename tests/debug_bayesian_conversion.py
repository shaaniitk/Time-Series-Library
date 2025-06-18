#!/usr/bin/env python3
"""
Debug script to check Bayesian layer conversion
"""

import torch
import torch.nn as nn
from argparse import Namespace

from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
from layers.BayesianLayers import BayesianLinear

def debug_bayesian_conversion():
    """Debug the Bayesian layer conversion process"""
    
    print("🔍 Debugging Bayesian Layer Conversion")
    print("=" * 50)
    
    # Create config
    config = Namespace(
        seq_len=24, label_len=12, pred_len=6, enc_in=10, dec_in=4, c_out=4,
        d_model=32, n_heads=2, e_layers=1, d_layers=1, d_ff=64, factor=1,
        dropout=0.1, embed='timeF', freq='h', activation='gelu',
        task_name='long_term_forecast', model='EnhancedAutoformer',
        moving_avg=25, output_attention=False, distil=True, mix=True
    )
    
    # Create Bayesian model
    print("1️⃣ Creating BayesianEnhancedAutoformer...")
    model = BayesianEnhancedAutoformer(
        config, 
        uncertainty_method='bayesian',
        bayesian_layers=['projection'],
        kl_weight=1e-3
    )
    
    print("\n2️⃣ Checking model structure...")
    print(f"Base model type: {type(model.base_model)}")
    print(f"Has projection layer: {hasattr(model.base_model, 'projection')}")
    
    if hasattr(model.base_model, 'projection'):
        proj = model.base_model.projection
        print(f"Projection layer type: {type(proj)}")
        print(f"Is Bayesian: {isinstance(proj, BayesianLinear)}")
        
        if isinstance(proj, BayesianLinear):
            print(f"Projection shape: {proj.in_features} -> {proj.out_features}")
        else:
            print(f"Projection shape: {proj.in_features} -> {proj.out_features}")
    
    print("\n3️⃣ Scanning for all Bayesian layers...")
    bayesian_layers = []
    for name, module in model.named_modules():
        if isinstance(module, BayesianLinear):
            bayesian_layers.append(name)
            print(f"   Found Bayesian layer: {name}")
    
    print(f"Total Bayesian layers found: {len(bayesian_layers)}")
    
    print("\n4️⃣ Testing KL divergence computation...")
    total_kl = model.get_kl_loss()
    print(f"Total KL loss: {total_kl}")
    
    if len(bayesian_layers) > 0:
        # Test individual layer KL
        for name in bayesian_layers:
            layer = dict(model.named_modules())[name]
            layer_kl = layer.kl_divergence()
            print(f"   {name} KL: {layer_kl.item():.6f}")
    else:
        print("❌ No Bayesian layers found! Conversion failed.")
        
        # Let's manually check what layers exist
        print("\n🔍 Manual inspection of model layers:")
        for name, module in model.base_model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"   Linear layer: {name} - {module.in_features}->{module.out_features}")
    
    print("\n5️⃣ Testing with manual Bayesian layer...")
    # Create a standalone Bayesian layer to verify KL computation works
    manual_bayesian = BayesianLinear(32, 4, prior_std=1.0)
    manual_kl = manual_bayesian.kl_divergence()
    print(f"Manual BayesianLinear KL: {manual_kl.item():.6f}")
    
    # Force some parameters to be non-zero
    with torch.no_grad():
        manual_bayesian.weight_mean.data.fill_(0.5)  # Set means to 0.5
        manual_bayesian.weight_logvar.data.fill_(-2.0)  # Set logvar to -2 (std ≈ 0.37)
    
    manual_kl_after = manual_bayesian.kl_divergence()
    print(f"Manual BayesianLinear KL (after setting params): {manual_kl_after.item():.6f}")
    
    return len(bayesian_layers) > 0

if __name__ == "__main__":
    success = debug_bayesian_conversion()
    if success:
        print("\n✅ Bayesian conversion working!")
    else:
        print("\n❌ Bayesian conversion failed!")
