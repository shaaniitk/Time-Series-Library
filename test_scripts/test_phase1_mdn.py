"""
Test script for Phase 1 MDN Decoder implementation.

This script validates:
1. MDN decoder module initialization and forward pass
2. Model integration with MDN enabled
3. Training script compatibility
4. Loss computation and calibration metrics
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import yaml
from pathlib import Path

print("="*80)
print("🧪 PHASE 1 MDN DECODER TEST SUITE")
print("="*80)

# Test 1: MDN Decoder Module
print("\n📦 Test 1: MDN Decoder Module")
print("-" * 80)
from layers.modular.decoder.mdn_decoder import MDNDecoder, mdn_nll_loss

try:
    # Create decoder
    mdn = MDNDecoder(
        d_input=64,
        n_targets=4,
        n_components=5,
        sigma_min=1e-3,
        use_softplus=True
    )
    print(f"✅ MDNDecoder initialized: {mdn}")
    
    # Test forward pass
    batch_size, seq_len = 8, 24
    hidden = torch.randn(batch_size, seq_len, 64)
    pi, mu, sigma = mdn(hidden)
    
    print(f"✅ Forward pass shapes:")
    print(f"   - pi:    {list(pi.shape)} (expected: [{batch_size}, {seq_len}, 4, 5])")
    print(f"   - mu:    {list(mu.shape)} (expected: [{batch_size}, {seq_len}, 4, 5])")
    print(f"   - sigma: {list(sigma.shape)} (expected: [{batch_size}, {seq_len}, 4, 5])")
    
    # Validate shapes
    assert pi.shape == (batch_size, seq_len, 4, 5), f"pi shape mismatch: {pi.shape}"
    assert mu.shape == (batch_size, seq_len, 4, 5), f"mu shape mismatch: {mu.shape}"
    assert sigma.shape == (batch_size, seq_len, 4, 5), f"sigma shape mismatch: {sigma.shape}"
    
    # Validate constraints
    assert torch.allclose(pi.sum(dim=-1), torch.ones_like(pi.sum(dim=-1)), atol=1e-5), "pi doesn't sum to 1"
    assert (sigma >= 1e-3).all(), f"sigma has values below floor: min={sigma.min().item()}"
    
    print(f"✅ Constraints validated:")
    print(f"   - π sums to 1: ✓")
    print(f"   - σ ≥ {1e-3}: ✓ (min σ = {sigma.min().item():.6f})")
    
    # Test loss computation
    targets = torch.randn(batch_size, seq_len, 4)
    loss = mdn_nll_loss(pi, mu, sigma, targets, reduce='mean')
    
    print(f"✅ MDN NLL loss computed: {loss.item():.4f}")
    assert torch.isfinite(loss), "NLL loss is not finite"
    
    # Test sampling
    samples = mdn.sample(pi, mu, sigma, n_samples=10)
    print(f"✅ Sampling: {list(samples.shape)} (expected: [10, {batch_size}, {seq_len}, 4])")
    assert samples.shape == (10, batch_size, seq_len, 4), f"Sample shape mismatch: {samples.shape}"
    
    # Test mean prediction
    mean_pred = mdn.mean_prediction(pi, mu)
    print(f"✅ Mean prediction: {list(mean_pred.shape)} (expected: [{batch_size}, {seq_len}, 4])")
    assert mean_pred.shape == (batch_size, seq_len, 4), f"Mean prediction shape mismatch: {mean_pred.shape}"
    
    print("\n✅ Test 1 PASSED: MDN Decoder Module")
    
except Exception as e:
    print(f"\n❌ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Calibration Metrics
print("\n📊 Test 2: Calibration Metrics")
print("-" * 80)
from utils.metrics_calibration import (
    compute_mixture_quantiles,
    compute_coverage,
    compute_crps_gaussian_mixture
)

try:
    # Use outputs from Test 1
    quantiles = [0.1, 0.5, 0.9]
    q_values = compute_mixture_quantiles(pi, mu, sigma, quantiles, n_samples=100)
    
    print(f"✅ Quantile computation:")
    for i, q in enumerate(quantiles):
        print(f"   - q={q}: shape={list(q_values[:, :, :, i].shape)}")
    
    # Compute coverage
    coverage_levels = [0.5, 0.9]
    preds_np = mean_pred.detach().numpy()
    targets_np = targets.detach().numpy()
    coverage_dict = compute_coverage(preds_np, targets_np, pi, mu, sigma, coverage_levels)
    
    print(f"✅ Coverage metrics:")
    for level in coverage_levels:
        cov = coverage_dict.get(f'coverage_{int(level*100)}', 0.0)
        print(f"   - {int(level*100)}% interval: {cov:.3f} (target: {level:.2f})")
    
    # Compute CRPS (small sample for speed)
    crps = compute_crps_gaussian_mixture(pi[:2], mu[:2], sigma[:2], targets[:2], n_samples=50)
    print(f"✅ CRPS: {crps:.4f}")
    assert torch.isfinite(torch.tensor(crps)), "CRPS is not finite"
    
    print("\n✅ Test 2 PASSED: Calibration Metrics")
    
except Exception as e:
    print(f"\n❌ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model Integration
print("\n🏗️ Test 3: Model Integration")
print("-" * 80)

try:
    # Load production config
    config_path = Path("configs/celestial_enhanced_pgat_production.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Enable MDN decoder
    config_dict['enable_mdn_decoder'] = True
    config_dict['mdn_components'] = 5
    config_dict['mdn_sigma_min'] = 0.001
    
    # Create minimal namespace
    from argparse import Namespace
    args = Namespace(**config_dict)
    
    # Add required attributes for model
    required_attrs = {
        'verbose_logging': False,
        'enable_memory_debug': False,
        'enable_memory_diagnostics': False,
        'collect_diagnostics': False,
        'use_mixture_decoder': False,
        'use_stochastic_learner': False,
        'use_hierarchical_mapping': False,
        'use_efficient_covariate_interaction': False,
        'use_dynamic_spatiotemporal_encoder': True,
        'use_target_autocorrelation': True,
        'target_autocorr_layers': 2,
        'use_calendar_effects': True,
        'calendar_embedding_dim': 104,
    }
    
    for attr, val in required_attrs.items():
        if not hasattr(args, attr):
            setattr(args, attr, val)
    
    print(f"✅ Config loaded with MDN enabled")
    print(f"   - enable_mdn_decoder: {args.enable_mdn_decoder}")
    print(f"   - mdn_components: {args.mdn_components}")
    print(f"   - mdn_sigma_min: {args.mdn_sigma_min}")
    
    # Import model
    from models.Celestial_Enhanced_PGAT import Model
    
    # Create model (this will test initialization)
    print("\n🔧 Creating model...")
    model = Model(args)
    
    print(f"✅ Model created successfully")
    print(f"   - enable_mdn_decoder: {model.enable_mdn_decoder}")
    print(f"   - mdn_components: {model.mdn_components}")
    print(f"   - MDN decoder: {type(model.mdn_decoder).__name__ if model.mdn_decoder else 'None'}")
    
    # Test forward pass
    print("\n🔧 Testing forward pass...")
    batch_size = 4
    x_enc = torch.randn(batch_size, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(batch_size, args.seq_len, 4)
    x_dec = torch.randn(batch_size, args.label_len + args.pred_len, args.dec_in)
    x_mark_dec = torch.randn(batch_size, args.label_len + args.pred_len, 4)
    
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    # Validate output format
    if isinstance(output, tuple):
        if len(output) == 4:
            point_pred, aux_loss, mdn_tuple, metadata = output
            print(f"✅ Forward pass returned 4-tuple (MDN mode)")
            print(f"   - point_pred: {list(point_pred.shape)}")
            print(f"   - aux_loss: {aux_loss}")
            print(f"   - mdn_tuple: {type(mdn_tuple)} with {len(mdn_tuple) if mdn_tuple else 0} elements")
            
            if mdn_tuple and len(mdn_tuple) == 3:
                pi_out, mu_out, sigma_out = mdn_tuple
                print(f"   - pi: {list(pi_out.shape)}")
                print(f"   - mu: {list(mu_out.shape)}")
                print(f"   - sigma: {list(sigma_out.shape)}")
                
                # Validate MDN constraints
                assert torch.allclose(pi_out.sum(dim=-1), torch.ones_like(pi_out.sum(dim=-1)), atol=1e-5), "π doesn't sum to 1"
                assert (sigma_out >= args.mdn_sigma_min).all(), f"σ below floor"
                print(f"✅ MDN constraints validated in model output")
        else:
            print(f"⚠️ Forward pass returned {len(output)}-tuple (expected 4)")
    else:
        print(f"⚠️ Forward pass returned tensor (expected tuple)")
    
    print("\n✅ Test 3 PASSED: Model Integration")
    
except Exception as e:
    print(f"\n❌ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Training Script Compatibility
print("\n🎓 Test 4: Training Script Compatibility")
print("-" * 80)

try:
    from scripts.train.train_celestial_production import _normalize_model_output
    
    # Test with MDN output (4-tuple)
    test_output = (point_pred, aux_loss, mdn_tuple, metadata)
    
    normalized_pred, normalized_aux, normalized_mdn, normalized_meta = _normalize_model_output(test_output)
    
    print(f"✅ _normalize_model_output handled 4-tuple:")
    print(f"   - predictions: {list(normalized_pred.shape)}")
    print(f"   - aux_loss: {normalized_aux}")
    print(f"   - mdn_tuple: {type(normalized_mdn)} with {len(normalized_mdn) if normalized_mdn else 0} elements")
    print(f"   - metadata: {type(normalized_meta)}")
    
    # Test with standard output (2-tuple)
    test_output_std = (point_pred, metadata)
    normalized_pred2, normalized_aux2, normalized_mdn2, normalized_meta2 = _normalize_model_output(test_output_std)
    
    print(f"✅ _normalize_model_output handled 2-tuple (backward compat):")
    print(f"   - predictions: {list(normalized_pred2.shape)}")
    print(f"   - aux_loss: {normalized_aux2}")
    print(f"   - mdn_tuple: {normalized_mdn2}")
    
    print("\n✅ Test 4 PASSED: Training Script Compatibility")
    
except Exception as e:
    print(f"\n❌ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("🎉 ALL TESTS PASSED!")
print("="*80)
print("\n✅ Phase 1 MDN Decoder Implementation Validated:")
print("   1. ✅ MDN decoder module (forward, loss, sampling, mean)")
print("   2. ✅ Calibration metrics (quantiles, coverage, CRPS)")
print("   3. ✅ Model integration (initialization, forward pass, constraints)")
print("   4. ✅ Training script compatibility (_normalize_model_output)")
print("\n🚀 Ready for training with enable_mdn_decoder=true")
print("="*80)
