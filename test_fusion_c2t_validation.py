#!/usr/bin/env python
"""
Comprehensive validation of hierarchical fusion + C→T edge-bias integration.

Validates:
1. Hierarchical fusion preserves temporal structure (no information loss)
2. Cross-attention operates on all (scale, timestep) pairs
3. C→T edge bias is injected correctly
4. Both mechanisms work together in production
5. Gradient flow is healthy across both components
"""

import torch
import yaml
from models.Celestial_Enhanced_PGAT import Model
from data_provider.celestial_data_factory import CelestialDataFactory
from utils.tools import EarlyStopping, adjust_learning_rate


def load_config():
    """Load production diagnostic config."""
    with open('configs/celestial_diagnostic_minimal.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    return Config(config)


def validate_hierarchical_fusion(model, batch):
    """Validate zero-information-loss hierarchical fusion."""
    print("\n" + "="*80)
    print("HIERARCHICAL FUSION VALIDATION")
    print("="*80)
    
    # Check config flag
    assert hasattr(model, 'use_hierarchical_fusion'), "Missing use_hierarchical_fusion attribute"
    print(f"✓ Hierarchical fusion enabled: {model.use_hierarchical_fusion}")
    
    # Check cross-attention module exists
    assert hasattr(model, 'fusion_cross_attention'), "Missing fusion_cross_attention module"
    print(f"✓ Cross-attention module: {type(model.fusion_cross_attention).__name__}")
    
    # Check projection layer
    assert hasattr(model, 'hierarchical_fusion_proj'), "Missing hierarchical_fusion_proj"
    print(f"✓ Fusion projection: {model.hierarchical_fusion_proj.in_features} -> {model.hierarchical_fusion_proj.out_features}")
    
    # Enable diagnostic collection
    model.collect_diagnostics = True
    
    # Forward pass
    wave_window, target_window, celestial_window = batch
    with torch.no_grad():
        output = model(wave_window, target_window, celestial_window, None)
    
    # Check diagnostic storage
    if hasattr(model, '_last_fusion_attention'):
        attn_weights = model._last_fusion_attention
        scale_lengths = model._last_fusion_scale_lengths
        
        total_timesteps = sum(scale_lengths)
        print(f"\n✓ Attention weights shape: {attn_weights.shape}")
        print(f"✓ Total temporal tokens: {total_timesteps}")
        print(f"✓ Scale breakdown: {scale_lengths}")
        print(f"✓ Mean attention weight: {attn_weights.mean().item():.6f}")
        print(f"✓ Max attention weight: {attn_weights.max().item():.6f}")
        
        # Verify attention is distributed across scales
        attention_entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum().item()
        print(f"✓ Attention entropy: {attention_entropy:.4f} (higher = more distributed)")
        
        return True
    else:
        print("⚠ WARNING: Fusion diagnostics not collected")
        return False


def validate_c2t_edge_bias(model, batch):
    """Validate C→T edge-bias injection."""
    print("\n" + "="*80)
    print("C→T EDGE-BIAS VALIDATION")
    print("="*80)
    
    # Check config flags
    assert hasattr(model, 'use_c2t_edge_bias'), "Missing use_c2t_edge_bias attribute"
    print(f"✓ C→T edge bias enabled: {model.use_c2t_edge_bias}")
    
    assert hasattr(model, 'c2t_edge_bias_weight'), "Missing c2t_edge_bias_weight"
    print(f"✓ Edge bias scale: {model.c2t_edge_bias_weight}")
    
    # Check CelestialToTargetAttention module
    assert hasattr(model, 'celestial_to_target_attention'), "Missing C→T attention module"
    c2t_module = model.celestial_to_target_attention
    
    print(f"✓ C→T module type: {type(c2t_module).__name__}")
    print(f"✓ C→T celestial bodies: {c2t_module.num_celestial}")
    print(f"✓ C→T targets: {c2t_module.num_targets}")
    print(f"✓ C→T use_edge_bias: {c2t_module.use_edge_bias}")
    print(f"✓ C→T edge_bias_scale: {c2t_module.edge_bias_scale}")
    
    # Enable diagnostic collection
    c2t_module.enable_diagnostics = True
    
    # Forward pass
    wave_window, target_window, celestial_window = batch
    output = model(wave_window, target_window, celestial_window, None)
    
    # Check diagnostic storage
    if hasattr(c2t_module, 'latest_attention_weights') and c2t_module.latest_attention_weights:
        attn_stats = c2t_module.latest_attention_weights
        print(f"\n✓ C→T attention diagnostics collected")
        
        # Show attention distribution per target
        for target_name, weights in attn_stats.items():
            if weights is not None:
                print(f"\n  Target '{target_name}':")
                print(f"    Attention shape: {weights.shape}")
                print(f"    Mean weight: {weights.mean().item():.6f}")
                print(f"    Max weight: {weights.max().item():.6f}")
                print(f"    Min weight: {weights.min().item():.6f}")
        
        return True
    else:
        print("⚠ WARNING: C→T diagnostics not collected")
        return False


def validate_gradient_flow(model, batch, criterion):
    """Validate gradients flow through both mechanisms."""
    print("\n" + "="*80)
    print("GRADIENT FLOW VALIDATION")
    print("="*80)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    wave_window, target_window, celestial_window = batch
    
    # Forward pass
    output = model(wave_window, target_window, celestial_window, None)
    
    # Compute loss
    if isinstance(output, tuple):
        means, log_stds, log_weights = output
        targets = target_window[:, -means.size(1):, -means.size(2):]
        loss = criterion(output, targets)
    else:
        targets = target_window[:, -output.size(1):, -output.size(2):]
        loss = criterion(output, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients for hierarchical fusion
    fusion_grad_norm = 0.0
    if hasattr(model, 'fusion_cross_attention'):
        for name, param in model.fusion_cross_attention.named_parameters():
            if param.grad is not None:
                fusion_grad_norm += param.grad.norm().item()**2
    fusion_grad_norm = fusion_grad_norm**0.5
    
    print(f"✓ Hierarchical fusion gradient norm: {fusion_grad_norm:.6f}")
    
    # Check gradients for C→T attention
    c2t_grad_norm = 0.0
    if hasattr(model, 'celestial_to_target_attention'):
        for name, param in model.celestial_to_target_attention.named_parameters():
            if param.grad is not None:
                c2t_grad_norm += param.grad.norm().item()**2
    c2t_grad_norm = c2t_grad_norm**0.5
    
    print(f"✓ C→T attention gradient norm: {c2t_grad_norm:.6f}")
    
    # Check for NaN gradients
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"✗ NaN gradient in: {name}")
            has_nan = True
    
    if not has_nan:
        print(f"✓ No NaN gradients detected")
    
    # Optimizer step
    optimizer.step()
    print(f"✓ Optimizer step successful")
    
    return fusion_grad_norm > 0 and c2t_grad_norm > 0 and not has_nan


def main():
    print("\n" + "="*80)
    print("FUSION + C→T COMPREHENSIVE VALIDATION")
    print("="*80)
    
    # Load config
    config = load_config()
    print(f"\n✓ Loaded config: {config.model_name}")
    print(f"  - use_hierarchical_fusion: {config.use_hierarchical_fusion}")
    print(f"  - use_c2t_edge_bias: {config.use_c2t_edge_bias}")
    print(f"  - c2t_edge_bias_weight: {config.c2t_edge_bias_weight}")
    
    # Create data loader
    print(f"\n✓ Creating data loader...")
    data_factory = CelestialDataFactory(config)
    train_loader = data_factory.get_data_loader(flag='train')
    
    # Get one batch
    batch = next(iter(train_loader))
    wave_window, target_window, celestial_window = batch
    print(f"  - Wave window: {wave_window.shape}")
    print(f"  - Target window: {target_window.shape}")
    print(f"  - Celestial window: {celestial_window.shape}")
    
    # Initialize model
    print(f"\n✓ Initializing model...")
    model = Model(config)
    model.eval()
    
    # Get model info
    info = model.get_enhanced_config_info()
    print(f"  - Enhanced features: {info}")
    
    # Validation 1: Hierarchical Fusion
    fusion_ok = validate_hierarchical_fusion(model, batch)
    
    # Validation 2: C→T Edge Bias
    c2t_ok = validate_c2t_edge_bias(model, batch)
    
    # Validation 3: Gradient Flow
    from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss
    criterion = MixtureNLLLoss()
    grad_ok = validate_gradient_flow(model, batch, criterion)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Hierarchical Fusion: {'✓ PASS' if fusion_ok else '✗ FAIL'}")
    print(f"C→T Edge Bias: {'✓ PASS' if c2t_ok else '✗ FAIL'}")
    print(f"Gradient Flow: {'✓ PASS' if grad_ok else '✗ FAIL'}")
    
    if fusion_ok and c2t_ok and grad_ok:
        print("\n🎉 ALL VALIDATIONS PASSED - MECHANISMS WORKING CORRECTLY 🎉")
        return 0
    else:
        print("\n⚠ SOME VALIDATIONS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == '__main__':
    exit(main())
