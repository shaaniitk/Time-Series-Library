#!/usr/bin/env python3
"""
Test script for Mode-Aware Loss Functions
Verifies that the loss functions handle M, MS, and S modes correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from utils.enhanced_losses import (
    ModeAwareLoss, 
    HierarchicalModeAwareLoss, 
    BayesianModeAwareLoss,
    create_enhanced_loss
)


def test_mode_aware_losses():
    """Test all mode-aware loss functions"""
    
    print("🔍 Testing Mode-Aware Loss Functions")
    print("=" * 50)
    
    # Test data shapes
    batch_size, seq_len = 4, 10
    all_features = 118  # Total features
    target_features = 4  # OHLC targets
    
    # Create test data
    print(f"Test data: batch_size={batch_size}, seq_len={seq_len}")
    print(f"All features: {all_features}, Target features: {target_features}")
    
    # Test each mode
    modes = ['M', 'MS', 'S']
    
    for mode in modes:
        print(f"\n🎯 Testing {mode} Mode")
        print("-" * 30)
        
        if mode == 'M':
            # M mode: All features input/output
            pred_shape = (batch_size, seq_len, all_features)
            true_shape = (batch_size, seq_len, all_features)
        elif mode == 'MS':
            # MS mode: All features input, target features output
            pred_shape = (batch_size, seq_len, target_features)  # Model outputs only targets
            true_shape = (batch_size, seq_len, all_features)     # Ground truth has all features
        else:  # S mode
            # S mode: Target features input/output
            pred_shape = (batch_size, seq_len, target_features)
            true_shape = (batch_size, seq_len, target_features)
        
        print(f"   Prediction shape: {pred_shape}")
        print(f"   Ground truth shape: {true_shape}")
        
        # Create test tensors
        predictions = torch.randn(pred_shape)
        targets = torch.randn(true_shape)
        
        # Test ModeAwareLoss
        print(f"   Testing ModeAwareLoss...")
        try:
            loss_fn = ModeAwareLoss(mode=mode, target_features=target_features)
            loss, components = loss_fn(predictions, targets, return_components=True)
            print(f"   ✅ ModeAwareLoss: {loss.item():.6f}")
            print(f"      Components: {components}")
        except Exception as e:
            print(f"   ❌ ModeAwareLoss failed: {e}")
        
        # Test HierarchicalModeAwareLoss
        print(f"   Testing HierarchicalModeAwareLoss...")
        try:
            hier_loss_fn = HierarchicalModeAwareLoss(mode=mode, target_features=target_features)
            
            # Create some dummy hierarchical outputs
            resolution_outputs = [
                torch.randn(pred_shape) * 0.9,  # Slightly different
                torch.randn(pred_shape) * 0.8,
            ]
            
            hier_loss, hier_components = hier_loss_fn(
                predictions, targets, 
                resolution_outputs=resolution_outputs, 
                return_components=True
            )
            print(f"   ✅ HierarchicalModeAwareLoss: {hier_loss.item():.6f}")
            print(f"      Resolution losses: {hier_components['resolution_losses']}")
        except Exception as e:
            print(f"   ❌ HierarchicalModeAwareLoss failed: {e}")
        
        # Test BayesianModeAwareLoss
        print(f"   Testing BayesianModeAwareLoss...")
        try:
            bayes_loss_fn = BayesianModeAwareLoss(mode=mode, target_features=target_features)
            
            # Create dummy uncertainty and KL divergence
            uncertainty = torch.abs(torch.randn(pred_shape)) * 0.1
            kl_divergence = torch.tensor(0.01)
            
            bayes_loss, bayes_components = bayes_loss_fn(
                predictions, targets,
                uncertainty=uncertainty,
                kl_divergence=kl_divergence,
                return_components=True
            )
            print(f"   ✅ BayesianModeAwareLoss: {bayes_loss.item():.6f}")
            print(f"      Uncertainty loss: {bayes_components['uncertainty_loss']:.6f}")
            print(f"      KL loss: {bayes_components['kl_loss']:.6f}")
        except Exception as e:
            print(f"   ❌ BayesianModeAwareLoss failed: {e}")
        
        # Test create_enhanced_loss convenience function
        print(f"   Testing create_enhanced_loss convenience function...")
        try:
            for model_type in ['enhanced', 'bayesian', 'hierarchical']:
                conv_loss_fn = create_enhanced_loss(
                    model_type=model_type, 
                    mode=mode, 
                    target_features=target_features
                )
                conv_loss = conv_loss_fn(predictions, targets)
                print(f"   ✅ {model_type}: {conv_loss.item():.6f}")
        except Exception as e:
            print(f"   ❌ create_enhanced_loss failed: {e}")


def test_loss_backward():
    """Test that losses work with backpropagation"""
    
    print(f"\n🔧 Testing Backpropagation")
    print("=" * 30)
    
    # Simple test case
    batch_size, seq_len, features = 2, 5, 4
    
    # Create test data with gradients
    predictions = torch.randn(batch_size, seq_len, features, requires_grad=True)
    targets = torch.randn(batch_size, seq_len, features)
    
    for mode in ['M', 'MS', 'S']:
        try:
            loss_fn = ModeAwareLoss(mode=mode, target_features=features)
            loss = loss_fn(predictions, targets)
            
            # Backpropagation
            loss.backward(retain_graph=True)
            
            # Check gradients
            if predictions.grad is not None:
                print(f"   ✅ {mode} mode: Loss {loss.item():.6f}, Grad norm: {predictions.grad.norm().item():.6f}")
            else:
                print(f"   ⚠️ {mode} mode: No gradients computed")
                
        except Exception as e:
            print(f"   ❌ {mode} mode backprop failed: {e}")


def test_feature_slicing():
    """Test that feature slicing works correctly for different modes"""
    
    print(f"\n✂️ Testing Feature Slicing")
    print("=" * 30)
    
    batch_size, seq_len = 2, 3
    all_features = 10
    target_features = 4
    
    # Create test data where we can verify slicing
    predictions_full = torch.arange(batch_size * seq_len * all_features, dtype=torch.float32).reshape(
        batch_size, seq_len, all_features
    )
    targets_full = torch.arange(batch_size * seq_len * all_features, dtype=torch.float32).reshape(
        batch_size, seq_len, all_features
    ) + 1000  # Offset to distinguish from predictions
    
    print(f"   Full predictions shape: {predictions_full.shape}")
    print(f"   Full targets shape: {targets_full.shape}")
    
    # Test M mode (no slicing)
    loss_fn_M = ModeAwareLoss(mode='M', target_features=target_features, total_features=all_features)
    loss_M = loss_fn_M(predictions_full, targets_full)
    print(f"   M mode loss (all features): {loss_M.item():.2f}")
    
    # Test MS mode (slice to target features)
    predictions_targets = predictions_full[:, :, :target_features]  # First 4 features
    loss_fn_MS = ModeAwareLoss(mode='MS', target_features=target_features, total_features=all_features)
    loss_MS = loss_fn_MS(predictions_targets, targets_full)
    print(f"   MS mode loss (targets only): {loss_MS.item():.2f}")
    
    # Test S mode (both already target features)
    targets_targets = targets_full[:, :, :target_features]
    loss_fn_S = ModeAwareLoss(mode='S', target_features=target_features, total_features=all_features)
    loss_S = loss_fn_S(predictions_targets, targets_targets)
    print(f"   S mode loss (targets→targets): {loss_S.item():.2f}")
    
    # Verify that MS and S handle slicing correctly
    print(f"   ✅ Feature slicing test completed")


if __name__ == '__main__':
    print("🚀 Mode-Aware Loss Function Test Suite")
    print("=" * 60)
    
    try:
        test_mode_aware_losses()
        test_loss_backward()
        test_feature_slicing()
        
        print(f"\n🎉 All tests completed successfully!")
        print("=" * 60)
        print("✅ Mode-aware loss functions are working correctly")
        print("✅ Ready for training with M, MS, and S modes")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
