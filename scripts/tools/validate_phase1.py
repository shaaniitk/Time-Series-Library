#!/usr/bin/env python3
"""
Phase 1 Advanced Loss Components Validation Test

This test validates that all our Phase 1 advanced loss components are working correctly.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_advanced_losses_direct():
    """Test advanced loss components directly"""
    print("🧪 Testing Advanced Loss Components (Phase 1)")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 0
    
    # Create sample data
    batch_size, seq_len, features = 4, 24, 7
    predictions = torch.randn(batch_size, seq_len, features)
    targets = torch.abs(torch.randn(batch_size, seq_len, features)) + 0.1  # Positive targets for MAPE/SMAPE
    
    # Test 1: MAPE Loss
    total_tests += 1
    try:
        from layers.modular.losses.advanced_losses import MAPELoss
        mape = MAPELoss()
        loss = mape(predictions, targets)
        
        # Validate loss properties
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "MAPE should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        # Test perfect prediction
        perfect_loss = mape(targets, targets)
        assert perfect_loss.item() < 1e-3, f"Perfect prediction should be ~0, got {perfect_loss.item()}"
        
        print(f"✅ MAPE Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ MAPE Loss: FAILED - {e}")
    
    # Test 2: SMAPE Loss
    total_tests += 1
    try:
        from layers.modular.losses.advanced_losses import SMAPELoss
        smape = SMAPELoss()
        loss = smape(predictions, targets)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "SMAPE should be non-negative"
        assert loss.item() <= 200, "SMAPE should be bounded by 200%"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        # Test symmetry
        loss_ab = smape(predictions, targets)
        loss_ba = smape(targets, predictions)
        assert abs(loss_ab.item() - loss_ba.item()) < 1e-3, "SMAPE should be symmetric"
        
        print(f"✅ SMAPE Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ SMAPE Loss: FAILED - {e}")
    
    # Test 3: MASE Loss
    total_tests += 1
    try:
        from layers.modular.losses.advanced_losses import MASELoss
        mase = MASELoss(freq=1)
        loss = mase(predictions, targets)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "MASE should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print(f"✅ MASE Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ MASE Loss: FAILED - {e}")
    
    # Test 4: PS (Patch-wise Structural) Loss
    total_tests += 1
    try:
        from layers.modular.losses.advanced_losses import PSLoss
        ps = PSLoss(pred_len=seq_len)
        loss = ps(predictions, targets)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "PS Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print(f"✅ PS Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ PS Loss: FAILED - {e}")
    
    # Test 5: Focal Loss
    total_tests += 1
    try:
        from layers.modular.losses.advanced_losses import FocalLoss
        focal = FocalLoss(alpha=1.0, gamma=2.0)
        loss = focal(predictions, targets)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "Focal Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print(f"✅ Focal Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ Focal Loss: FAILED - {e}")
    
    # Test 6: Frequency-Aware Loss
    total_tests += 1
    try:
        from layers.modular.losses.adaptive_bayesian_losses import FrequencyAwareLoss
        freq_aware = FrequencyAwareLoss(base_loss='mse')
        loss = freq_aware(predictions, targets)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "Frequency-Aware Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print(f"✅ Frequency-Aware Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ Frequency-Aware Loss: FAILED - {e}")
    
    # Test 7: Multi-Quantile Loss
    total_tests += 1
    try:
        from layers.modular.losses.adaptive_bayesian_losses import QuantileLoss
        quantile = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        # For quantile loss, predictions need additional dimension for each quantile
        pred_quantile = predictions.unsqueeze(-1).repeat(1, 1, 1, 3)  # 3 quantiles
        loss = quantile(pred_quantile, targets)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "Multi-Quantile Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print(f"✅ Multi-Quantile Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ Multi-Quantile Loss: FAILED - {e}")
    
    # Test 8: Uncertainty Calibration Loss
    total_tests += 1
    try:
        from layers.modular.losses.adaptive_bayesian_losses import UncertaintyCalibrationLoss
        uncertainty_cal = UncertaintyCalibrationLoss(calibration_weight=1.0)
        uncertainties = torch.abs(torch.randn_like(predictions)) + 0.1
        loss = uncertainty_cal(predictions, targets, uncertainties)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "Uncertainty Calibration Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print(f"✅ Uncertainty Calibration Loss: {loss.item():.4f} - PASSED")
        passed_tests += 1
        
    except Exception as e:
        print(f"❌ Uncertainty Calibration Loss: FAILED - {e}")
    
    return passed_tests, total_tests

def test_loss_registry():
    """Test that losses are properly registered"""
    print("\n🔧 Testing Loss Registry Integration")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 0
    
    try:
        from layers.modular.losses.registry import LossRegistry
        registry = LossRegistry()
        
        # Test advanced loss registrations
        advanced_losses = [
            'mape', 'smape', 'mase', 'ps_loss', 'focal',
            'frequency_aware', 'multi_quantile', 'uncertainty_calibration'
        ]
        
        for loss_name in advanced_losses:
            total_tests += 1
            try:
                loss_class = registry.get(loss_name)
                assert loss_class is not None, f"Loss {loss_name} should be registered"
                print(f"✅ {loss_name}: Registered and available")
                passed_tests += 1
            except Exception as e:
                print(f"❌ {loss_name}: Registration failed - {e}")
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
    
    return passed_tests, total_tests

def test_component_types():
    """Test that component types are properly defined"""
    print("\n📋 Testing Component Type Definitions")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 0
    
    try:
        from configs.schemas import ComponentType
        
        # Test new component types
        new_types = [
            'MAPE_LOSS', 'SMAPE_LOSS', 'MASE_LOSS', 'PS_LOSS', 'FOCAL_LOSS',
            'FREQUENCY_AWARE_LOSS', 'MULTI_QUANTILE_LOSS', 'UNCERTAINTY_CALIBRATION_LOSS'
        ]
        
        for type_name in new_types:
            total_tests += 1
            try:
                comp_type = getattr(ComponentType, type_name)
                assert comp_type is not None, f"Component type {type_name} should exist"
                print(f"✅ {type_name}: {comp_type.value}")
                passed_tests += 1
            except AttributeError:
                print(f"❌ {type_name}: Component type not found")
            except Exception as e:
                print(f"❌ {type_name}: Error - {e}")
        
    except Exception as e:
        print(f"❌ Component type test failed: {e}")
    
    return passed_tests, total_tests

def run_all_tests():
    """Run all Phase 1 validation tests"""
    print("🚀 Phase 1 Advanced Loss Components - Full Validation")
    print("=" * 80)
    
    # Run all test suites
    passed1, total1 = test_advanced_losses_direct()
    passed2, total2 = test_loss_registry() 
    passed3, total3 = test_component_types()
    
    # Calculate overall results
    total_passed = passed1 + passed2 + passed3
    total_tests = total1 + total2 + total3
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 PHASE 1 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Advanced Loss Components: {passed1}/{total1} ({'✅' if passed1 == total1 else '❌'})")
    print(f"Registry Integration:     {passed2}/{total2} ({'✅' if passed2 == total2 else '❌'})")
    print(f"Component Type Schemas:   {passed3}/{total3} ({'✅' if passed3 == total3 else '❌'})")
    print("-" * 80)
    print(f"TOTAL PASSED:            {total_passed}/{total_tests}")
    print(f"SUCCESS RATE:            {success_rate:.1f}%")
    
    if total_passed == total_tests:
        print("\n🎉 PHASE 1 INTEGRATION: COMPLETE SUCCESS!")
        print("   All advanced loss components working correctly")
        print("   Ready to proceed with Phase 2 (Attention Components)")
        return True
    else:
        print(f"\n⚠️  PHASE 1 INTEGRATION: PARTIAL SUCCESS ({success_rate:.1f}%)")
        print("   Some components need attention before proceeding")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
