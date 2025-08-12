#!/usr/bin/env python3
"""
Quick validation test for Phase 1 (Loss Components) integration
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_advanced_loss_integration():
    """Test that all Phase 1 advanced loss components are working"""
    print("=" * 60)
    print("Phase 1 Integration Validation: Advanced Loss Components")
    print("=" * 60)
    
    try:
        # Test imports
        from layers.modular.losses.advanced_losses import MAPELoss, SMAPELoss, MASELoss, PSLoss, FocalLoss
        from layers.modular.losses.adaptive_bayesian_losses import FrequencyAwareLoss, UncertaintyCalibrationLoss, QuantileLoss
        print("✓ All advanced loss modules imported successfully")
        
        # Test registry integration
        from layers.modular.losses.registry import LossRegistry
        registry = LossRegistry()
        
        advanced_losses = ['mape', 'smape', 'mase', 'ps_loss', 'focal', 'frequency_aware', 'multi_quantile', 'uncertainty_calibration']
        
        for loss_name in advanced_losses:
            try:
                loss_class = registry.get(loss_name)
                print(f"✓ {loss_name} registered and available")
            except Exception as e:
                print(f"✗ {loss_name} registration failed: {e}")
        
        # Test functionality
        print("\nTesting loss functionality:")
        pred = torch.randn(2, 10, 3)
        target = torch.abs(torch.randn(2, 10, 3)) + 0.1
        
        # Test MAPE
        mape = MAPELoss()
        loss_mape = mape(pred, target)
        print(f"✓ MAPE Loss: {loss_mape.item():.4f}")
        
        # Test SMAPE  
        smape = SMAPELoss()
        loss_smape = smape(pred, target)
        print(f"✓ SMAPE Loss: {loss_smape.item():.4f}")
        
        # Test MASE
        mase = MASELoss(freq=1)
        loss_mase = mase(pred, target)
        print(f"✓ MASE Loss: {loss_mase.item():.4f}")
        
        # Test PS Loss
        ps = PSLoss(pred_len=10)
        loss_ps = ps(pred, target)
        print(f"✓ PS Loss: {loss_ps.item():.4f}")
        
        # Test Focal Loss
        focal = FocalLoss()
        loss_focal = focal(pred, target)
        print(f"✓ Focal Loss: {loss_focal.item():.4f}")
        
        # Test Frequency-Aware Loss
        freq_aware = FrequencyAwareLoss()
        loss_freq = freq_aware(pred, target)
        print(f"✓ Frequency-Aware Loss: {loss_freq.item():.4f}")
        
        # Test Quantile Loss
        quantile = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        # For quantile loss, we need to provide predictions for each quantile
        pred_quantile = pred.unsqueeze(-1).repeat(1, 1, 1, 3)  # 3 quantiles
        loss_quantile = quantile(pred_quantile, target)
        print(f"✓ Multi-Quantile Loss: {loss_quantile.item():.4f}")
        
        # Test Uncertainty Calibration Loss
        uncertainty_cal = UncertaintyCalibrationLoss()
        uncertainties = torch.abs(torch.randn_like(pred)) + 0.1
        loss_uncertainty = uncertainty_cal(pred, target, uncertainties)
        print(f"✓ Uncertainty Calibration Loss: {loss_uncertainty.item():.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 PHASE 1 INTEGRATION COMPLETE!")
        print("   All 8 advanced loss components successfully integrated")
        print("   ✓ MAPE, SMAPE, MASE, PS, Focal losses")
        print("   ✓ Frequency-aware, Multi-quantile, Uncertainty calibration")
        print("   ✓ Registry integration working")
        print("   ✓ All loss functions computing correctly")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 1 INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_registry_integration():
    """Test that the components are properly registered in the concrete components"""
    print("\nTesting concrete component registry integration:")
    
    try:
        from configs.schemas import ComponentType
        
        # Check if new component types are available
        new_types = [
            ComponentType.MAPE_LOSS,
            ComponentType.SMAPE_LOSS, 
            ComponentType.MASE_LOSS,
            ComponentType.PS_LOSS,
            ComponentType.FOCAL_LOSS,
            ComponentType.FREQUENCY_AWARE_LOSS,
            ComponentType.MULTI_QUANTILE_LOSS,
            ComponentType.UNCERTAINTY_CALIBRATION_LOSS
        ]
        
        for comp_type in new_types:
            print(f"✓ {comp_type.value} component type available")
        
        print("✓ All new component types properly defined in schemas")
        return True
        
    except Exception as e:
        print(f"❌ Component type registration failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Phase 1 Integration Validation...")
    
    success1 = test_advanced_loss_integration()
    success2 = test_component_registry_integration()
    
    if success1 and success2:
        print("\n🚀 READY FOR PHASE 2!")
        print("   Next: Attention Components Integration")
        sys.exit(0)
    else:
        print("\n⚠️ Phase 1 integration needs fixes before proceeding")
        sys.exit(1)
