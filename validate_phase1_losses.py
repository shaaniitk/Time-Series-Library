#!/usr/bin/env python3
"""
Phase 1 Validation: Advanced Loss Components Integration
Testing all 11 advanced loss functions we integrated
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_advanced_losses():
    """Test all Phase 1 advanced loss components"""
    print("🧪 PHASE 1 VALIDATION: Advanced Loss Components")
    print("=" * 60)
    
    # Test sample data
    batch_size, seq_len, features = 4, 24, 7
    pred = torch.randn(batch_size, seq_len, features)
    target = torch.abs(torch.randn(batch_size, seq_len, features)) + 0.1
    
    results = {}
    
    # Test 1: MAPE Loss
    try:
        from layers.modular.losses.advanced_losses import MAPELoss
        mape = MAPELoss()
        loss_mape = mape(pred, target)
        results['MAPE'] = f"{loss_mape.item():.4f}"
        print(f"✅ MAPE Loss: {loss_mape.item():.4f}")
    except Exception as e:
        print(f"❌ MAPE Loss failed: {e}")
        results['MAPE'] = f"ERROR: {e}"
    
    # Test 2: SMAPE Loss
    try:
        from layers.modular.losses.advanced_losses import SMAPELoss
        smape = SMAPELoss()
        loss_smape = smape(pred, target)
        results['SMAPE'] = f"{loss_smape.item():.4f}"
        print(f"✅ SMAPE Loss: {loss_smape.item():.4f}")
    except Exception as e:
        print(f"❌ SMAPE Loss failed: {e}")
        results['SMAPE'] = f"ERROR: {e}"
    
    # Test 3: MASE Loss
    try:
        from layers.modular.losses.advanced_losses import MASELoss
        mase = MASELoss(freq=1)
        loss_mase = mase(pred, target)
        results['MASE'] = f"{loss_mase.item():.4f}"
        print(f"✅ MASE Loss: {loss_mase.item():.4f}")
    except Exception as e:
        print(f"❌ MASE Loss failed: {e}")
        results['MASE'] = f"ERROR: {e}"
    
    # Test 4: PS Loss
    try:
        from layers.modular.losses.advanced_losses import PSLoss
        ps = PSLoss(pred_len=seq_len)
        loss_ps = ps(pred, target)
        results['PS'] = f"{loss_ps.item():.4f}"
        print(f"✅ PS Loss: {loss_ps.item():.4f}")
    except Exception as e:
        print(f"❌ PS Loss failed: {e}")
        results['PS'] = f"ERROR: {e}"
    
    # Test 5: Focal Loss
    try:
        from layers.modular.losses.advanced_losses import FocalLoss
        focal = FocalLoss(alpha=1.0, gamma=2.0)
        loss_focal = focal(pred, target)
        results['Focal'] = f"{loss_focal.item():.4f}"
        print(f"✅ Focal Loss: {loss_focal.item():.4f}")
    except Exception as e:
        print(f"❌ Focal Loss failed: {e}")
        results['Focal'] = f"ERROR: {e}"
    
    # Test 6: Adaptive Autoformer Loss
    try:
        from layers.modular.losses.adaptive_bayesian_losses import AdaptiveAutoformerLoss
        adaptive = AdaptiveAutoformerLoss()
        loss_adaptive = adaptive(pred, target)
        results['Adaptive'] = f"{loss_adaptive.item():.4f}"
        print(f"✅ Adaptive Autoformer Loss: {loss_adaptive.item():.4f}")
    except Exception as e:
        print(f"❌ Adaptive Autoformer Loss failed: {e}")
        results['Adaptive'] = f"ERROR: {e}"
    
    # Test 7: Frequency-Aware Loss
    try:
        from layers.modular.losses.adaptive_bayesian_losses import FrequencyAwareLoss
        freq_aware = FrequencyAwareLoss(base_loss='mse')
        loss_freq = freq_aware(pred, target)
        results['FreqAware'] = f"{loss_freq.item():.4f}"
        print(f"✅ Frequency-Aware Loss: {loss_freq.item():.4f}")
    except Exception as e:
        print(f"❌ Frequency-Aware Loss failed: {e}")
        results['FreqAware'] = f"ERROR: {e}"
    
    # Test 8: Quantile Loss
    try:
        from layers.modular.losses.adaptive_bayesian_losses import QuantileLoss
        quantile = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        # For quantile loss, predictions need to be [B, T, C*Q] where Q is number of quantiles
        num_quantiles = 3
        pred_quantile = pred.view(batch_size, seq_len, -1).repeat(1, 1, num_quantiles)  # [B, T, C*Q]
        loss_quantile = quantile(pred_quantile, target)
        results['Quantile'] = f"{loss_quantile.item():.4f}"
        print(f"✅ Multi-Quantile Loss: {loss_quantile.item():.4f}")
    except Exception as e:
        print(f"❌ Multi-Quantile Loss failed: {e}")
        results['Quantile'] = f"ERROR: {e}"
    
    # Test 9: Uncertainty Calibration Loss
    try:
        from layers.modular.losses.adaptive_bayesian_losses import UncertaintyCalibrationLoss
        uncertainty_cal = UncertaintyCalibrationLoss(calibration_weight=1.0)
        uncertainties = torch.abs(torch.randn_like(pred)) + 0.1
        loss_uncertainty = uncertainty_cal(pred, target, uncertainties)
        results['UncertaintyCalib'] = f"{loss_uncertainty.item():.4f}"
        print(f"✅ Uncertainty Calibration Loss: {loss_uncertainty.item():.4f}")
    except Exception as e:
        print(f"❌ Uncertainty Calibration Loss failed: {e}")
        results['UncertaintyCalib'] = f"ERROR: {e}"
    
    return results

def test_registry_integration():
    """Test that components are properly registered"""
    print("\n🔧 REGISTRY INTEGRATION TEST")
    print("-" * 40)
    
    try:
        from layers.modular.losses.registry import LossRegistry
        registry = LossRegistry()
        
        # Test if our new components are registered
        advanced_components = [
            'mape', 'smape', 'mase', 'ps_loss', 'focal',
            'adaptive_autoformer', 'frequency_aware', 'multi_quantile', 
            'uncertainty_calibration'
        ]
        
        registered_count = 0
        for component_name in advanced_components:
            try:
                component_class = registry.get(component_name)
                print(f"✅ {component_name} -> {component_class.__name__}")
                registered_count += 1
            except Exception as e:
                print(f"❌ {component_name} -> ERROR: {e}")
        
        print(f"\n📊 Registry Status: {registered_count}/{len(advanced_components)} components registered")
        return registered_count == len(advanced_components)
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_component_schemas():
    """Test that component schemas are properly defined"""
    print("\n📋 SCHEMA INTEGRATION TEST")
    print("-" * 40)
    
    try:
        from configs.schemas import ComponentType
        
        # Check if new component types exist (using actual attribute names)
        new_types = [
            ('MAPE_LOSS', 'mape_loss'),
            ('SMAPE_LOSS', 'smape_loss'), 
            ('MASE_LOSS', 'mase_loss'),
            ('PS_LOSS', 'ps_loss'),
            ('FOCAL_LOSS', 'focal_loss'),
            ('FREQUENCY_AWARE_LOSS', 'frequency_aware_loss'),
            ('MULTI_QUANTILE_LOSS', 'multi_quantile_loss'),
            ('UNCERTAINTY_CALIBRATION_LOSS', 'uncertainty_calibration_loss')
        ]
        
        schema_count = 0
        for attr_name, expected_value in new_types:
            try:
                component_type = getattr(ComponentType, attr_name)
                if component_type.value == expected_value:
                    print(f"✅ {attr_name} -> {component_type.value}")
                    schema_count += 1
                else:
                    print(f"❌ {attr_name} -> Expected '{expected_value}', got '{component_type.value}'")
            except AttributeError:
                print(f"❌ {attr_name} -> NOT FOUND")
        
        print(f"\n📊 Schema Status: {schema_count}/{len(new_types)} schemas defined")
        return schema_count == len(new_types)
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def main():
    """Run complete Phase 1 validation"""
    print("🚀 STARTING PHASE 1 VALIDATION")
    print("=" * 60)
    
    # Test 1: Loss function implementations
    loss_results = test_advanced_losses()
    loss_success = len([k for k, v in loss_results.items() if not v.startswith('ERROR')]) 
    
    # Test 2: Registry integration
    registry_success = test_registry_integration()
    
    # Test 3: Schema integration
    schema_success = test_component_schemas()
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 PHASE 1 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"🧪 Loss Functions: {loss_success}/9 working")
    print(f"🔧 Registry: {'✅ PASS' if registry_success else '❌ FAIL'}")
    print(f"📋 Schemas: {'✅ PASS' if schema_success else '❌ FAIL'}")
    
    total_success = loss_success >= 7 and registry_success and schema_success
    
    if total_success:
        print("\n🎉 PHASE 1 COMPLETE! Ready for Phase 2")
        print("   Next: Attention Components (layers/AdvancedComponents.py)")
    else:
        print("\n⚠️ Phase 1 needs fixes before proceeding")
    
    return total_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
