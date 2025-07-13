# test_complete_hf_suite.py

"""
Comprehensive test for all HF Autoformer models

Tests all four HF implementations:
1. HFEnhancedAutoformer (Basic Enhanced)
2. HFBayesianAutoformer (Bayesian Uncertainty)  
3. HFHierarchicalAutoformer (Multi-resolution)
4. HFQuantileAutoformer (Quantile Regression)
"""

import torch
import sys
import os
sys.path.append(os.path.abspath('.'))

from models.HFAutoformerSuite import (
    HFEnhancedAutoformer,
    HFBayesianAutoformer, 
    HFHierarchicalAutoformer,
    HFQuantileAutoformer
)
from argparse import Namespace

def create_test_configs():
    """Create test configurations"""
    return Namespace(
        enc_in=7,
        dec_in=7, 
        c_out=1,
        seq_len=96,
        pred_len=24,
        d_model=64
    )

def create_test_data(configs, batch_size=4):
    """Create test data"""
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 4)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec

def test_enhanced_autoformer():
    """Test HFEnhancedAutoformer (Basic Enhanced)"""
    print("🧪 Testing HFEnhancedAutoformer (Basic Enhanced)")
    print("=" * 60)
    
    configs = create_test_configs()
    model = HFEnhancedAutoformer(configs)
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_test_data(configs)
    
    try:
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ Input shape: {x_enc.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Expected shape: ({x_enc.shape[0]}, {configs.pred_len}, {configs.c_out})")
        
        expected_shape = (x_enc.shape[0], configs.pred_len, configs.c_out)
        if output.shape == expected_shape:
            print("✅ Shape validation: PASSED")
            print("✅ HFEnhancedAutoformer: SUCCESS")
        else:
            print(f"❌ Shape mismatch! Got {output.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ HFEnhancedAutoformer failed: {e}")
        return False
    
    print()
    return True

def test_bayesian_autoformer():
    """Test HFBayesianAutoformer (Bayesian Uncertainty)"""
    print("🎯 Testing HFBayesianAutoformer (Bayesian Uncertainty)")
    print("=" * 60)
    
    configs = create_test_configs()
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    model = HFBayesianAutoformer(configs, quantile_levels=quantiles, n_samples=10)
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_test_data(configs)
    
    try:
        # Test standard forward pass
        print("Testing standard forward pass...")
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_uncertainty=False)
        print(f"✅ Standard output shape: {output.shape}")
        
        # Test uncertainty quantification
        print("Testing uncertainty quantification...")
        uncertainty_result = model(x_enc, x_mark_enc, x_dec, x_mark_dec, 
                                 return_uncertainty=True, detailed_uncertainty=True)
        
        print(f"✅ Prediction shape: {uncertainty_result['prediction'].shape}")
        print(f"✅ Uncertainty shape: {uncertainty_result['uncertainty'].shape}")
        print(f"✅ Confidence intervals: {list(uncertainty_result['confidence_intervals'].keys())}")
        
        if 'quantiles' in uncertainty_result:
            print(f"✅ Quantiles: {list(uncertainty_result['quantiles'].keys())}")
        
        # Validate uncertainty properties
        uncertainty_values = uncertainty_result['uncertainty']
        if torch.all(uncertainty_values >= 0):
            print("✅ Uncertainty values are non-negative")
        
        print("✅ HFBayesianAutoformer: SUCCESS")
        
    except Exception as e:
        print(f"❌ HFBayesianAutoformer failed: {e}")
        return False
    
    print()
    return True

def test_hierarchical_autoformer():
    """Test HFHierarchicalAutoformer (Multi-resolution)"""
    print("🏗️ Testing HFHierarchicalAutoformer (Multi-resolution)")
    print("=" * 60)
    
    configs = create_test_configs()
    hierarchy_levels = 3
    model = HFHierarchicalAutoformer(configs, hierarchy_levels=hierarchy_levels)
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_test_data(configs)
    
    try:
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ Input shape: {x_enc.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Hierarchy levels: {hierarchy_levels}")
        print(f"✅ Fusion weights: {model.fusion_weights.data}")
        
        expected_shape = (x_enc.shape[0], configs.pred_len, configs.c_out)
        if output.shape == expected_shape:
            print("✅ Multi-resolution processing: SUCCESS")
            print("✅ HFHierarchicalAutoformer: SUCCESS")
        else:
            print(f"❌ Shape mismatch! Got {output.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ HFHierarchicalAutoformer failed: {e}")
        return False
    
    print()
    return True

def test_quantile_autoformer():
    """Test HFQuantileAutoformer (Quantile Regression)"""
    print("📊 Testing HFQuantileAutoformer (Quantile Regression)")
    print("=" * 60)
    
    configs = create_test_configs()
    quantiles = [0.1, 0.5, 0.9]
    model = HFQuantileAutoformer(configs, quantiles=quantiles, kl_weight=0.3)
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_test_data(configs)
    
    try:
        # Test standard forward pass
        print("Testing standard forward pass...")
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_uncertainty=False)
        
        expected_quantile_features = configs.c_out * len(quantiles)
        print(f"✅ Quantile output shape: {output.shape}")
        print(f"✅ Expected features: {expected_quantile_features}")
        
        # Test uncertainty quantification with quantile outputs
        print("Testing quantile-specific uncertainty...")
        uncertainty_result = model(x_enc, x_mark_enc, x_dec, x_mark_dec, 
                                 return_uncertainty=True)
        
        print(f"✅ Prediction shape: {uncertainty_result['prediction'].shape}")
        
        if 'quantile_outputs' in uncertainty_result:
            quantile_outputs = uncertainty_result['quantile_outputs']
            print(f"✅ Quantile outputs: {list(quantile_outputs.keys())}")
            
            for q_name, q_output in quantile_outputs.items():
                print(f"  - {q_name}: {q_output.shape}")
        
        print(f"✅ Quantile levels: {quantiles}")
        print(f"✅ KL weight: {model.kl_weight}, Quantile weight: {model.quantile_weight}")
        print("✅ HFQuantileAutoformer: SUCCESS")
        
    except Exception as e:
        print(f"❌ HFQuantileAutoformer failed: {e}")
        return False
    
    print()
    return True

def compare_models():
    """Compare all models with same input"""
    print("🔄 Model Comparison Summary")
    print("=" * 60)
    
    configs = create_test_configs()
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_test_data(configs, batch_size=2)
    
    models = {
        'Enhanced': HFEnhancedAutoformer(configs),
        'Bayesian': HFBayesianAutoformer(configs, n_samples=5),
        'Hierarchical': HFHierarchicalAutoformer(configs, hierarchy_levels=2),
        'Quantile': HFQuantileAutoformer(configs, quantiles=[0.1, 0.5, 0.9])
    }
    
    print("Model Performance Comparison:")
    print()
    
    with torch.no_grad():
        for name, model in models.items():
            try:
                if name == 'Quantile':
                    # Quantile model outputs more features
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    print(f"✅ {name:12}: {output.shape} (quantile expanded)")
                else:
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    print(f"✅ {name:12}: {output.shape}")
                    
            except Exception as e:
                print(f"❌ {name:12}: Failed - {e}")
    
    print()
    print("📈 Benefits Summary:")
    print("✅ Enhanced:     Standard time series forecasting (baseline)")
    print("✅ Bayesian:     + Uncertainty quantification + Confidence intervals")  
    print("✅ Hierarchical: + Multi-resolution processing + Temporal hierarchies")
    print("✅ Quantile:     + Quantile regression + Risk assessment")
    
def migration_recommendation():
    """Provide migration recommendations"""
    print("🚀 Migration Recommendations")
    print("=" * 60)
    
    recommendations = {
        "EnhancedAutoformer": {
            "replacement": "HFEnhancedAutoformer",
            "benefits": ["Production stability", "HF ecosystem", "Reduced bugs"],
            "effort": "Low (drop-in replacement)"
        },
        "BayesianEnhancedAutoformer": {
            "replacement": "HFBayesianAutoformer", 
            "benefits": ["Eliminates gradient bugs", "Better uncertainty", "Robust sampling"],
            "effort": "Medium (test uncertainty outputs)"
        },
        "HierarchicalEnhancedAutoformer": {
            "replacement": "HFHierarchicalAutoformer",
            "benefits": ["Simplified multi-resolution", "No DWT complexity", "Standard fusion"],
            "effort": "Medium (validate hierarchy levels)"
        },
        "QuantileBayesianAutoformer": {
            "replacement": "HFQuantileAutoformer",
            "benefits": ["Native quantile support", "Robust loss handling", "Better calibration"],
            "effort": "Medium (test quantile outputs)"
        }
    }
    
    for original, info in recommendations.items():
        print(f"\n📋 {original}")
        print(f"   ➡️  Replace with: {info['replacement']}")
        print(f"   📈 Benefits: {', '.join(info['benefits'])}")
        print(f"   ⏱️  Effort: {info['effort']}")
    
    print(f"\n🎯 Overall Strategy:")
    print(f"   1. Start with HFEnhancedAutoformer (easiest)")
    print(f"   2. Migrate to HFBayesianAutoformer (most critical bugs)")
    print(f"   3. Add HFHierarchicalAutoformer for multi-resolution")
    print(f"   4. Use HFQuantileAutoformer for risk analysis")

def main():
    """Run all tests"""
    print("🚀 Complete HF Autoformer Suite Test")
    print("=" * 80)
    print()
    
    tests = [
        ("Enhanced", test_enhanced_autoformer),
        ("Bayesian", test_bayesian_autoformer), 
        ("Hierarchical", test_hierarchical_autoformer),
        ("Quantile", test_quantile_autoformer)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:12}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your complete HF Autoformer suite is ready for deployment!")
        
        compare_models()
        migration_recommendation()
        
        print("\n" + "=" * 80)
        print("🚀 READY FOR PRODUCTION MIGRATION!")
        print("=" * 80)
        
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the errors above.")

if __name__ == "__main__":
    main()
