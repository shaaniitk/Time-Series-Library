#!/usr/bin/env python3
"""
Unified Factory Test Script

Tests the unified factory functionality:
- Model creation with different frameworks
- Framework preference handling
- Config completion
- Compatible pair creation
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.unified_autoformer_factory import (
    UnifiedAutoformerFactory,
    UnifiedModelInterface,
    create_autoformer,
    compare_implementations,
    list_available_models
)

def test_unified_factory():
    """Test unified factory functionality."""
    print("=== UNIFIED FACTORY TEST ===")
    
    # Basic test config
    config = {
        'seq_len': 96,
        'pred_len': 24,
        'label_len': 48,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 64,
        'n_heads': 8,
        'd_ff': 256,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'activation': 'gelu'
    }
    
    results = {}
    
    # Test 1: Model availability listing
    print("\nTesting model availability listing...")
    try:
        available_models = list_available_models()
        assert isinstance(available_models, dict)
        assert 'custom' in available_models
        assert 'hf' in available_models
        
        custom_count = len(available_models['custom'])
        hf_count = len(available_models['hf'])
        
        print(f"✅ Model listing - Custom: {custom_count}, HF: {hf_count}")
        results['model_listing'] = "PASSED"
        
    except Exception as e:
        print(f"❌ Model listing - FAILED: {e}")
        results['model_listing'] = f"FAILED: {e}"
    
    # Test 2: Custom model creation
    print("\nTesting custom model creation...")
    try:
        model = create_autoformer('enhanced', config, framework='custom')
        assert isinstance(model, UnifiedModelInterface)
        
        model_info = model.get_model_info()
        assert 'framework_type' in model_info
        
        print(f"✅ Custom model creation - Framework: {model_info['framework_type']}")
        results['custom_creation'] = "PASSED"
        
    except Exception as e:
        print(f"❌ Custom model creation - FAILED: {e}")
        results['custom_creation'] = f"FAILED: {e}"
    
    # Test 3: HF model creation
    print("\nTesting HF model creation...")
    try:
        model = create_autoformer('hf_enhanced', config, framework='hf')
        assert isinstance(model, UnifiedModelInterface)
        
        model_info = model.get_model_info()
        assert 'framework_type' in model_info
        
        print(f"✅ HF model creation - Framework: {model_info['framework_type']}")
        results['hf_creation'] = "PASSED"
        
    except Exception as e:
        print(f"❌ HF model creation - FAILED: {e}")
        results['hf_creation'] = f"FAILED: {e}"
    
    # Test 4: Framework auto-detection
    print("\nTesting framework auto-detection...")
    try:
        # Should detect custom
        model_custom = create_autoformer('enhanced', config, framework='auto')
        # Should detect HF
        model_hf = create_autoformer('hf_enhanced', config, framework='auto')
        
        assert isinstance(model_custom, UnifiedModelInterface)
        assert isinstance(model_hf, UnifiedModelInterface)
        
        print("✅ Framework auto-detection")
        results['auto_detection'] = "PASSED"
        
    except Exception as e:
        print(f"❌ Framework auto-detection - FAILED: {e}")
        results['auto_detection'] = f"FAILED: {e}"
    
    # Test 5: Compatible pair creation
    print("\nTesting compatible pair creation...")
    try:
        comparison = compare_implementations('enhanced', config)
        assert isinstance(comparison, dict)
        assert len(comparison) > 0
        
        frameworks = list(comparison.keys())
        print(f"✅ Compatible pairs - Frameworks: {frameworks}")
        results['compatible_pairs'] = "PASSED"
        
    except Exception as e:
        print(f"❌ Compatible pairs - FAILED: {e}")
        results['compatible_pairs'] = f"FAILED: {e}"
    
    # Test 6: Prediction interface
    print("\nTesting prediction interface...")
    try:
        model = create_autoformer('enhanced', config, framework='custom')
        
        # Create test data
        batch_size = 2
        x_enc = torch.randn(batch_size, config['seq_len'], config['enc_in'])
        x_mark_enc = torch.randn(batch_size, config['seq_len'], 4)
        x_dec = torch.randn(batch_size, config['label_len'] + config['pred_len'], config['dec_in'])
        x_mark_dec = torch.randn(batch_size, config['label_len'] + config['pred_len'], 4)
        
        # Test prediction
        with torch.no_grad():
            prediction = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (batch_size, config['pred_len'], config['c_out'])
        assert prediction.shape == expected_shape
        assert not torch.isnan(prediction).any()
        
        print(f"✅ Prediction interface - Shape: {prediction.shape}")
        results['prediction_interface'] = "PASSED"
        
    except Exception as e:
        print(f"❌ Prediction interface - FAILED: {e}")
        results['prediction_interface'] = f"FAILED: {e}"
    
    # Summary
    print(f"\n=== UNIFIED FACTORY SUMMARY ===")
    passed = sum(1 for status in results.values() if status == "PASSED")
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, status in results.items():
        icon = "✅" if status == "PASSED" else "❌"
        print(f"{icon} {test_name}: {status}")
    
    return passed == total

if __name__ == "__main__":
    success = test_unified_factory()
    sys.exit(0 if success else 1)
