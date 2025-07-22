#!/usr/bin/env python3
"""
Individual Component Test Scripts Generator

Creates individual test scripts for each component type to enable
focused testing of specific components.
"""

import os
from pathlib import Path

# Create scripts directory if it doesn't exist
scripts_dir = Path(__file__).parent
scripts_dir.mkdir(exist_ok=True)

def create_attention_test_script():
    """Create test script for attention components."""
    content = '''#!/usr/bin/env python3
"""
Attention Components Test Script

Tests all attention mechanisms in the modular framework:
- AUTOCORRELATION_LAYER
- ADAPTIVE_AUTOCORRELATION_LAYER  
- CROSS_RESOLUTION
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType

def test_attention_components():
    """Test all attention components."""
    print("=== ATTENTION COMPONENTS TEST ===")
    
    # Ensure components are registered
    register_all_components()
    
    # Test parameters
    batch_size = 2
    seq_len = 96
    d_model = 64
    n_heads = 8
    
    test_input = torch.randn(batch_size, seq_len, d_model)
    test_cross = torch.randn(batch_size, seq_len, d_model)
    
    attention_types = [
        ComponentType.AUTOCORRELATION_LAYER,
        ComponentType.ADAPTIVE_AUTOCORRELATION_LAYER,
        ComponentType.CROSS_RESOLUTION
    ]
    
    results = {}
    
    for attention_type in attention_types:
        print(f"\\nTesting {attention_type.value}...")
        
        try:
            # Get component
            component_info = component_registry.get_component(attention_type)
            assert component_info is not None
            
            # Create instance
            params = {
                'd_model': d_model,
                'n_heads': n_heads,
                'dropout': 0.1,
                'factor': 1
            }
            
            component = component_info.factory(**params)
            
            # Test forward pass
            with torch.no_grad():
                if attention_type == ComponentType.CROSS_RESOLUTION:
                    output, weights = component(test_input, test_cross, test_cross)
                else:
                    output, weights = component(test_input, test_input, test_input)
            
            # Validate
            assert output.shape == test_input.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            print(f"✅ {attention_type.value} - PASSED")
            results[attention_type.value] = "PASSED"
            
        except Exception as e:
            print(f"❌ {attention_type.value} - FAILED: {e}")
            results[attention_type.value] = f"FAILED: {e}"
    
    # Summary
    print(f"\\n=== ATTENTION COMPONENTS SUMMARY ===")
    passed = sum(1 for status in results.values() if status == "PASSED")
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for comp_type, status in results.items():
        icon = "✅" if status == "PASSED" else "❌"
        print(f"{icon} {comp_type}: {status}")
    
    return passed == total

if __name__ == "__main__":
    success = test_attention_components()
    sys.exit(0 if success else 1)
'''
    
    with open(scripts_dir / "test_attention_components.py", "w", encoding='utf-8') as f:
        f.write(content)

def create_decomposition_test_script():
    """Create test script for decomposition components."""
    content = '''#!/usr/bin/env python3
"""
Decomposition Components Test Script

Tests all decomposition mechanisms in the modular framework:
- SERIES_DECOMP
- STABLE_DECOMP
- LEARNABLE_DECOMP
- WAVELET_DECOMP
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType

def test_decomposition_components():
    """Test all decomposition components."""
    print("=== DECOMPOSITION COMPONENTS TEST ===")
    
    # Ensure components are registered
    register_all_components()
    
    # Test parameters
    batch_size = 2
    seq_len = 96
    d_model = 64
    
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    decomp_types = [
        ComponentType.SERIES_DECOMP,
        ComponentType.STABLE_DECOMP,
        ComponentType.LEARNABLE_DECOMP,
        ComponentType.WAVELET_DECOMP
    ]
    
    results = {}
    
    for decomp_type in decomp_types:
        print(f"\\nTesting {decomp_type.value}...")
        
        try:
            # Get component
            component_info = component_registry.get_component(decomp_type)
            assert component_info is not None
            
            # Create instance with appropriate parameters
            if decomp_type in [ComponentType.SERIES_DECOMP, ComponentType.STABLE_DECOMP]:
                params = {'kernel_size': 25}
            elif decomp_type == ComponentType.LEARNABLE_DECOMP:
                params = {'input_dim': d_model}
            elif decomp_type == ComponentType.WAVELET_DECOMP:
                params = {'seq_len': seq_len, 'd_model': d_model}
            
            component = component_info.factory(**params)
            
            # Test forward pass
            with torch.no_grad():
                seasonal, trend = component(test_input)
            
            # Validate
            assert seasonal.shape == test_input.shape
            assert trend.shape == test_input.shape
            assert not torch.isnan(seasonal).any()
            assert not torch.isnan(trend).any()
            
            # Test reconstruction
            reconstruction = seasonal + trend
            assert reconstruction.shape == test_input.shape
            
            print(f"✅ {decomp_type.value} - PASSED")
            results[decomp_type.value] = "PASSED"
            
        except Exception as e:
            print(f"❌ {decomp_type.value} - FAILED: {e}")
            results[decomp_type.value] = f"FAILED: {e}"
    
    # Summary
    print(f"\\n=== DECOMPOSITION COMPONENTS SUMMARY ===")
    passed = sum(1 for status in results.values() if status == "PASSED")
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for comp_type, status in results.items():
        icon = "✅" if status == "PASSED" else "❌"
        print(f"{icon} {comp_type}: {status}")
    
    return passed == total

if __name__ == "__main__":
    success = test_decomposition_components()
    sys.exit(0 if success else 1)
'''
    
    with open(scripts_dir / "test_decomposition_components.py", "w", encoding='utf-8') as f:
        f.write(content)

def create_unified_factory_test_script():
    """Create test script for unified factory."""
    content = '''#!/usr/bin/env python3
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
    print("\\nTesting model availability listing...")
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
    print("\\nTesting custom model creation...")
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
    print("\\nTesting HF model creation...")
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
    print("\\nTesting framework auto-detection...")
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
    print("\\nTesting compatible pair creation...")
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
    print("\\nTesting prediction interface...")
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
    print(f"\\n=== UNIFIED FACTORY SUMMARY ===")
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
'''
    
    with open(scripts_dir / "test_unified_factory.py", "w", encoding='utf-8') as f:
        f.write(content)

def create_master_test_runner():
    """Create master test runner script."""
    content = '''#!/usr/bin/env python3
"""
Master Test Runner for Modular Autoformer Framework

Runs all test categories in sequence:
1. Individual component tests
2. Integration tests  
3. Factory tests
4. Performance tests
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_script(script_path, description):
    """Run a test script and return success status."""
    print(f"\\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, text=True)
        success = result.returncode == 0
        
        if success:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
        
        return success
        
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run complete test suite."""
    print("="*80)
    print("MODULAR AUTOFORMER FRAMEWORK - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    start_time = time.time()
    scripts_dir = Path(__file__).parent
    
    # Define test sequence
    test_sequence = [
        (scripts_dir / "test_attention_components.py", "Attention Components"),
        (scripts_dir / "test_decomposition_components.py", "Decomposition Components"),
        (scripts_dir / "test_unified_factory.py", "Unified Factory"),
        (scripts_dir / "test_all_components.py", "All Components Integration"),
        (scripts_dir / "test_all_integrations.py", "Full Integration Suite"),
    ]
    
    results = {}
    
    # Run each test script
    for script_path, description in test_sequence:
        if script_path.exists():
            success = run_script(script_path, description)
            results[description] = success
        else:
            print(f"⚠️  {description} - SKIPPED (script not found: {script_path})")
            results[description] = None
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print final summary
    print(f"\\n{'='*80}")
    print("FINAL TEST SUMMARY")
    print('='*80)
    
    passed = sum(1 for success in results.values() if success is True)
    failed = sum(1 for success in results.values() if success is False)
    skipped = sum(1 for success in results.values() if success is None)
    total = len(results)
    
    print(f"Total test suites: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Duration: {duration:.2f} seconds")
    
    print(f"\\nDetailed Results:")
    for description, success in results.items():
        if success is True:
            print(f"✅ {description}")
        elif success is False:
            print(f"❌ {description}")
        else:
            print(f"⚠️  {description} (skipped)")
    
    # Overall result
    if failed == 0 and passed > 0:
        print(f"\\n🎉 ALL TEST SUITES PASSED! 🎉")
        print("✨ Modular autoformer framework is fully functional! ✨")
        return 0
    elif failed > 0:
        print(f"\\n❌ {failed} TEST SUITE(S) FAILED")
        return 1
    else:
        print(f"\\n⚠️  NO TESTS WERE RUN")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(scripts_dir / "run_all_tests.py", "w", encoding='utf-8') as f:
        f.write(content)

# Create all test scripts
print("Creating individual component test scripts...")

create_attention_test_script()
print("✓ Created test_attention_components.py")

create_decomposition_test_script() 
print("✓ Created test_decomposition_components.py")

create_unified_factory_test_script()
print("✓ Created test_unified_factory.py")

create_master_test_runner()
print("✓ Created run_all_tests.py")

print("\n🎉 All test scripts created successfully!")
print("\nAvailable test scripts:")
print("- test_attention_components.py")
print("- test_decomposition_components.py") 
print("- test_unified_factory.py")
print("- test_all_components.py")
print("- test_all_integrations.py")
print("- run_all_tests.py (master runner)")
